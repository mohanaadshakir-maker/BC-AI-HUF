"""
Train real intrusion-detection models on the preprocessed NSL-KDD data and
compute genuine performance metrics: accuracy, AUC-ROC, false-alarm rate
(false positive rate on normal traffic), and per-sample detection time.

Two models are trained:
  1. XGBoost gradient-boosted trees  -- strong tabular baseline, also used
     as the model SHAP explains (TreeExplainer) and LIME explains (as a
     generic black box).
  2. A small feed-forward net with a self-attention layer over the input
     features -- differentiable, so it supports both a genuine "Attention"
     interpretability channel (the learned attention weights) and Integrated
     Gradients via Captum.

All metrics are computed on the held-out KDDTest+ set, which NSL-KDD keeps
disjoint from KDDTrain+ specifically so reported numbers reflect
generalisation rather than memorisation.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              f1_score, recall_score, precision_score)
import xgboost as xgb

DATA_DIR = "/home/claude/bc_ai_huf_simulation/data/processed"
OUT_DIR = "/home/claude/bc_ai_huf_simulation/results"
MODEL_DIR = "/home/claude/bc_ai_huf_simulation/ai_layer/saved_models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42


def load_data():
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(DATA_DIR, "y_train_bin.parquet"))["y"]
    y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test_bin.parquet"))["y"]
    return X_train, X_test, y_train, y_test


def false_alarm_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def train_xgb(X_train, y_train, X_test, y_test, seed=SEED):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=seed, n_jobs=4,
    )
    model.fit(X_train, y_train)

    # Detection time: mean per-sample inference latency, measured on this
    # evaluation machine (single-sample calls to mirror real-time scoring
    # rather than batched throughput).
    sample = X_test.sample(n=min(500, len(X_test)), random_state=seed)
    t0 = time.perf_counter()
    for i in range(len(sample)):
        _ = model.predict(sample.iloc[[i]])
    t1 = time.perf_counter()
    per_sample_ms = (t1 - t0) / len(sample) * 1000

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "XGBoost",
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "false_alarm_rate": false_alarm_rate(y_test, y_pred),
        "mean_detection_time_ms_per_sample": per_sample_ms,
        "n_test": len(y_test),
    }
    model.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    return model, metrics, y_pred, y_proba


class AttentionNet(nn.Module):
    """Feed-forward net with a learned per-feature attention/gating layer.

    The attention weights are a genuine model component (not a post-hoc
    add-on): a sigmoid gate over the input features is learned jointly with
    the classifier, so the resulting weights reflect what the trained model
    itself relies on -- this is what gets reported as the "Attention"
    interpretability channel.
    """

    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, return_attention=False):
        a = self.attention(x)
        gated = x * a
        out = self.net(gated)
        if return_attention:
            return out, a
        return out


def train_attention_net(X_train, y_train, X_test, y_test, seed=SEED, epochs=15):
    torch.manual_seed(seed)
    n_features = X_train.shape[1]
    model = AttentionNet(n_features)

    Xt = torch.tensor(X_train.values, dtype=torch.float32)
    yt = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_test.values, dtype=torch.float32)
    yv = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    batch_size = 1024
    n = Xt.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 3 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch}: train loss {total_loss / n:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xv)
        proba = torch.sigmoid(logits).numpy().ravel()
    y_pred = (proba >= 0.5).astype(int)

    sample_idx = np.random.RandomState(seed).choice(len(Xv), size=min(500, len(Xv)), replace=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in sample_idx:
            _ = model(Xv[i:i + 1])
    t1 = time.perf_counter()
    per_sample_ms = (t1 - t0) / len(sample_idx) * 1000

    metrics = {
        "model": "AttentionNet",
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "false_alarm_rate": false_alarm_rate(y_test, y_pred),
        "mean_detection_time_ms_per_sample": per_sample_ms,
        "n_test": len(y_test),
    }
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "attention_net.pt"))
    return model, metrics, y_pred, proba


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Data loaded:", X_train.shape, X_test.shape)

    print("\n=== Training XGBoost ===")
    xgb_model, xgb_metrics, xgb_pred, xgb_proba = train_xgb(X_train, y_train, X_test, y_test)
    print(json.dumps(xgb_metrics, indent=2))

    print("\n=== Training AttentionNet ===")
    att_model, att_metrics, att_pred, att_proba = train_attention_net(X_train, y_train, X_test, y_test)
    print(json.dumps(att_metrics, indent=2))

    results = {"xgboost": xgb_metrics, "attention_net": att_metrics}
    with open(os.path.join(OUT_DIR, "ai_layer_core_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions for downstream zero-day analysis
    np.save(os.path.join(OUT_DIR, "xgb_pred.npy"), xgb_pred)
    np.save(os.path.join(OUT_DIR, "xgb_proba.npy"), xgb_proba)
    np.save(os.path.join(OUT_DIR, "att_pred.npy"), att_pred)
    np.save(os.path.join(OUT_DIR, "att_proba.npy"), att_proba)

    print("\nSaved core metrics to results/ai_layer_core_metrics.json")
