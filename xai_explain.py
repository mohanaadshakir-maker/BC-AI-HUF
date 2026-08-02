"""
Generate real explanations from four interpretability methods on the trained
NSL-KDD models:
  - SHAP        (TreeExplainer on the XGBoost model)
  - LIME        (LimeTabularExplainer, model-agnostic, on the XGBoost model)
  - Attention   (the AttentionNet's own learned attention/gating weights)
  - Integrated Gradients (Captum, on the AttentionNet)

All four are run on the SAME sample of test instances so downstream
evaluation compares them on equal footing.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_tabular import LimeTabularExplainer
from captum.attr import IntegratedGradients
import xgboost as xgb

from train_models import AttentionNet

DATA_DIR = "/home/claude/bc_ai_huf_simulation/data/processed"
MODEL_DIR = "/home/claude/bc_ai_huf_simulation/ai_layer/saved_models"
OUT_DIR = "/home/claude/bc_ai_huf_simulation/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_EXPLAIN_FAST = 1000   # SHAP / Attention / Integrated Gradients
N_EXPLAIN_LIME = 150    # LIME is per-instance perturbation-heavy -> much slower


def main():
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    feature_names = list(X_train.columns)

    rng = np.random.RandomState(SEED)
    idx_fast = rng.choice(len(X_test), size=N_EXPLAIN_FAST, replace=False)
    idx_lime = idx_fast[:N_EXPLAIN_LIME]

    X_fast = X_test.iloc[idx_fast].reset_index(drop=True)
    X_lime = X_test.iloc[idx_lime].reset_index(drop=True)

    # --- Load trained models ---
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))

    att_model = AttentionNet(n_features=X_train.shape[1])
    att_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "attention_net.pt")))
    att_model.eval()

    # --- SHAP (TreeExplainer on XGBoost) ---
    print("Computing SHAP...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_fast)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    np.save(os.path.join(OUT_DIR, "shap_values.npy"), shap_values)
    print("SHAP done:", shap_values.shape)

    # --- Attention weights (native model component) ---
    print("Computing Attention weights...")
    Xf_t = torch.tensor(X_fast.values, dtype=torch.float32)
    with torch.no_grad():
        _, attn = att_model(Xf_t, return_attention=True)
    attn = attn.numpy()
    np.save(os.path.join(OUT_DIR, "attention_weights.npy"), attn)
    print("Attention done:", attn.shape)

    # --- Integrated Gradients (Captum, on AttentionNet) ---
    print("Computing Integrated Gradients...")
    ig = IntegratedGradients(lambda x: att_model(x))
    baseline = torch.zeros_like(Xf_t[:1])
    ig_attr = []
    batch = 50
    for i in range(0, len(Xf_t), batch):
        xb = Xf_t[i:i + batch]
        bl = baseline.repeat(xb.shape[0], 1)
        attr = ig.attribute(xb, baselines=bl, n_steps=32)
        ig_attr.append(attr.detach().numpy())
    ig_attr = np.concatenate(ig_attr, axis=0)
    np.save(os.path.join(OUT_DIR, "ig_attributions.npy"), ig_attr)
    print("Integrated Gradients done:", ig_attr.shape)

    # --- LIME (model-agnostic, on XGBoost predict_proba) ---
    print(f"Computing LIME on {N_EXPLAIN_LIME} instances (this is the slow one)...")
    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["normal", "attack"],
        discretize_continuous=True,
        random_state=SEED,
    )

    def predict_fn(x):
        return xgb_model.predict_proba(pd.DataFrame(x, columns=feature_names))

    lime_matrix = np.zeros((len(X_lime), len(feature_names)))
    for i in range(len(X_lime)):
        exp = lime_explainer.explain_instance(
            X_lime.values[i], predict_fn, num_features=len(feature_names), num_samples=500,
        )
        for feat_idx, weight in exp.local_exp[1]:
            lime_matrix[i, feat_idx] = weight
        if (i + 1) % 25 == 0:
            print(f"  LIME progress: {i + 1}/{len(X_lime)}")
    np.save(os.path.join(OUT_DIR, "lime_attributions.npy"), lime_matrix)
    print("LIME done:", lime_matrix.shape)

    meta = {
        "idx_fast": idx_fast.tolist(),
        "idx_lime": idx_lime.tolist(),
        "n_explain_fast": N_EXPLAIN_FAST,
        "n_explain_lime": N_EXPLAIN_LIME,
        "feature_names": feature_names,
    }
    with open(os.path.join(OUT_DIR, "xai_explain_meta.json"), "w") as f:
        json.dump(meta, f)
    print("\nAll explanation methods complete.")


if __name__ == "__main__":
    main()
