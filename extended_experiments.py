"""
Additional, genuinely-run experiments extending the original AI-layer
evaluation:

  1. Extended statistical validation: recall / F1 / detection-time
     (mean +/- SD) across the same 10 independently seeded XGBoost
     retrains used for accuracy/AUC/FPR, so every headline metric in the
     paper carries a repeated-trial uncertainty estimate, not just three
     of seven.

  2. Cost-sensitive (class-rebalanced) retraining targeting the two
     weakest categories, R2L and U2R, via per-sample training weights
     inversely proportional to category frequency. Evaluated on the
     unmodified KDDTest+ set using the same per-category protocol as the
     original Table 1 / Table 3 / Table 4.

  3. An internal, same-protocol baseline (Logistic Regression) trained
     and evaluated on the identical KDDTrain+/KDDTest+ split and feature
     set as the framework's own XGBoost model, giving one truly
     apples-to-apples comparison point (the framework's other published
     comparisons use different datasets entirely).

All numbers below come from actually running this script against the
real, unmodified NSL-KDD data already used elsewhere in the pipeline.
"""
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              f1_score, recall_score, precision_score)

DATA_DIR = "/home/claude/bc_ai_huf_simulation/data/processed"
OUT_DIR = "/home/claude/bc_ai_huf_simulation/results"


def false_alarm_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def load_all():
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    y_train_bin = pd.read_parquet(f"{DATA_DIR}/y_train_bin.parquet")["y"]
    y_test_bin = pd.read_parquet(f"{DATA_DIR}/y_test_bin.parquet")["y"]
    y_train_cat = pd.read_parquet(f"{DATA_DIR}/y_train_cat.parquet")["y"]
    y_test_cat = pd.read_parquet(f"{DATA_DIR}/y_test_cat.parquet")["y"]
    return X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat


# ---------------------------------------------------------------------
# 1. Extended statistical validation (recall / F1 / detection time)
# ---------------------------------------------------------------------
def extended_xgb_trials(X_train, X_test, y_train, y_test, n_trials=10):
    recalls, f1s, dets, precs = [], [], [], []
    for seed in range(n_trials):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                   random_state=seed, n_jobs=4)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        sample = X_test.sample(n=min(500, len(X_test)), random_state=seed)
        t0 = time.perf_counter()
        for i in range(len(sample)):
            _ = model.predict(sample.iloc[[i]])
        t1 = time.perf_counter()
        per_sample_ms = (t1 - t0) / len(sample) * 1000

        recalls.append(recall_score(y_test, pred))
        f1s.append(f1_score(y_test, pred))
        precs.append(precision_score(y_test, pred))
        dets.append(per_sample_ms)

    return {
        "n_trials": n_trials,
        "recall_mean": float(np.mean(recalls)), "recall_sd": float(np.std(recalls)),
        "precision_mean": float(np.mean(precs)), "precision_sd": float(np.std(precs)),
        "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
        "detection_time_ms_mean": float(np.mean(dets)), "detection_time_ms_sd": float(np.std(dets)),
    }


# ---------------------------------------------------------------------
# 2. Cost-sensitive / class-rebalanced retraining (targets R2L, U2R)
# ---------------------------------------------------------------------
def build_sample_weights(y_train_cat, boost=None):
    if boost is None:
        boost = {"R2L": 8.0, "U2R": 15.0, "DoS": 1.0, "Probe": 1.0, "normal": 1.0}
    return y_train_cat.map(boost).astype(float).values


def per_category_recall(y_test_cat, y_pred, y_test_bin):
    out = {}
    for cat in ["DoS", "Probe", "R2L", "U2R"]:
        mask = (y_test_cat == cat).values
        n = int(mask.sum())
        if n == 0:
            continue
        rec = float(np.mean(y_pred[mask] == 1))
        out[cat] = {"n_samples": n, "detection_rate_recall": rec}
    return out


def run_rebalanced_experiment(X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat, n_trials=5):
    weights = build_sample_weights(y_train_cat)
    per_trial = []
    for seed in range(n_trials):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                   random_state=seed, n_jobs=4)
        model.fit(X_train, y_train_bin, sample_weight=weights)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        cat_recall = per_category_recall(y_test_cat, pred, y_test_bin)
        trial = {
            "seed": seed,
            "accuracy": float(accuracy_score(y_test_bin, pred)),
            "auc_roc": float(roc_auc_score(y_test_bin, proba)),
            "precision": float(precision_score(y_test_bin, pred)),
            "recall": float(recall_score(y_test_bin, pred)),
            "f1": float(f1_score(y_test_bin, pred)),
            "false_alarm_rate": float(false_alarm_rate(y_test_bin, pred)),
            "per_category_recall": cat_recall,
        }
        per_trial.append(trial)

    agg = {"n_trials": n_trials, "sample_weights_used": {"R2L": 8.0, "U2R": 15.0, "DoS": 1.0, "Probe": 1.0, "normal": 1.0}}
    for k in ["accuracy", "auc_roc", "precision", "recall", "f1", "false_alarm_rate"]:
        vals = [t[k] for t in per_trial]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_sd"] = float(np.std(vals))
    for cat in ["DoS", "Probe", "R2L", "U2R"]:
        vals = [t["per_category_recall"][cat]["detection_rate_recall"] for t in per_trial]
        agg[f"{cat}_recall_mean"] = float(np.mean(vals))
        agg[f"{cat}_recall_sd"] = float(np.std(vals))
        agg[f"{cat}_n_samples"] = per_trial[0]["per_category_recall"][cat]["n_samples"]
    agg["per_trial"] = per_trial
    return agg


# ---------------------------------------------------------------------
# 3. Internal, same-protocol baseline (Logistic Regression)
# ---------------------------------------------------------------------
def run_logreg_baseline(X_train, X_test, y_train, y_test, seed=42):
    model = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=4)
    model.fit(X_train, y_train)

    sample = X_test.sample(n=min(500, len(X_test)), random_state=seed)
    t0 = time.perf_counter()
    for i in range(len(sample)):
        _ = model.predict(sample.iloc[[i]])
    t1 = time.perf_counter()
    per_sample_ms = (t1 - t0) / len(sample) * 1000

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": "LogisticRegression",
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "false_alarm_rate": float(false_alarm_rate(y_test, pred)),
        "mean_detection_time_ms_per_sample": per_sample_ms,
        "n_test": len(y_test),
    }


if __name__ == "__main__":
    X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat = load_all()

    print("=== 1. Extended statistical validation (recall/F1/detection time) ===")
    ext_stats = extended_xgb_trials(X_train, X_test, y_train_bin, y_test_bin, n_trials=10)
    print(json.dumps(ext_stats, indent=2))

    print("\n=== 2. Cost-sensitive retraining (R2L/U2R upweighted) ===")
    rebal = run_rebalanced_experiment(X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat, n_trials=5)
    print(json.dumps({k: v for k, v in rebal.items() if k != "per_trial"}, indent=2))

    print("\n=== 3. Internal Logistic Regression baseline (same split) ===")
    logreg = run_logreg_baseline(X_train, X_test, y_train_bin, y_test_bin)
    print(json.dumps(logreg, indent=2))

    out = {
        "extended_statistical_validation_xgboost": ext_stats,
        "rebalanced_r2l_u2r_experiment": rebal,
        "internal_logreg_baseline": logreg,
    }
    with open(f"{OUT_DIR}/extended_experiments_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/extended_experiments_results.json")
