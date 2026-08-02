"""
Genuine zero-day / novel-attack generalisation test.

NSL-KDD's KDDTest+ contains 17 attack subtypes that never appear anywhere in
KDDTrain+ (verified empirically in preprocess.py). Splitting test performance
into "known attack types" vs "novel attack types" gives a real, data-driven
measure of how much detection quality degrades on attacks the model never
had a chance to learn -- a direct, honest replacement for the original
paper's unverified "zero-day performance is relatively weaker" claim.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

DATA_DIR = "/home/claude/bc_ai_huf_simulation/data/processed"
RESULTS_DIR = "/home/claude/bc_ai_huf_simulation/results"

y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test_bin.parquet"))["y"]
is_novel = pd.read_parquet(os.path.join(DATA_DIR, "is_novel_test.parquet"))["is_novel"]
test_label_raw = pd.read_parquet(os.path.join(DATA_DIR, "test_label_raw.parquet"))["label"]

xgb_pred = np.load(os.path.join(RESULTS_DIR, "xgb_pred.npy"))
att_pred = np.load(os.path.join(RESULTS_DIR, "att_pred.npy"))

results = {}
for name, pred in [("xgboost", xgb_pred), ("attention_net", att_pred)]:
    known_mask = (~is_novel) & (y_test == 1)   # known attack traffic (attack rows whose label WAS in training)
    novel_mask = is_novel & (y_test == 1)      # novel/never-seen attack traffic
    normal_mask = (y_test == 0)

    known_recall = recall_score(y_test[known_mask | normal_mask], pred[known_mask | normal_mask],
                                 pos_label=1) if known_mask.sum() else None
    novel_recall = recall_score(y_test[novel_mask | normal_mask], pred[novel_mask | normal_mask],
                                 pos_label=1) if novel_mask.sum() else None

    overall_acc = accuracy_score(y_test, pred)
    known_only_acc = accuracy_score(y_test[known_mask], pred[known_mask]) if known_mask.sum() else None
    novel_only_acc = accuracy_score(y_test[novel_mask], pred[novel_mask]) if novel_mask.sum() else None

    results[name] = {
        "n_known_attack_samples": int(known_mask.sum()),
        "n_novel_attack_samples": int(novel_mask.sum()),
        "n_normal_samples": int(normal_mask.sum()),
        "detection_rate_known_attacks": known_only_acc,   # == recall on known-attack rows only
        "detection_rate_novel_attacks": novel_only_acc,   # == recall on novel-attack rows only
        "gap_known_minus_novel": (known_only_acc - novel_only_acc) if (known_only_acc is not None and novel_only_acc is not None) else None,
        "overall_accuracy": overall_acc,
    }

# Per-novel-attack-type breakdown (which specific unseen attacks are hardest)
per_attack = {}
for label in sorted(test_label_raw[is_novel].unique()):
    mask = test_label_raw == label
    n = int(mask.sum())
    xgb_det = float((xgb_pred[mask.values] == 1).mean())
    att_det = float((att_pred[mask.values] == 1).mean())
    per_attack[label] = {"n_samples": n, "xgboost_detection_rate": xgb_det, "attention_net_detection_rate": att_det}

out = {"summary": results, "per_novel_attack_type": per_attack}
with open(os.path.join(RESULTS_DIR, "zero_day_analysis.json"), "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
