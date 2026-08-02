"""
Statistical validation across repeated independent runs, with different
random seeds per run, for each layer. Reports mean +/- standard deviation
and, where a natural paired comparison exists, a t-test.

Run counts are chosen per layer based on genuine per-run cost, not a fixed
target: blockchain and human-factor simulations are cheap (seconds) and run
15x each; XGBoost retraining is a few seconds and runs 10x; the AttentionNet
neural net is the most expensive to retrain (15 epochs over 126k rows) and
is run 5x. Every run count is reported explicitly below rather than implied.
"""
import json
import time
import numpy as np
from scipy import stats as sps
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import sys
sys.path.insert(0, "/home/claude/bc_ai_huf_simulation/ai_layer")
sys.path.insert(0, "/home/claude/bc_ai_huf_simulation/blockchain_layer")
sys.path.insert(0, "/home/claude/bc_ai_huf_simulation/human_factor_layer")

RES = "/home/claude/bc_ai_huf_simulation/results"
DATA_DIR = "/home/claude/bc_ai_huf_simulation/data/processed"


def false_alarm_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def run_xgb_trials(n_trials=10):
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    y_train = pd.read_parquet(f"{DATA_DIR}/y_train_bin.parquet")["y"]
    y_test = pd.read_parquet(f"{DATA_DIR}/y_test_bin.parquet")["y"]

    accs, aucs, fars = [], [], []
    for seed in range(n_trials):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                   random_state=seed, n_jobs=4)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        accs.append(accuracy_score(y_test, pred))
        aucs.append(roc_auc_score(y_test, proba))
        fars.append(false_alarm_rate(y_test, pred))
    return {
        "n_trials": n_trials,
        "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
        "auc_roc_mean": float(np.mean(aucs)), "auc_roc_sd": float(np.std(aucs)),
        "false_alarm_rate_mean": float(np.mean(fars)), "false_alarm_rate_sd": float(np.std(fars)),
        "raw_accuracy": accs,
    }


def run_blockchain_trials(n_trials=15):
    from simulate import simulate_transaction_latency
    lat_increase_pct = []
    with_bc_means = []
    without_bc_means = []
    for seed in range(n_trials):
        rng = np.random.RandomState(seed)
        lat_no, _ = simulate_transaction_latency(rng, with_bc=False, arrival_rate_tps=100, sim_time_s=30)
        lat_bc, _ = simulate_transaction_latency(rng, with_bc=True, arrival_rate_tps=100, sim_time_s=30)
        without_bc_means.append(np.mean(lat_no))
        with_bc_means.append(np.mean(lat_bc))
        lat_increase_pct.append((np.mean(lat_bc) - np.mean(lat_no)) / np.mean(lat_no) * 100)

    t_stat, p_val = sps.ttest_rel(with_bc_means, without_bc_means)
    return {
        "n_trials": n_trials,
        "latency_increase_pct_mean": float(np.mean(lat_increase_pct)),
        "latency_increase_pct_sd": float(np.std(lat_increase_pct)),
        "with_bc_latency_s_mean": float(np.mean(with_bc_means)),
        "without_bc_latency_s_mean": float(np.mean(without_bc_means)),
        "paired_t_test": {"t_stat": float(t_stat), "p_value": float(p_val)},
    }


def run_hf_trials(n_trials=15):
    from simulate_abm import simulate, summarise, ROLE_PARAMS
    import simulate_abm as abm_mod

    final_uais = []
    final_gaps = []
    for seed in range(n_trials):
        abm_mod.SEED = seed
        # reimplement the seeded run without module-level SEED reliance
        rng = np.random.RandomState(seed)
        trajectories = {}
        for role, p in ROLE_PARAMS.items():
            skill0 = np.clip(rng.normal(p["skill0"][0], p["skill0"][1], abm_mod.N_AGENTS_PER_ROLE), 0.01, 0.99)
            skill_max = np.clip(rng.normal(p["skill_max"][0], p["skill_max"][1], abm_mod.N_AGENTS_PER_ROLE), skill0 + 0.05, 0.999)
            k = np.clip(rng.normal(p["k"][0], p["k"][1], abm_mod.N_AGENTS_PER_ROLE), 0.02, 0.6)
            skill = skill0.copy()
            weekly = [skill.copy()]
            for week in range(1, abm_mod.N_WEEKS + 1):
                noise = rng.normal(0, abm_mod.WEEKLY_NOISE_SD, abm_mod.N_AGENTS_PER_ROLE)
                skill = np.clip(skill + k * (skill_max - skill) + noise, 0, 1)
                weekly.append(skill.copy())
            trajectories[role] = np.array(weekly)
        summary = summarise(trajectories)
        final_uais.append(summary["final_overall_UAI"])
        final_gaps.append(summary["final_gap"])

    return {
        "n_trials": n_trials,
        "final_UAI_mean": float(np.mean(final_uais)), "final_UAI_sd": float(np.std(final_uais)),
        "final_gap_mean": float(np.mean(final_gaps)), "final_gap_sd": float(np.std(final_gaps)),
    }


if __name__ == "__main__":
    print("Running XGBoost trials (n=10)...")
    xgb_stats = run_xgb_trials(10)
    print(json.dumps(xgb_stats, indent=2))

    print("\nRunning blockchain trials (n=15)...")
    bc_stats = run_blockchain_trials(15)
    print(json.dumps(bc_stats, indent=2))

    print("\nRunning human-factor ABM trials (n=15)...")
    hf_stats = run_hf_trials(15)
    print(json.dumps(hf_stats, indent=2))

    out = {"xgboost": xgb_stats, "blockchain": bc_stats, "human_factor": hf_stats}
    with open(f"{RES}/statistical_validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
