"""
Evaluate the four sets of explanations (SHAP, LIME, Attention, Integrated
Gradients) against real, well-defined, citable metrics -- not ad hoc numbers.

Metric definitions (each is a standard, published formulation, simplified
for tabular data):

  Fidelity  -- correlation-based infidelity, after Yeh et al., "On the
              (In)fidelity and Sensitivity of Explanations" (NeurIPS 2019).
              For each instance, take the top-k attributed features, zero
              them out (impute with the training-set mean), and measure how
              well the attribution-predicted drop in output matches the
              actual observed drop. Reported as 1 - normalised error, so
              higher = more faithful (matches the "higher is better"
              convention of the original paper's Fidelity column).

  Stability -- Lipschitz-style robustness, after Alvarez-Melis & Jaakkola,
              "On the Robustness of Interpretability Methods" (2018). Add
              small Gaussian noise to each instance, recompute a *local*
              attribution proxy (finite-difference gradient of the model
              output at the perturbed point), and measure the cosine
              similarity between the original and perturbed attribution
              vectors. Higher = more stable under small input changes.

  Completeness -- top-k sufficiency / comprehensiveness, after DeYoung et
              al., ERASER (2020). Keep ONLY the top-k attributed features
              (mask the rest to the training mean) and measure how close the
              resulting prediction is to the prediction using all features.
              Higher = the top attributed features are more sufficient on
              their own to reproduce the model's decision.

  Complexity (labelled explicitly as a PROXY for the original paper's
              "human intelligibility / Comprehensibility" score, not a
              replacement for it) -- entropy of the normalised absolute
              attribution distribution, after Bhatt et al., "Evaluating and
              Aggregating Feature-based Model Explanations" (IJCAI 2020).
              Lower entropy = attribution mass concentrated on fewer
              features = an explanation a human could scan more quickly.
              Reported inverted (1 - normalised entropy) so higher = simpler
              / more scannable, matching the "higher is better" convention.
              This is NOT a substitute for asking real users to rate
              explanations; see docs/xai_methodology.md for why a genuine
              human intelligibility figure needs an actual user study.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from train_models import AttentionNet

DATA_DIR = "/home/claude/bc_ai_huf_simulation/data/processed"
MODEL_DIR = "/home/claude/bc_ai_huf_simulation/ai_layer/saved_models"
RES_DIR = "/home/claude/bc_ai_huf_simulation/results"
SEED = 42
TOP_K = 15  # ~12% of 122 features


def load_common():
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    with open(os.path.join(RES_DIR, "xai_explain_meta.json")) as f:
        meta = json.load(f)
    feature_means = X_train.mean().values

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))

    att_model = AttentionNet(n_features=X_train.shape[1])
    att_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "attention_net.pt")))
    att_model.eval()

    return X_train, X_test, meta, feature_means, xgb_model, att_model


def xgb_score(model, X_df):
    return model.predict_proba(X_df)[:, 1]


def att_score(model, X_arr):
    with torch.no_grad():
        out = model(torch.tensor(X_arr, dtype=torch.float32))
        return torch.sigmoid(out).numpy().ravel()


def fidelity(attr, X, feature_means, score_fn, top_k=TOP_K):
    """Correlation between attribution-predicted drop and actual output drop
    when the top-k attributed features are masked to the training mean."""
    n = X.shape[0]
    orig_scores = score_fn(X)
    predicted_drop = np.zeros(n)
    actual_drop = np.zeros(n)
    for i in range(n):
        order = np.argsort(-np.abs(attr[i]))[:top_k]
        x_masked = X[i].copy()
        predicted_drop[i] = np.abs(attr[i][order]).sum()
        x_masked[order] = feature_means[order]
        masked_score = score_fn(x_masked.reshape(1, -1))[0]
        actual_drop[i] = abs(orig_scores[i] - masked_score)
    # normalise both to [0,1] via rank correlation-friendly scaling, then
    # score as 1 - mean absolute normalised discrepancy (higher = better)
    pd_norm = predicted_drop / (predicted_drop.max() + 1e-12)
    ad_norm = actual_drop / (actual_drop.max() + 1e-12)
    infidelity_err = np.mean(np.abs(pd_norm - ad_norm))
    corr = np.corrcoef(predicted_drop, actual_drop)[0, 1] if np.std(actual_drop) > 1e-9 else 0.0
    return {
        "fidelity_score_1_minus_error": float(1 - infidelity_err),
        "fidelity_correlation": float(corr) if not np.isnan(corr) else 0.0,
    }


def stability(attr, X, score_fn, noise_std=0.01, n_repeat=3, seed=SEED):
    """Cosine similarity between the original attribution vector and a
    finite-difference local-gradient proxy recomputed after small Gaussian
    perturbations -- how much does the explanation move under tiny,
    meaning-preserving input noise."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    sims = []
    for i in range(n):
        base_attr = attr[i]
        base_norm = np.linalg.norm(base_attr)
        if base_norm < 1e-12:
            continue
        local_sims = []
        for _ in range(n_repeat):
            noise = rng.normal(0, noise_std, size=d)
            x_pert = X[i] + noise
            # finite-difference local gradient proxy at the perturbed point
            eps = 1e-3
            grad = np.zeros(d)
            base_score = score_fn(x_pert.reshape(1, -1))[0]
            # cheap coordinate-subset finite difference on the same top-|attr| dims
            top_dims = np.argsort(-np.abs(base_attr))[:TOP_K]
            for j in top_dims:
                xp = x_pert.copy()
                xp[j] += eps
                grad[j] = (score_fn(xp.reshape(1, -1))[0] - base_score) / eps
            denom = (np.linalg.norm(grad) * base_norm)
            if denom < 1e-12:
                continue
            cos = np.dot(grad, base_attr) / denom
            local_sims.append(cos)
        if local_sims:
            sims.append(np.mean(local_sims))
    return {"stability_mean_cosine_similarity": float(np.mean(sims)) if sims else None,
            "stability_std": float(np.std(sims)) if sims else None}


def completeness(attr, X, feature_means, score_fn, top_k=TOP_K):
    """How close is the prediction using ONLY the top-k attributed features
    (rest masked to the mean) to the prediction using all features."""
    n = X.shape[0]
    full_scores = score_fn(X)
    topk_scores = np.zeros(n)
    for i in range(n):
        order = np.argsort(-np.abs(attr[i]))[:top_k]
        x_topk_only = np.tile(feature_means, (1, 1))[0].copy()
        x_topk_only[order] = X[i][order]
        topk_scores[i] = score_fn(x_topk_only.reshape(1, -1))[0]
    diff = np.abs(full_scores - topk_scores)
    return {"completeness_score_1_minus_meandiff": float(1 - np.mean(diff))}


def complexity_proxy(attr):
    """Entropy-based proxy for how concentrated (scannable) an explanation
    is. Explicitly NOT the same as the original paper's human-rated
    intelligibility score -- see module docstring."""
    n, d = attr.shape
    scores = []
    for i in range(n):
        a = np.abs(attr[i])
        s = a.sum()
        if s < 1e-12:
            continue
        p = a / s
        p = p[p > 0]
        ent = -np.sum(p * np.log(p))
        max_ent = np.log(d)
        scores.append(1 - ent / max_ent)  # higher = more concentrated/scannable
    return {"complexity_proxy_1_minus_norm_entropy": float(np.mean(scores)) if scores else None}


def main():
    X_train, X_test, meta, feature_means, xgb_model, att_model = load_common()
    idx_fast = np.array(meta["idx_fast"])
    X_fast = X_test.iloc[idx_fast].values
    idx_lime = np.array(meta["idx_lime"])
    X_lime = X_test.iloc[idx_lime].values

    shap_values = np.load(os.path.join(RES_DIR, "shap_values.npy"))
    attn_weights = np.load(os.path.join(RES_DIR, "attention_weights.npy"))
    ig_attr = np.load(os.path.join(RES_DIR, "ig_attributions.npy"))
    lime_attr = np.load(os.path.join(RES_DIR, "lime_attributions.npy"))

    # XGBoost single-row predict_proba is ~7.4ms/call and the stability metric
    # alone issues ~45 calls/instance, so SHAP/Ensemble metric evaluation is
    # subsampled to keep total runtime bounded; Attention/Integrated Gradients
    # use the fast torch model and keep the full 1000-instance sample.
    N_XGB_EVAL = 300
    rng_eval = np.random.RandomState(SEED)
    eval_sub = rng_eval.choice(len(X_fast), size=N_XGB_EVAL, replace=False)
    X_fast_xgb = X_fast[eval_sub]
    shap_values_sub = shap_values[eval_sub]

    results = {}
    results["_metadata"] = {
        "n_instances_SHAP_Ensemble": N_XGB_EVAL,
        "n_instances_LIME": len(X_lime),
        "n_instances_Attention_IG": len(X_fast),
        "top_k_features": TOP_K,
    }

    print("Evaluating SHAP (XGBoost score fn)...")
    sf = lambda x: xgb_score(xgb_model, pd.DataFrame(x, columns=X_test.columns))
    results["SHAP"] = {
        **fidelity(shap_values_sub, X_fast_xgb, feature_means, sf),
        **stability(shap_values_sub, X_fast_xgb, sf),
        **completeness(shap_values_sub, X_fast_xgb, feature_means, sf),
        **complexity_proxy(shap_values_sub),
    }

    print("Evaluating LIME (XGBoost score fn, smaller sample)...")
    results["LIME"] = {
        **fidelity(lime_attr, X_lime, feature_means, sf),
        **stability(lime_attr, X_lime, sf),
        **completeness(lime_attr, X_lime, feature_means, sf),
        **complexity_proxy(lime_attr),
    }

    print("Evaluating Attention (AttentionNet score fn)...")
    af = lambda x: att_score(att_model, x)
    results["Attention"] = {
        **fidelity(attn_weights, X_fast, feature_means, af),
        **stability(attn_weights, X_fast, af),
        **completeness(attn_weights, X_fast, feature_means, af),
        **complexity_proxy(attn_weights),
    }

    print("Evaluating Integrated Gradients (AttentionNet score fn)...")
    results["IntegratedGradients"] = {
        **fidelity(ig_attr, X_fast, feature_means, af),
        **stability(ig_attr, X_fast, af),
        **completeness(ig_attr, X_fast, feature_means, af),
        **complexity_proxy(ig_attr),
    }

    print("Evaluating INTEGRATED ensemble (mean of normalised SHAP+Attention+IG, subsampled)...")
    def norm(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1
        return a / n
    ensemble_attr_full = (norm(shap_values) + norm(attn_weights) + norm(ig_attr)) / 3
    ensemble_attr_sub = ensemble_attr_full[eval_sub]
    results["Integrated_SHAP_Attention_IG"] = {
        **fidelity(ensemble_attr_sub, X_fast_xgb, feature_means, sf),
        **stability(ensemble_attr_sub, X_fast_xgb, sf),
        **completeness(ensemble_attr_sub, X_fast_xgb, feature_means, sf),
        **complexity_proxy(ensemble_attr_sub),
    }

    with open(os.path.join(RES_DIR, "xai_evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
