# BC-AI-HUF

Code and result files for *"An Explainable AI and Blockchain-Enhanced Framework for Adaptive Human-Centric Cybersecurity in Cloud-IoT Environments"* (BC-AI-HUF), submitted to the International Journal of Intelligent Engineering and Systems.

This repository accompanies the paper's **Data and Code Availability** statement. Every number reported in the paper's tables and figures is produced by the scripts here, run against the real, publicly available NSL-KDD benchmark (for the AI layer) and the parameterised simulations described in the paper's Methodology (for the blockchain and human-factor layers).

## Layout

- `ai_layer/` — NSL-KDD preprocessing, XGBoost + attention-gated feed-forward network training, LIME/SHAP/Integrated-Gradients explainability evaluation, zero-day (held-out attack subtype) analysis, and the extended experiments reported in the Discussion section: the cost-sensitive R2L/U2R class-rebalancing experiment and the internal Logistic Regression baseline (`extended_experiments.py`).
- `blockchain_layer/` — discrete-event simulation (via `simpy`) of a five-node, permissioned Proof-of-Stake network with a 4-of-5 BFT-style quorum, used to measure transaction latency, throughput, storage growth, and availability with and without blockchain in place.
- `human_factor_layer/` — the agent-based model (160 simulated agents across four roles, 12 weekly time steps) used to compute the User Adaptation Index (UAI) trajectory.
- `results/` — every JSON/`.npy` result file the paper's tables are drawn from, plus `statistical_validation.py` (repeated-seed statistical validation across the AI, blockchain, and human-factor layers) and `synergy_index.py` (the Security Effectiveness Index computation and its weighting-sensitivity check).
- `docs/methodology_and_results.md` — a working methodology/results note used while building the pipeline.

## Data

The AI layer trains and evaluates on the official NSL-KDD split (Tavallaee et al., 2009): `KDDTrain+.txt` (125,973 rows) and `KDDTest+.txt` (22,544 rows), available from the standard public UNB mirror. The raw files are not committed here (they are a third-party public benchmark, not this project's output); download them and place them under `data/` before running `ai_layer/preprocess.py`, which regenerates the processed feature parquet files consumed by the rest of the pipeline.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Preprocess NSL-KDD (expects data/KDDTrain+.txt and data/KDDTest+.txt)
python ai_layer/preprocess.py

# 2. Train the two detection models and compute core AI-layer metrics
python ai_layer/train_models.py

# 3. Explainability evaluation (LIME / SHAP / Attention / Integrated Gradients)
python ai_layer/xai_explain.py
python ai_layer/xai_evaluate.py

# 4. Zero-day (held-out attack subtype) analysis
python ai_layer/zero_day_analysis.py

# 5. Extended experiments: cost-sensitive R2L/U2R retraining + internal baseline
python ai_layer/extended_experiments.py

# 6. Blockchain discrete-event simulation
python blockchain_layer/simulate.py

# 7. Human-factor agent-based model
python human_factor_layer/simulate_abm.py

# 8. Repeated-seed statistical validation and the Security Effectiveness Index
python results/statistical_validation.py
python results/synergy_index.py
```

Each script writes its output as a JSON (or `.npy`) file under `results/`, which is what is already committed in this repository so the paper's figures can be checked without re-running the full pipeline.

## Citation

If you use this code, please cite the BC-AI-HUF paper (citation details to be added on acceptance).
