"""
Cross-layer synergy test: replaces the original paper's Table 13 (which
implied blockchain and human-factor layers directly raise AI detection
accuracy -- a causal link that has no real mechanism, since neither
blockchain logging nor user-adaptation training changes what a classifier
learned from training data).

Instead this defines an explicit, disclosed composite "Security
Effectiveness Index" (SEI):

    SEI = w_detect * detection_accuracy
        + w_avail  * system_availability
        + w_adapt  * mean_user_adaptation_index

with weights (0.5, 0.25, 0.25) reflecting that detection accuracy is the
framework's primary security function, with availability and human
adaptation as supporting dimensions -- disclosed here so the weighting
choice is inspectable/challengeable, not hidden inside a single number.

Four scenarios are compared. Since the AI layer's own detection accuracy is
NOT mechanically changed by adding blockchain or human factors (each layer
solves a different problem), "AI-only" and "AI+Blockchain" share the same
detection_accuracy term; what changes across scenarios is which of the
other two terms are included. This makes explicit and honest what the
original paper's Table 13 implied without justification: the apparent
"synergy" comes from combining genuinely complementary capabilities (better
detection, higher availability, better-adapted users) into one framework,
not from one layer improving another layer's own metric.
"""
import json

RES = "/home/claude/bc_ai_huf_simulation/results"

with open(f"{RES}/ai_layer_core_metrics.json") as f:
    ai = json.load(f)
with open(f"{RES}/blockchain_layer_results.json") as f:
    bc = json.load(f)
with open(f"{RES}/human_factor_layer_results.json") as f:
    hf = json.load(f)

detection_accuracy = ai["xgboost"]["accuracy"]  # use the stronger of the two trained models
availability_with_bc = bc["availability_model"]["system_availability_with_blockchain"]
availability_without_bc = bc["availability_model"]["baseline_availability_without_blockchain"]
final_uai = hf["summary"]["final_overall_UAI"]
initial_uai = hf["summary"]["initial_overall_UAI"]

W_DETECT, W_AVAIL, W_ADAPT = 0.5, 0.25, 0.25


def sei(detect, avail, adapt):
    return W_DETECT * detect + W_AVAIL * avail + W_ADAPT * adapt

scenarios = {
    "AI_only": sei(detection_accuracy, availability_without_bc, 0.0),
    "AI_plus_Blockchain": sei(detection_accuracy, availability_with_bc, 0.0),
    "AI_plus_HumanFactor": sei(detection_accuracy, availability_without_bc, final_uai),
    "Full_Framework_BC_AI_HUF": sei(detection_accuracy, availability_with_bc, final_uai),
}

# also report at t=0 (before user adaptation) to show the human factor's
# genuine marginal contribution grows over the 12-week deployment window
scenarios_week0 = {
    "Full_Framework_at_week0_UAI": sei(detection_accuracy, availability_with_bc, initial_uai),
}

result = {
    "weights": {"detection": W_DETECT, "availability": W_AVAIL, "adaptation": W_ADAPT},
    "inputs": {
        "detection_accuracy_xgboost": detection_accuracy,
        "availability_with_blockchain": availability_with_bc,
        "availability_without_blockchain": availability_without_bc,
        "final_UAI_week12": final_uai,
        "initial_UAI_week0": initial_uai,
    },
    "SEI_by_scenario": scenarios,
    "SEI_full_framework_week0_vs_week12": {
        "week0": scenarios_week0["Full_Framework_at_week0_UAI"],
        "week12": scenarios["Full_Framework_BC_AI_HUF"],
    },
    "note": (
        "Detection accuracy is held constant across scenarios because "
        "neither blockchain nor human-factor layers mechanically change "
        "what the trained classifier learned; the composite index instead "
        "shows that the FULL framework's combined security posture "
        "(detection + availability + user adaptation) exceeds any "
        "single-layer or two-layer subset, which is the honest form of "
        "the 'synergy' claim -- complementary capability, not mutual "
        "accuracy inflation."
    ),
}

with open(f"{RES}/synergy_index_results.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
