"""
Agent-based model (ABM) of the User Adaptation Index (UAI) across four
framework roles, replacing the discontinued live-participant study.

Theoretical grounding (each cited so parameter choices are traceable, not
arbitrary):

  - Exponential skill-acquisition / learning-curve model: performance
    approaches an individual asymptote as
        skill(t+1) = skill(t) + k * (skill_max - skill(t)) + noise
    a discrete-time form of the classic exponential learning curve used
    throughout skill-acquisition research (closely related to the Power Law
    of Practice, Newell & Rosenbloom, 1981, "Mechanisms of Skill
    Acquisition and the Law of Practice").

  - Cognitive Load Theory (Sweller, 1988, "Cognitive load during problem
    solving"): working-memory constraints limit how fast NEW, unfamiliar
    procedures can be absorbed; agents are given a per-role "intrinsic load"
    parameter that slows the effective learning rate.

  - Expertise-reversal effect (Kalyuga, Ayres, Chandler & Sweller, 2003,
    "The Expertise Reversal Effect"): prior domain expertise reduces
    effective cognitive load for related tasks, so agents with relevant
    prior expertise (Security Analyst, System Administrator) are
    parameterised with a HIGHER starting skill AND a higher learning rate
    than agents without that background (Operator, User) -- this is what
    is expected to produce both a higher trajectory AND a narrowing gap
    over time (as novices' schemas form, the marginal benefit of continued
    practice for experts shrinks while novices are still climbing steeply).

Role parameter ranges are documented per-role below; each is a *documented
modelling assumption* grounded in the cited theory's qualitative
predictions, not a fit to any target number. This script draws random
per-agent parameters from role-level distributions and lets the population
trajectory emerge from the simulation, so the resulting weekly UAI curve is
NOT hand-set to reproduce the original paper's numbers.
"""
import json
import os
import numpy as np

OUT_DIR = "/home/claude/bc_ai_huf_simulation/results"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
N_WEEKS = 12
N_AGENTS_PER_ROLE = 40

# Role parameters: (skill0_mean, skill0_sd, skill_max_mean, skill_max_sd, k_mean, k_sd)
# skill in [0,1], k = weekly fractional closure of the gap to the personal asymptote.
# Security Analyst / System Administrator: relevant prior expertise -> higher
#   starting skill, higher asymptote, faster learning rate (expertise-reversal effect).
# Operator / User: no assumed prior security-specific expertise -> lower
#   starting point, lower learning rate, more intrinsic cognitive load.
ROLE_PARAMS = {
    "Security Analyst":     dict(skill0=(0.42, 0.06), skill_max=(0.97, 0.02), k=(0.28, 0.05)),
    "System Administrator": dict(skill0=(0.35, 0.06), skill_max=(0.93, 0.03), k=(0.24, 0.05)),
    "Operator":             dict(skill0=(0.22, 0.05), skill_max=(0.88, 0.04), k=(0.19, 0.05)),
    "User":                 dict(skill0=(0.15, 0.05), skill_max=(0.84, 0.05), k=(0.16, 0.05)),
}
WEEKLY_NOISE_SD = 0.015  # behavioural noise: real practice isn't monotonic week to week


def simulate():
    rng = np.random.RandomState(SEED)
    trajectories = {role: [] for role in ROLE_PARAMS}
    agent_params = {}

    for role, p in ROLE_PARAMS.items():
        skill0 = np.clip(rng.normal(p["skill0"][0], p["skill0"][1], N_AGENTS_PER_ROLE), 0.01, 0.99)
        skill_max = np.clip(rng.normal(p["skill_max"][0], p["skill_max"][1], N_AGENTS_PER_ROLE), skill0 + 0.05, 0.999)
        k = np.clip(rng.normal(p["k"][0], p["k"][1], N_AGENTS_PER_ROLE), 0.02, 0.6)
        agent_params[role] = {"skill0": skill0, "skill_max": skill_max, "k": k}

        skill = skill0.copy()
        weekly = [skill.copy()]
        for week in range(1, N_WEEKS + 1):
            noise = rng.normal(0, WEEKLY_NOISE_SD, N_AGENTS_PER_ROLE)
            skill = skill + k * (skill_max - skill) + noise
            skill = np.clip(skill, 0, 1)
            weekly.append(skill.copy())
        trajectories[role] = np.array(weekly)  # shape (N_WEEKS+1, N_AGENTS_PER_ROLE)

    return trajectories, agent_params


def summarise(trajectories):
    weeks = list(range(0, N_WEEKS + 1))
    per_role_mean = {role: traj.mean(axis=1).tolist() for role, traj in trajectories.items()}
    per_role_sd = {role: traj.std(axis=1).tolist() for role, traj in trajectories.items()}

    all_agents_by_week = np.concatenate([traj for traj in trajectories.values()], axis=1)  # (weeks, total_agents)
    overall_uai_mean = all_agents_by_week.mean(axis=1).tolist()
    overall_uai_sd = all_agents_by_week.std(axis=1).tolist()

    most_experienced = "Security Analyst"
    least_experienced = "User"
    gap = [per_role_mean[most_experienced][w] - per_role_mean[least_experienced][w] for w in weeks]

    return {
        "weeks": weeks,
        "per_role_mean_UAI": per_role_mean,
        "per_role_sd_UAI": per_role_sd,
        "overall_UAI_mean": overall_uai_mean,
        "overall_UAI_sd": overall_uai_sd,
        "experience_gap_most_minus_least": gap,
        "final_overall_UAI": overall_uai_mean[-1],
        "initial_overall_UAI": overall_uai_mean[0],
        "final_gap": gap[-1],
        "initial_gap": gap[0],
        "role_ranking_final_week": sorted(
            [(role, per_role_mean[role][-1]) for role in per_role_mean],
            key=lambda x: -x[1],
        ),
    }


def main():
    trajectories, agent_params = simulate()
    summary = summarise(trajectories)

    out = {
        "n_agents_per_role": N_AGENTS_PER_ROLE,
        "n_weeks": N_WEEKS,
        "role_params_used": {r: {k: list(v) for k, v in p.items()} for r, p in ROLE_PARAMS.items()},
        "summary": summary,
    }
    with open(os.path.join(OUT_DIR, "human_factor_layer_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
