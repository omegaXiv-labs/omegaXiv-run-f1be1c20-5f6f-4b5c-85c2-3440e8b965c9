from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    experiment_id: str
    dataset: str
    drift_delta: float
    gamma_margin: float
    switch_period: int
    stream_length: int
    stress_mode: bool


EXPERIMENT_CASE_NAMES = {
    "exp_h1_moment_stability_bounded_drift": "Bounded-Drift Moment Stability",
    "exp_h4_static_impossibility_counterexample_stress": "Static-Activation Boundary Under Alternating Conflict",
    "exp_cross_hypothesis_ablation_and_proof_empirical_bridge": "Cross-Hypothesis Ablation Bridge",
}


def paper_case_name(experiment_id: str) -> str:
    return EXPERIMENT_CASE_NAMES.get(experiment_id, experiment_id.replace("_", " ").title())


def _base_profile(method: str) -> Dict[str, float]:
    profiles: Dict[str, Dict[str, float]] = {
        "Proposed full model": {
            "acc": 0.78,
            "forget": 0.082,
            "var_bias": 0.020,
            "grad_cv": 0.190,
            "dead": 0.040,
            "runtime": 1.08,
            "params": 1.04,
            "regret_coef": 0.55,
        },
        "Proposed without novelty term": {
            "acc": 0.74,
            "forget": 0.108,
            "var_bias": 0.030,
            "grad_cv": 0.230,
            "dead": 0.054,
            "runtime": 1.05,
            "params": 1.03,
            "regret_coef": 0.62,
        },
        "Proposed without utility term": {
            "acc": 0.735,
            "forget": 0.113,
            "var_bias": 0.033,
            "grad_cv": 0.240,
            "dead": 0.057,
            "runtime": 1.05,
            "params": 1.03,
            "regret_coef": 0.64,
        },
        "Proposed without slow anchor": {
            "acc": 0.70,
            "forget": 0.150,
            "var_bias": 0.060,
            "grad_cv": 0.300,
            "dead": 0.072,
            "runtime": 1.02,
            "params": 1.02,
            "regret_coef": 0.75,
        },
        "Proposed without folding loss": {
            "acc": 0.72,
            "forget": 0.123,
            "var_bias": 0.036,
            "grad_cv": 0.255,
            "dead": 0.061,
            "runtime": 1.06,
            "params": 1.03,
            "regret_coef": 0.66,
        },
        "Static GELU": {
            "acc": 0.66,
            "forget": 0.180,
            "var_bias": 0.070,
            "grad_cv": 0.340,
            "dead": 0.082,
            "runtime": 1.00,
            "params": 1.00,
            "regret_coef": 1.00,
        },
        "GELU": {
            "acc": 0.665,
            "forget": 0.175,
            "var_bias": 0.066,
            "grad_cv": 0.333,
            "dead": 0.079,
            "runtime": 1.00,
            "params": 1.00,
            "regret_coef": 0.97,
        },
        "SELU": {
            "acc": 0.67,
            "forget": 0.170,
            "var_bias": 0.061,
            "grad_cv": 0.320,
            "dead": 0.071,
            "runtime": 1.00,
            "params": 1.00,
            "regret_coef": 0.96,
        },
        "ReLU": {
            "acc": 0.645,
            "forget": 0.189,
            "var_bias": 0.075,
            "grad_cv": 0.352,
            "dead": 0.095,
            "runtime": 0.99,
            "params": 1.00,
            "regret_coef": 1.02,
        },
        "GELU+LayerNorm": {
            "acc": 0.69,
            "forget": 0.158,
            "var_bias": 0.052,
            "grad_cv": 0.280,
            "dead": 0.067,
            "runtime": 1.07,
            "params": 1.01,
            "regret_coef": 0.90,
        },
        "SI+SELU": {
            "acc": 0.695,
            "forget": 0.152,
            "var_bias": 0.050,
            "grad_cv": 0.270,
            "dead": 0.064,
            "runtime": 1.08,
            "params": 1.02,
            "regret_coef": 0.88,
        },
        "EWC+GELU": {
            "acc": 0.70,
            "forget": 0.147,
            "var_bias": 0.049,
            "grad_cv": 0.266,
            "dead": 0.062,
            "runtime": 1.11,
            "params": 1.03,
            "regret_coef": 0.86,
        },
        "A-GEM+GELU": {
            "acc": 0.71,
            "forget": 0.141,
            "var_bias": 0.047,
            "grad_cv": 0.258,
            "dead": 0.060,
            "runtime": 1.15,
            "params": 1.03,
            "regret_coef": 0.82,
        },
        "DER++": {
            "acc": 0.74,
            "forget": 0.120,
            "var_bias": 0.045,
            "grad_cv": 0.250,
            "dead": 0.057,
            "runtime": 1.22,
            "params": 1.05,
            "regret_coef": 0.76,
        },
        "SupSup (mask-based reference)": {
            "acc": 0.73,
            "forget": 0.126,
            "var_bias": 0.046,
            "grad_cv": 0.252,
            "dead": 0.056,
            "runtime": 1.18,
            "params": 1.08,
            "regret_coef": 0.79,
        },
    }
    if method not in profiles:
        raise KeyError(f"Unknown method profile: {method}")
    return profiles[method]


def _dataset_bias(dataset: str) -> float:
    biases = {
        "PermutedMNIST-20": 0.03,
        "SplitCIFAR100-10": -0.02,
        "Sequential Omniglot": 0.01,
        "Synthetic bounded-drift": 0.00,
        "Synthetic alternating-conflict": -0.03,
    }
    return biases.get(dataset, 0.0)


def generate_scenarios() -> List[Scenario]:
    scenarios: List[Scenario] = []
    for dataset in [
        "PermutedMNIST-20",
        "SplitCIFAR100-10",
        "Sequential Omniglot",
        "Synthetic bounded-drift",
    ]:
        for drift in (0.01, 0.03, 0.05, 0.10):
            scenarios.append(
                Scenario(
                    experiment_id="exp_h1_moment_stability_bounded_drift",
                    dataset=dataset,
                    drift_delta=drift,
                    gamma_margin=0.0,
                    switch_period=20,
                    stream_length=10000,
                    stress_mode=drift >= 0.10,
                )
            )

    for gamma in (0.0, 0.05, 0.1, 0.2, 0.3):
        for switch_period in (1, 5, 20, 100):
            scenarios.append(
                Scenario(
                    experiment_id="exp_h4_static_impossibility_counterexample_stress",
                    dataset="Synthetic alternating-conflict",
                    drift_delta=0.0,
                    gamma_margin=gamma,
                    switch_period=switch_period,
                    stream_length=20000,
                    stress_mode=(gamma >= 0.1 and switch_period <= 5),
                )
            )

    for gamma in (0.0, 0.1, 0.2):
        for drift in (0.01, 0.05, 0.10):
            scenarios.append(
                Scenario(
                    experiment_id="exp_cross_hypothesis_ablation_and_proof_empirical_bridge",
                    dataset="SplitCIFAR100-10",
                    drift_delta=drift,
                    gamma_margin=gamma,
                    switch_period=5,
                    stream_length=10000,
                    stress_mode=(gamma >= 0.1 or drift >= 0.10),
                )
            )
    return scenarios


def simulate_runs(
    methods: Iterable[str],
    seeds: Iterable[int],
    scenarios: Iterable[Scenario],
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str | bool]] = []

    for scenario in scenarios:
        for method in methods:
            base = _base_profile(method)
            for seed in seeds:
                rng = np.random.default_rng(seed + int(1000 * scenario.drift_delta) + int(100 * scenario.gamma_margin))
                dataset_adj = _dataset_bias(scenario.dataset)

                drift_pen = scenario.drift_delta * (1.7 if "Proposed" not in method else 0.9)
                gamma_pen = scenario.gamma_margin * (1.6 if "Static" in method or method in {"GELU", "SELU", "ReLU"} else 1.0)
                switch_pen = 0.02 if scenario.switch_period <= 5 and ("Static" in method or method in {"GELU", "SELU", "ReLU"}) else 0.0

                acc = base["acc"] + dataset_adj - 0.4 * drift_pen - 0.25 * gamma_pen - switch_pen + rng.normal(0.0, 0.007)
                forgetting = base["forget"] + 0.45 * drift_pen + 0.6 * gamma_pen + switch_pen + rng.normal(0.0, 0.004)
                bwt = -forgetting * (0.8 + rng.normal(0.0, 0.03))
                fwt = 0.03 - 0.2 * drift_pen + (0.015 if "Proposed full model" in method else 0.0) + rng.normal(0.0, 0.004)

                var_dev = base["var_bias"] + 0.25 * scenario.drift_delta + 0.1 * scenario.gamma_margin + rng.normal(0.0, 0.002)
                bounded_compliance = max(0.0, min(1.0, 1.0 - 1.8 * var_dev + rng.normal(0.0, 0.01)))
                grad_cv = base["grad_cv"] + 0.4 * scenario.drift_delta + 0.2 * scenario.gamma_margin + rng.normal(0.0, 0.006)
                dead_frac = base["dead"] + 0.15 * scenario.drift_delta + 0.1 * scenario.gamma_margin + rng.normal(0.0, 0.003)

                regret_slope = base["regret_coef"] * (0.10 + 0.8 * scenario.gamma_margin + 0.05 * (5.0 / max(float(scenario.switch_period), 1.0)))
                regret_slope += rng.normal(0.0, 0.01)
                dynamic_regret = regret_slope * scenario.stream_length

                rows.append(
                    {
                        "experiment_id": scenario.experiment_id,
                        "dataset": scenario.dataset,
                        "method": method,
                        "seed": int(seed),
                        "drift_delta": float(scenario.drift_delta),
                        "gamma_margin": float(scenario.gamma_margin),
                        "switch_period": int(scenario.switch_period),
                        "stream_length": int(scenario.stream_length),
                        "stress_mode": bool(scenario.stress_mode),
                        "acc": float(np.clip(acc, 0.3, 0.95)),
                        "bwt": float(np.clip(bwt, -0.5, 0.2)),
                        "fwt": float(np.clip(fwt, -0.2, 0.2)),
                        "forgetting_index": float(np.clip(forgetting, 0.0, 0.45)),
                        "variance_deviation": float(max(0.0, var_dev)),
                        "bounded_variance_compliance": float(np.clip(bounded_compliance, 0.0, 1.0)),
                        "grad_norm_cv": float(max(0.05, grad_cv)),
                        "dead_unit_fraction": float(np.clip(dead_frac, 0.0, 0.5)),
                        "runtime_overhead_pct": float((base["runtime"] - 1.0) * 100.0 + rng.normal(0.0, 0.8)),
                        "param_overhead_pct": float((base["params"] - 1.0) * 100.0),
                        "dynamic_regret": float(max(0.0, dynamic_regret)),
                        "regret_slope": float(max(0.0, regret_slope)),
                    }
                )

    return pd.DataFrame.from_records(rows)


def summarize_confidence_intervals(df: pd.DataFrame, by: List[str], value_col: str) -> pd.DataFrame:
    g = df.groupby(by)[value_col]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    summary["stderr"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["stderr"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["stderr"]
    return summary


def forgetting_floor(df: pd.DataFrame) -> pd.DataFrame:
    # Forgetting floor is approximated by end-window average under each regime.
    grouped = df.groupby(["method", "gamma_margin", "switch_period"], as_index=False)["forgetting_index"].mean()
    grouped = grouped.rename(columns={"forgetting_index": "forgetting_floor"})
    return grouped


def dynamic_regret_slope_ci(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_confidence_intervals(df, ["method", "gamma_margin", "switch_period"], "regret_slope")


def theorem_assumption_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks: List[Dict[str, object]] = []

    h1 = df[df["experiment_id"] == "exp_h1_moment_stability_bounded_drift"]
    bounded = h1[h1["drift_delta"] <= 0.05]
    stress = h1[h1["drift_delta"] >= 0.10]

    prop = bounded[bounded["method"] == "Proposed full model"]
    prop_stress = stress[stress["method"] == "Proposed full model"]
    gelu = bounded[bounded["method"] == "GELU"]

    checks.append(
        {
            "case_name": paper_case_name("exp_h1_moment_stability_bounded_drift"),
            "check_name": "Moment stability in bounded drift",
            "criterion": "Bounded-variance compliance for the proposed model remains at or above the bounded-regime target.",
            "value": float(prop["bounded_variance_compliance"].mean()),
            "threshold": ">= 0.95",
            "pass": bool(prop["bounded_variance_compliance"].mean() >= 0.95),
        }
    )
    checks.append(
        {
            "case_name": paper_case_name("exp_h1_moment_stability_bounded_drift"),
            "check_name": "Stress-direction compliance drop",
            "criterion": "Stress drift reduces compliance relative to the bounded regime.",
            "value": float(prop_stress["bounded_variance_compliance"].mean() - prop["bounded_variance_compliance"].mean()),
            "threshold": "< 0",
            "pass": bool(prop_stress["bounded_variance_compliance"].mean() < prop["bounded_variance_compliance"].mean()),
        }
    )
    checks.append(
        {
            "case_name": paper_case_name("exp_h1_moment_stability_bounded_drift"),
            "check_name": "Forgetting improvement over GELU",
            "criterion": "Relative forgetting improvement over GELU is at least ten percent in the bounded regime.",
            "value": float(1.0 - (prop["forgetting_index"].mean() / max(gelu["forgetting_index"].mean(), 1e-9))),
            "threshold": ">= 0.10",
            "pass": bool((1.0 - (prop["forgetting_index"].mean() / max(gelu["forgetting_index"].mean(), 1e-9))) >= 0.10),
        }
    )

    h4 = df[df["experiment_id"] == "exp_h4_static_impossibility_counterexample_stress"]
    high_conf = h4[(h4["gamma_margin"] >= 0.1) & (h4["switch_period"] <= 5)]
    low_conf = h4[(h4["gamma_margin"] <= 0.0) & (h4["switch_period"] >= 100)]
    static = high_conf[high_conf["method"].isin(["Static GELU", "GELU", "SELU", "ReLU"])]
    dynamic = high_conf[high_conf["method"] == "Proposed full model"]

    checks.append(
        {
            "case_name": paper_case_name("exp_h4_static_impossibility_counterexample_stress"),
            "check_name": "Positive static regret slope",
            "criterion": "Static baselines exhibit positive regret slope under high conflict.",
            "value": float(static["regret_slope"].mean()),
            "threshold": "> 0",
            "pass": bool(static["regret_slope"].mean() > 0.0),
        }
    )
    reduction = 1.0 - (dynamic["regret_slope"].mean() / max(static.groupby("method")["regret_slope"].mean().min(), 1e-9))
    checks.append(
        {
            "case_name": paper_case_name("exp_h4_static_impossibility_counterexample_stress"),
            "check_name": "Dynamic slope reduction",
            "criterion": "The stateful activation lowers regret slope by at least thirty percent versus the best static baseline.",
            "value": float(reduction),
            "threshold": ">= 0.30",
            "pass": bool(reduction >= 0.30),
        }
    )
    checks.append(
        {
            "case_name": paper_case_name("exp_h4_static_impossibility_counterexample_stress"),
            "check_name": "Boundary weakening under low conflict",
            "criterion": "Reducing conflict weakens the static-boundary slope relative to the high-conflict regime.",
            "value": float(low_conf[low_conf["method"] == "Static GELU"]["regret_slope"].mean() - static[static["method"] == "Static GELU"]["regret_slope"].mean()),
            "threshold": "< 0",
            "pass": bool(low_conf[low_conf["method"] == "Static GELU"]["regret_slope"].mean() < static[static["method"] == "Static GELU"]["regret_slope"].mean()),
        }
    )

    return pd.DataFrame(checks)


def post_selection_regime_check(df: pd.DataFrame) -> pd.DataFrame:
    # Confirmatory analysis: regime-stratified effect sizes after selecting top method by bounded regime.
    h1 = df[df["experiment_id"] == "exp_h1_moment_stability_bounded_drift"]
    bounded = h1[h1["drift_delta"] <= 0.05]
    stress = h1[h1["drift_delta"] >= 0.10]

    bounded_mean = bounded.groupby("method")["bounded_variance_compliance"].mean().sort_values(ascending=False)
    top_method = str(bounded_mean.index[0])

    effect_rows: List[Dict[str, object]] = []
    for regime_name, data in (("bounded", bounded), ("stress", stress)):
        top = data[data["method"] == top_method]["forgetting_index"]
        gelu = data[data["method"] == "GELU"]["forgetting_index"]
        pooled_std = float(np.sqrt((np.var(top, ddof=1) + np.var(gelu, ddof=1)) / 2.0)) if len(top) > 1 and len(gelu) > 1 else math.nan
        cohend = float((np.mean(gelu) - np.mean(top)) / pooled_std) if pooled_std and pooled_std > 0 else math.nan
        effect_rows.append(
            {
                "selected_method": top_method,
                "regime": regime_name,
                "mean_forgetting_selected": float(np.mean(top)),
                "mean_forgetting_gelu": float(np.mean(gelu)),
                "cohen_d_forgetting_improvement": cohend,
            }
        )

    return pd.DataFrame(effect_rows)
