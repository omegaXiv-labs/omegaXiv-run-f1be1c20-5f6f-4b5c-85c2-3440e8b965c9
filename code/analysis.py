from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pandas as pd

from .core import (
    dynamic_regret_slope_ci,
    forgetting_floor,
    post_selection_regime_check,
    summarize_confidence_intervals,
    theorem_assumption_checks,
)


@dataclass(frozen=True)
class AnalysisOutputs:
    metrics_table: Path
    theorem_table: Path
    slope_table: Path
    confirmatory_table: Path


def build_tables(df: pd.DataFrame, table_dir: Path) -> AnalysisOutputs:
    table_dir.mkdir(parents=True, exist_ok=True)

    metrics = summarize_confidence_intervals(
        df,
        ["experiment_id", "dataset", "method", "drift_delta", "gamma_margin", "switch_period"],
        "forgetting_index",
    )
    metrics_path = table_dir / "validation_metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False)

    theorem = theorem_assumption_checks(df)
    theorem_path = table_dir / "theorem_assumption_checks.csv"
    theorem.to_csv(theorem_path, index=False)

    slope = dynamic_regret_slope_ci(df[df["experiment_id"] == "exp_h4_static_impossibility_counterexample_stress"])
    slope_path = table_dir / "regret_slope_ci.csv"
    slope.to_csv(slope_path, index=False)

    confirm = post_selection_regime_check(df)
    confirm_path = table_dir / "confirmatory_regime_check.csv"
    confirm.to_csv(confirm_path, index=False)

    return AnalysisOutputs(
        metrics_table=metrics_path,
        theorem_table=theorem_path,
        slope_table=slope_path,
        confirmatory_table=confirm_path,
    )


def aggregate_key_metrics(df: pd.DataFrame) -> Dict[str, float]:
    proposed = df[df["method"] == "Proposed full model"]
    gelu = df[df["method"] == "GELU"]

    out = {
        "proposed_acc_mean": float(proposed["acc"].mean()),
        "gelu_acc_mean": float(gelu["acc"].mean()),
        "proposed_forgetting_mean": float(proposed["forgetting_index"].mean()),
        "gelu_forgetting_mean": float(gelu["forgetting_index"].mean()),
        "forgetting_improvement_vs_gelu": float(1.0 - proposed["forgetting_index"].mean() / max(gelu["forgetting_index"].mean(), 1e-9)),
        "proposed_runtime_overhead_pct_mean": float(proposed["runtime_overhead_pct"].mean()),
        "proposed_param_overhead_pct_mean": float(proposed["param_overhead_pct"].mean()),
    }
    return out


def forgetting_floor_table(df: pd.DataFrame, table_dir: Path) -> Path:
    table_dir.mkdir(parents=True, exist_ok=True)
    floor = forgetting_floor(df[df["experiment_id"] == "exp_h4_static_impossibility_counterexample_stress"])
    out = table_dir / "forgetting_floor.csv"
    floor.to_csv(out, index=False)
    return out


def save_dataset(df: pd.DataFrame, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "simulated_validation_runs.csv"
    df.to_csv(out, index=False)
    return out


def acceptance_snapshot(df: pd.DataFrame) -> List[Dict[str, object]]:
    checks = theorem_assumption_checks(df)
    return checks.to_dict(orient="records")
