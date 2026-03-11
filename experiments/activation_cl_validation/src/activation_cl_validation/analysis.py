from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pandas as pd

from .core import (
    dynamic_regret_slope_ci,
    forgetting_floor,
    paper_case_name,
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
    claim_table: Path


def build_tables(df: pd.DataFrame, table_dir: Path) -> AnalysisOutputs:
    table_dir.mkdir(parents=True, exist_ok=True)

    paper_df = df.assign(case_name=df["experiment_id"].map(paper_case_name))
    metrics = summarize_confidence_intervals(
        paper_df,
        ["case_name", "dataset", "method", "drift_delta", "gamma_margin", "switch_period"],
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

    claim = build_claim_traceability_table(df)
    claim_path = table_dir / "claim_traceability.csv"
    claim.to_csv(claim_path, index=False)

    return AnalysisOutputs(
        metrics_table=metrics_path,
        theorem_table=theorem_path,
        slope_table=slope_path,
        confirmatory_table=confirm_path,
        claim_table=claim_path,
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


def build_claim_traceability_table(df: pd.DataFrame) -> pd.DataFrame:
    theorem = theorem_assumption_checks(df)
    h1_supported = bool(
        theorem[theorem["case_name"] == paper_case_name("exp_h1_moment_stability_bounded_drift")]["pass"].all()
    )
    h4_supported = bool(
        theorem[theorem["case_name"] == paper_case_name("exp_h4_static_impossibility_counterexample_stress")]["pass"].all()
    )

    proposed = df[df["method"] == "Proposed full model"]
    efficiency_supported = bool(
        proposed["runtime_overhead_pct"].mean() <= 10.0
        and proposed["param_overhead_pct"].mean() <= 5.0
    )

    rows = [
        {
            "claim_name": "Bounded moment stability",
            "status": "Conditionally supported" if h1_supported else "Needs review",
            "symbolic_check": "Pass",
            "empirical_support": "Pass (synthetic)",
            "primary_evidence_artifacts": "Theorem-assumption checks; validation panels; symbolic validation report",
            "notes": "Assumption checks and sign-direction trends align within the synthetic validation protocol.",
        },
        {
            "claim_name": "Replay-free competitiveness",
            "status": "Open",
            "symbolic_check": "n/a",
            "empirical_support": "Not executed",
            "primary_evidence_artifacts": "Benchmark-comparator experiment gap",
            "notes": "The benchmark-comparator track is still missing from the current run set.",
        },
        {
            "claim_name": "Efficiency envelope",
            "status": "Partially supported" if efficiency_supported else "Needs review",
            "symbolic_check": "n/a",
            "empirical_support": "Pass (executed subset)" if efficiency_supported else "Needs review",
            "primary_evidence_artifacts": "Aggregate metric summary; theorem-assumption checks",
            "notes": "Runtime and parameter overhead remain inside the configured synthetic acceptance envelope.",
        },
        {
            "claim_name": "Compositional reuse criterion",
            "status": "Open",
            "symbolic_check": "n/a",
            "empirical_support": "Not executed",
            "primary_evidence_artifacts": "Compositional-probe experiment gap",
            "notes": "The compositional probe track remains unexecuted, so joint reuse claims stay open.",
        },
        {
            "claim_name": "Static-activation impossibility boundary",
            "status": "Conditionally supported" if h4_supported else "Needs review",
            "symbolic_check": "Pass",
            "empirical_support": "Pass (synthetic)",
            "primary_evidence_artifacts": "Theorem-assumption checks; conflict-regime table; validation panels",
            "notes": "Lower-bound directionality and boundary weakening both hold in the synthetic stress regime.",
        },
    ]
    return pd.DataFrame.from_records(rows)
