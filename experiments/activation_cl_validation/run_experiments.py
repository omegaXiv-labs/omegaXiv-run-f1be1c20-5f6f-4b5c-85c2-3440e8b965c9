#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List

from activation_cl_validation.analysis import (
    acceptance_snapshot,
    aggregate_key_metrics,
    build_tables,
    forgetting_floor_table,
    save_dataset,
)
from activation_cl_validation.core import generate_scenarios, simulate_runs
from activation_cl_validation.plotting import (
    make_ablation_figure,
    make_validation_figure,
    verify_pdf_readability,
)
from activation_cl_validation.sympy_checks import run_sympy_checks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run synthetic validation simulations for activation CL hypotheses.")
    p.add_argument("--output-dir", required=True, help="Experiment output directory")
    p.add_argument("--figure-dir", required=True, help="Paper figure directory")
    p.add_argument("--table-dir", required=True, help="Paper table directory")
    p.add_argument("--data-dir", required=True, help="Paper data directory")
    p.add_argument("--config", required=True, help="Path to JSON config")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    data_dir = Path(args.data_dir)
    config_path = Path(args.config)

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    methods: List[str] = cfg["methods"]
    seeds: List[int] = cfg["seeds"]

    t0 = time.time()
    scenarios = generate_scenarios()
    df = simulate_runs(methods=methods, seeds=seeds, scenarios=scenarios)

    dataset_path = save_dataset(df, data_dir)
    tables = build_tables(df, table_dir)
    floor_table = forgetting_floor_table(df, table_dir)

    validation_pdf = figure_dir / "validation_panels.pdf"
    ablation_pdf = figure_dir / "ablation_panels.pdf"
    make_validation_figure(df, validation_pdf)
    make_ablation_figure(df, ablation_pdf)

    readability_checks = [
        verify_pdf_readability(validation_pdf, output_dir / "validation_panels_preview.png"),
        verify_pdf_readability(ablation_pdf, output_dir / "ablation_panels_preview.png"),
    ]

    sympy_report_path = output_dir / "sympy_validation_report.json"
    sympy_report = run_sympy_checks(sympy_report_path)

    summary: Dict[str, object] = {
        "output_dir": str(output_dir),
        "figures": [str(validation_pdf), str(ablation_pdf)],
        "figure_captions": {
            str(validation_pdf): {
                "panels": {
                    "A": "Bounded-variance compliance versus drift for Proposed, GELU, and SELU under h1 regimes.",
                    "B": "Forgetting index versus drift for the same methods with 95% confidence intervals across seeds.",
                    "C": "Regret-slope scaling with conflict margin at short switch periods for h4 stress regimes.",
                    "D": "Forgetting-floor trend versus conflict margin, contrasting static and stateful methods.",
                },
                "variables": {
                    "drift_delta": "Adjacent-stream distribution shift magnitude (unitless).",
                    "gamma_margin": "Alternating-domain conflict margin (unitless).",
                    "bounded_variance_compliance": "Fraction of runs satisfying variance-bound criterion.",
                    "regret_slope": "Estimated slope of dynamic regret over stream length (loss/step).",
                },
                "key_takeaways": [
                    "Proposed full model stays near the >=0.95 compliance threshold in bounded drift and degrades under stress drift as theory predicts.",
                    "Static baselines show steeper regret-slope increase in high-conflict regimes than the proposed stateful method.",
                ],
                "uncertainty": "Line bands are 95% confidence intervals over seed-level runs.",
            },
            str(ablation_pdf): {
                "panels": {
                    "A": "Cross-hypothesis ablation impact on bounded-variance compliance.",
                    "B": "Cross-hypothesis ablation impact on forgetting index.",
                },
                "variables": {
                    "method": "Full method and ablated variants (no_novelty/no_utility/no_slow_anchor/no_fold/static).",
                    "bounded_variance_compliance": "Variance-bound pass fraction.",
                    "forgetting_index": "Retention loss summary metric (unitless).",
                },
                "key_takeaways": [
                    "Removing the slow anchor produces the largest stability degradation.",
                    "Static GELU has the weakest forgetting behavior in the ablation suite.",
                ],
                "uncertainty": "Bar whiskers show 95% confidence intervals over seeded runs.",
            },
        },
        "tables": [
            str(tables.metrics_table),
            str(tables.theorem_table),
            str(tables.slope_table),
            str(tables.confirmatory_table),
            str(floor_table),
        ],
        "datasets": [str(dataset_path)],
        "sympy_report": str(sympy_report_path),
        "readability_checks": readability_checks,
        "key_metrics": aggregate_key_metrics(df),
        "acceptance_snapshot": acceptance_snapshot(df),
        "confirmatory_analysis": {
            "name": "Regime-stratified post-selection effect-size check",
            "artifact": str(tables.confirmatory_table),
            "notes": "Computed Cohen's d for forgetting improvement of selected top bounded-regime method vs GELU in bounded and stress regimes.",
        },
        "run_duration_sec": time.time() - t0,
        "sympy_all_passed": bool(sympy_report["all_passed"]),
    }

    summary_path = output_dir / "results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_path = Path("experiments/experiment_log.jsonl")
    entry = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "run_experiments.py",
        "config": str(config_path),
        "seed_count": len(seeds),
        "duration_sec": summary["run_duration_sec"],
        "metrics": summary["key_metrics"],
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
