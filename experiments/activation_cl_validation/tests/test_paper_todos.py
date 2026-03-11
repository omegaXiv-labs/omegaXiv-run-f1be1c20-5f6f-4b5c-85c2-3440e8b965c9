from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from activation_cl_validation.analysis import aggregate_key_metrics, build_tables
from activation_cl_validation.core import generate_scenarios, paper_case_name, simulate_runs, summarize_confidence_intervals
from activation_cl_validation.plotting import make_ablation_figure, make_validation_figure


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "experiments/activation_cl_validation/configs/simulation_config.json"


def _configured_methods() -> list[str]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return list(cfg["methods"])


def _simulation_frame(
    *,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    return simulate_runs(
        methods=methods or _configured_methods(),
        seeds=seeds or [7, 13, 29, 43],
        scenarios=generate_scenarios(),
    )


def test_validation_metrics_summary_is_derived_from_generated_runs(tmp_path: Path) -> None:
    df = _simulation_frame(methods=["Proposed full model", "GELU", "Static GELU", "SELU"])

    outputs = build_tables(df, tmp_path)
    actual = pd.read_csv(outputs.metrics_table)
    paper_df = df.assign(case_name=df["experiment_id"].map(paper_case_name))
    expected = summarize_confidence_intervals(
        paper_df,
        ["case_name", "dataset", "method", "drift_delta", "gamma_margin", "switch_period"],
        "forgetting_index",
    )

    sort_cols = ["case_name", "dataset", "method", "drift_delta", "gamma_margin", "switch_period"]
    pd.testing.assert_frame_equal(
        actual.sort_values(sort_cols).reset_index(drop=True),
        expected.sort_values(sort_cols).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-9,
        atol=1e-12,
    )


def test_aggregate_key_metrics_change_when_seed_schedule_changes() -> None:
    methods = ["Proposed full model", "GELU"]
    df_a = _simulation_frame(methods=methods, seeds=[7, 13])
    df_b = _simulation_frame(methods=methods, seeds=[17, 19])

    metrics_a = aggregate_key_metrics(df_a)
    metrics_b = aggregate_key_metrics(df_b)

    assert any(abs(metrics_a[key] - metrics_b[key]) > 1e-9 for key in metrics_a)


def test_todo_validation_metrics_table_uses_case_names_not_internal_ids(tmp_path: Path) -> None:
    df = _simulation_frame(methods=["Proposed full model", "GELU", "Static GELU", "SELU"])

    outputs = build_tables(df, tmp_path)
    summary = pd.read_csv(outputs.metrics_table)

    assert "experiment_id" not in summary.columns
    assert "case_name" in summary.columns
    flattened_cells = summary.astype(str).to_numpy().ravel()
    assert not any(cell.startswith("exp_") for cell in flattened_cells)


def test_todo_theorem_table_uses_descriptive_case_names(tmp_path: Path) -> None:
    df = _simulation_frame(methods=["Proposed full model", "GELU", "Static GELU", "SELU", "ReLU"])

    outputs = build_tables(df, tmp_path)
    theorem = pd.read_csv(outputs.theorem_table)

    assert "check_id" not in theorem.columns
    assert any(column in theorem.columns for column in ("check_name", "case_name", "criterion_name"))
    flattened_cells = theorem.astype(str).to_numpy().ravel()
    assert not any(cell.startswith(("h1_", "h4_")) for cell in flattened_cells)


def test_todo_claim_traceability_omits_file_names_and_paths() -> None:
    claim_table = pd.read_csv(REPO_ROOT / "paper/tables/claim_traceability.csv")

    assert "claim_id" not in claim_table.columns
    assert "claim_name" in claim_table.columns

    evidence = claim_table["primary_evidence_artifacts"].fillna("").astype(str)
    assert not evidence.str.contains(r"/", regex=True).any()
    assert not evidence.str.contains(r"\.(csv|pdf|json)\b", regex=True).any()


def test_todo_validation_figure_has_no_overlap_between_axes_and_legends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    closed_figures: list[object] = []
    monkeypatch.setattr(plt, "close", lambda fig=None: closed_figures.append(fig))

    df = _simulation_frame(methods=["Proposed full model", "GELU", "Static GELU", "SELU"])
    pdf_path = tmp_path / "validation_panels.pdf"
    make_validation_figure(df, pdf_path)

    assert closed_figures, "Expected make_validation_figure to close the saved figure."
    fig = closed_figures[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    figure_bbox = fig.bbox
    for ax in fig.axes:
        tight_bbox = ax.get_tightbbox(renderer)
        assert tight_bbox.x0 >= figure_bbox.x0
        assert tight_bbox.y0 >= figure_bbox.y0
        assert tight_bbox.x1 <= figure_bbox.x1
        assert tight_bbox.y1 <= figure_bbox.y1


def test_ablation_figure_has_no_overlapping_tick_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    closed_figures: list[object] = []
    monkeypatch.setattr(plt, "close", lambda fig=None: closed_figures.append(fig))

    df = _simulation_frame()
    pdf_path = tmp_path / "ablation_panels.pdf"
    make_ablation_figure(df, pdf_path)

    assert closed_figures, "Expected make_ablation_figure to close the saved figure."
    fig = closed_figures[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    figure_bbox = fig.bbox
    for ax in fig.axes:
        tight_bbox = ax.get_tightbbox(renderer)
        assert tight_bbox.x0 >= figure_bbox.x0
        assert tight_bbox.y0 >= figure_bbox.y0
        assert tight_bbox.x1 <= figure_bbox.x1
        assert tight_bbox.y1 <= figure_bbox.y1
