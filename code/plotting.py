from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="deep", context="talk")


METHOD_LABELS = {
    "Proposed full model": "Proposed",
    "Proposed without novelty term": "No novelty",
    "Proposed without utility term": "No utility",
    "Proposed without slow anchor": "No slow anchor",
    "Proposed without folding loss": "No folding loss",
    "Static GELU": "Static GELU",
    "GELU": "GELU",
    "SELU": "SELU",
    "ReLU": "ReLU",
    "GELU+LayerNorm": "GELU + LayerNorm",
    "SI+SELU": "SI + SELU",
    "EWC+GELU": "EWC + GELU",
    "A-GEM+GELU": "A-GEM + GELU",
    "DER++": "DER++",
    "SupSup (mask-based reference)": "SupSup",
}


def _with_plot_labels(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["plot_method"] = plot_df["method"].map(METHOD_LABELS).fillna(plot_df["method"])
    return plot_df


def make_validation_figure(df: pd.DataFrame, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    plot_df = _with_plot_labels(df)
    h1 = plot_df[plot_df["experiment_id"] == "exp_h1_moment_stability_bounded_drift"]
    h4 = plot_df[plot_df["experiment_id"] == "exp_h4_static_impossibility_counterexample_stress"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)

    ax = axes[0, 0]
    subset = h1[h1["method"].isin(["Proposed full model", "GELU", "SELU"])]
    sns.lineplot(
        data=subset,
        x="drift_delta",
        y="bounded_variance_compliance",
        hue="plot_method",
        style="plot_method",
        marker="o",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_title("Bounded-Variance Compliance vs Drift")
    ax.set_xlabel("Drift Delta (unitless)")
    ax.set_ylabel("Compliance Rate (0-1)")
    ax.legend(title="Method", loc="lower left", fontsize=11, title_fontsize=12)

    ax = axes[0, 1]
    subset2 = h1[h1["method"].isin(["Proposed full model", "GELU", "SELU"])]
    sns.lineplot(
        data=subset2,
        x="drift_delta",
        y="forgetting_index",
        hue="plot_method",
        style="plot_method",
        marker="o",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_title("Forgetting vs Drift")
    ax.set_xlabel("Drift Delta (unitless)")
    ax.set_ylabel("Forgetting Index (unitless)")
    ax.legend(title="Method", loc="upper left", fontsize=11, title_fontsize=12)

    ax = axes[1, 0]
    subset3 = h4[h4["method"].isin(["Proposed full model", "Static GELU", "GELU", "SELU"]) & (h4["switch_period"] <= 5)]
    sns.lineplot(
        data=subset3,
        x="gamma_margin",
        y="regret_slope",
        hue="plot_method",
        style="plot_method",
        marker="o",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_title("Regret Slope vs Conflict Margin")
    ax.set_xlabel("Gamma Margin (unitless)")
    ax.set_ylabel("Regret Slope (loss / step)")
    ax.legend(title="Method", loc="upper left", fontsize=11, title_fontsize=12)

    ax = axes[1, 1]
    subset4 = h4[h4["method"].isin(["Proposed full model", "Static GELU", "GELU", "SELU"])]
    sns.lineplot(
        data=subset4,
        x="gamma_margin",
        y="forgetting_index",
        hue="plot_method",
        style="plot_method",
        marker="o",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_title("Forgetting Floor Trend vs Conflict")
    ax.set_xlabel("Gamma Margin (unitless)")
    ax.set_ylabel("Forgetting Index (unitless)")
    ax.legend(title="Method", loc="upper left", fontsize=11, title_fontsize=12)

    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def make_ablation_figure(df: pd.DataFrame, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    cross = _with_plot_labels(df[df["experiment_id"] == "exp_cross_hypothesis_ablation_and_proof_empirical_bridge"])
    order = [
        "Proposed",
        "No novelty",
        "No utility",
        "No slow anchor",
        "No folding loss",
        "GELU + LayerNorm",
        "SI + SELU",
        "EWC + GELU",
        "A-GEM + GELU",
        "DER++",
        "Static GELU",
        "GELU",
        "SELU",
        "ReLU",
        "SupSup",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    ax = axes[0]
    sns.barplot(
        data=cross,
        y="plot_method",
        x="bounded_variance_compliance",
        order=order,
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_title("Ablation: Stability Compliance")
    ax.set_xlabel("Compliance Rate (0-1)")
    ax.set_ylabel("Model Variant")

    ax = axes[1]
    sns.barplot(
        data=cross,
        y="plot_method",
        x="forgetting_index",
        order=order,
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_title("Ablation: Forgetting")
    ax.set_xlabel("Forgetting Index (unitless)")
    ax.set_ylabel("Model Variant")

    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def verify_pdf_readability(pdf_path: Path, raster_png: Path) -> Dict[str, float | int | str]:
    # Rasterize first page to verify the PDF opens and has expected dimensions.
    import pypdfium2 as pdfium
    import numpy as np

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[0]
    bitmap = page.render(scale=2.0)
    image = bitmap.to_pil()
    raster_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(raster_png)

    arr = np.asarray(image.convert("L"), dtype=float)
    contrast = float(arr.std())
    width, height = image.size
    return {
        "pdf_path": str(pdf_path),
        "raster_path": str(raster_png),
        "width_px": int(width),
        "height_px": int(height),
        "grayscale_std": contrast,
    }
