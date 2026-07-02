#!/usr/bin/env python3
"""Generate publication-ready, non-redundant figures for the ICSE evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "figures"
DATASET_DIR = ROOT / "datasets_50k"

BLUE = "#2468A2"
ORANGE = "#D55E00"
GREEN = "#14866D"
GRAY = "#777777"
LIGHT_GRAY = "#D5D5D5"


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def metric_pair(dataset: str, method: str) -> tuple[float, float]:
    result_dir = DATASET_DIR / dataset / "results"
    if method == "EsCapturer-full":
        payload = read_json(
            result_dir / "rq1_repaired_balanced" / "full" / "seed_7" / "metrics.json"
        )["test"]
    elif method == "TF-IDF/SGD":
        payload = read_json(result_dir / "ours_metrics.json")
        return payload["test_f1"], payload["test_auc"]
    elif method == "GPT4-API-BERT-CNN":
        payload = read_json(result_dir / "gpt4_api_bert_cnn_metrics.json")["tasks"][
            "detection"
        ]["test"]
    elif method == "DeepCapa":
        payload = read_json(result_dir / "deepcapa_adapted_metrics.json")["test"]
    elif method == "APILI":
        payload = read_json(result_dir / "apili_adapted_metrics.json")["test"]
    elif method == "MME-TextCNN":
        payload = read_json(result_dir / "mme_metrics.json")["test"]
    else:
        file_name = {
            "Nebula": "nebula_metrics.json",
            "API2Vec++": "api2vecpp_metrics.json",
            "DawnGNN": "dawngnn_reimpl_metrics.json",
        }[method]
        payload = read_json(result_dir / file_name)
        return payload["test_f1"], payload["test_auc"]
    return payload["f1"], payload["auc"]


def plot_rq1() -> None:
    methods = (
        "EsCapturer-full",
        "TF-IDF/SGD",
        "DeepCapa",
        "GPT4-API-BERT-CNN",
        "APILI",
        "MME-TextCNN",
        "API2Vec++",
        "Nebula",
        "DawnGNN",
    )
    datasets = (
        ("quo_vadis", "Quo Vadis"),
        ("zenodo_11079764", "Zenodo 11079764"),
    )
    y = np.arange(len(methods))
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 3.25), sharey=True)

    for axis, (dataset, title) in zip(axes, datasets):
        pairs = [metric_pair(dataset, method) for method in methods]
        for index, (f1, auc) in enumerate(pairs):
            is_ours = index == 0
            color = BLUE if is_ours else GRAY
            axis.plot(
                [f1, auc],
                [index, index],
                color=BLUE if is_ours else LIGHT_GRAY,
                linewidth=2.2 if is_ours else 1.0,
                zorder=1,
            )
            axis.scatter(
                f1,
                index,
                s=30 if is_ours else 18,
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            axis.scatter(
                auc,
                index,
                s=34 if is_ours else 22,
                marker="D",
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlim(0.775, 1.005)
        axis.set_xticks((0.80, 0.85, 0.90, 0.95, 1.00))
        axis.grid(axis="x", color="#E8E8E8", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xlabel("Score")

    axes[0].set_yticks(y, labels=methods)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=GRAY, label="F1"),
        plt.Line2D([], [], marker="D", linestyle="none", color=GRAY, label="AUC"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.56, -0.01),
        frameon=False,
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.22, right=0.99, top=0.90, bottom=0.18, wspace=0.08)
    figure.savefig(FIGURE_DIR / "rq1_balanced_paired_scores.pdf", bbox_inches="tight")
    plt.close(figure)


def load_ablation(dataset: str) -> list[dict[str, str]]:
    path = DATASET_DIR / dataset / "results" / "rq3_balanced_ablation" / "summary_seed7.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_rq3() -> None:
    variants = (
        (("w/o semantic elements", "wo_semantic_elements"), "Semantic elements"),
        (("w/o graph view", "wo_graph_view"), "Graph view"),
        (
            ("w/o interleaving prior", "wo_interleaving_prior_in_gating"),
            "Interleaving prior",
        ),
    )
    datasets = (
        ("quo_vadis", "Quo Vadis"),
        ("zenodo_11079764", "Zenodo 11079764"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 3.65), sharex=True)

    for axis, (dataset, title) in zip(axes, datasets):
        rows = load_ablation(dataset)
        by_variant = {row["Variant"]: row for row in rows}
        values: dict[str, float] = {}
        for aliases, label in variants:
            for key in aliases:
                if key in by_variant:
                    values[label] = 100 * float(by_variant[key]["Delta_F1"])
                    break
        labels = ("Semantic elements", "Graph view", "Interleaving prior")
        deltas = [values[label] for label in labels]
        positions = np.arange(len(labels))
        colors = [GREEN if value >= 0 else ORANGE for value in deltas]
        axis.barh(positions, deltas, color=colors, height=0.58)
        axis.axvline(0, color="#333333", linewidth=0.75)
        for position, value in zip(positions, deltas):
            axis.text(
                value + (0.06 if value >= 0 else -0.06),
                position,
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7,
            )
        axis.set_yticks(positions, labels=labels)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#E8E8E8", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_xlim(-1.45, 0.35)
    axes[-1].set_xlabel(r"F1 change from Full (percentage points)")
    figure.subplots_adjust(left=0.39, right=0.98, top=0.93, bottom=0.14, hspace=0.38)
    figure.savefig(FIGURE_DIR / "rq3_ablation_delta_f1.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_rq4() -> None:
    datasets = (
        ("quo_vadis", "Quo Vadis"),
        ("zenodo_11079764", "Zenodo 11079764"),
    )
    perturbations = (
        ("insertion", "Insertion"),
        ("deletion", "Deletion"),
        ("local_reordering", "Local reorder"),
    )
    intensities = (0.1, 0.2, 0.3)
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 3.85))

    for axis, (dataset, title) in zip(axes, datasets):
        rows = read_json(
            DATASET_DIR
            / dataset
            / "results"
            / "rq4_robustness"
            / "full"
            / "seed_7"
            / "summary_seed7.json"
        )
        lookup = {
            (row["Perturbation"], float(row["Intensity"])): -100 * row["Delta_F1"]
            for row in rows
        }
        matrix = np.array(
            [[lookup[(key, intensity)] for intensity in intensities] for key, _ in perturbations]
        )
        image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=32, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 15 else "#222222",
                    fontsize=8,
                    fontweight="bold" if value >= 15 else "normal",
                )
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(np.arange(3), labels=("10%", "20%", "30%"))
        axis.set_yticks(np.arange(3), labels=[label for _, label in perturbations])
        axis.set_xlabel("Perturbation intensity")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    colorbar = figure.colorbar(image, ax=axes, fraction=0.045, pad=0.04)
    colorbar.set_label("F1 loss (percentage points)")
    figure.subplots_adjust(left=0.27, right=0.83, top=0.94, bottom=0.10, hspace=0.58)
    figure.savefig(FIGURE_DIR / "rq4_robustness_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_rq1()
    plot_rq3()
    plot_rq4()


if __name__ == "__main__":
    main()
