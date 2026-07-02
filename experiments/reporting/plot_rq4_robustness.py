#!/usr/bin/env python3
"""Plot clean-trained RQ4 robustness trends from verified summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATASETS = (
    ("quo_vadis", "Quo Vadis", {"F1": 0.9626, "AUC": 0.9935}),
    ("zenodo_11079764", "Zenodo 11079764", {"F1": 0.9409, "AUC": 0.9791}),
)
PERTURBATIONS = (
    ("insertion", "Insertion", "#D55E00", "o"),
    ("deletion", "Deletion", "#0072B2", "s"),
    ("local_reordering", "Local reordering", "#009E73", "^"),
)


def load_summary(dataset: str) -> list[dict]:
    path = (
        ROOT
        / "datasets_50k"
        / dataset
        / "results"
        / "rq4_robustness"
        / "full"
        / "seed_7"
        / "summary_seed7.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(4.4, 5.0), sharex=True)
    for axis, (dataset, title, clean) in zip(axes, DATASETS):
        rows = load_summary(dataset)
        for key, label, color, marker in PERTURBATIONS:
            selected = sorted(
                (row for row in rows if row["Perturbation"] == key),
                key=lambda row: row["Intensity"],
            )
            x_values = [0, *[int(round(row["Intensity"] * 100)) for row in selected]]
            y_values = [clean["F1"], *[row["F1"] for row in selected]]
            axis.plot(
                x_values,
                y_values,
                label=label,
                color=color,
                marker=marker,
                linewidth=1.7,
                markersize=4.5,
            )
        axis.set_title(title, fontsize=9)
        axis.set_ylabel("F1")
        axis.set_ylim(0.6, 1.005)
        axis.set_xticks((0, 10, 20, 30))
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.8)
        axis.tick_params(labelsize=8)
    axes[-1].set_xlabel("Perturbation intensity (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7)
    figure.subplots_adjust(top=0.91, bottom=0.10, left=0.14, right=0.98, hspace=0.32)
    output = ROOT / "docs" / "figures" / "rq4_robustness_trends.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
