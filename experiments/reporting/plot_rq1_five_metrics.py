#!/usr/bin/env python3
"""Plot RQ1 five-metric line comparisons from existing result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("Accuracy", "Precision", "Recall", "F1", "AUC")
METHODS = (
    "EsCapturer-full",
    "TF-IDF/SGD prototype",
    "DeepCapa-adapted",
    "GPT4-API-BERT-CNN",
    "API2Vec++",
    "Nebula",
)
def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def five_flat(payload: dict) -> list[float]:
    return [payload[f"test_{name.lower()}"] for name in METRICS]


def five_nested(payload: dict, key: str = "test") -> list[float]:
    values = payload[key] if key else payload
    return [values[name.lower()] for name in METRICS]


def dataset_results(dataset: str) -> dict[str, list[float]]:
    result_dir = ROOT / "datasets_50k" / dataset / "results"
    gpt = read(result_dir / "gpt4_api_bert_cnn_metrics.json")
    full = read(result_dir / "rq1_repaired_balanced" / "full" / "seed_7" / "metrics.json")
    return {
        "EsCapturer-full": five_nested(full["test"], key=""),
        "TF-IDF/SGD prototype": five_flat(read(result_dir / "ours_metrics.json")),
        "DeepCapa-adapted": five_nested(read(result_dir / "deepcapa_adapted_metrics.json")),
        "GPT4-API-BERT-CNN": five_nested(gpt["tasks"]["detection"]),
        "API2Vec++": five_flat(read(result_dir / "api2vecpp_metrics.json")),
        "Nebula": five_flat(read(result_dir / "nebula_metrics.json")),
    }


def main() -> None:
    datasets = (
        ("quo_vadis", "Quo Vadis 50k balanced"),
        ("zenodo_11079764", "Zenodo 11079764 50k balanced"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#4D4D4D")
    markers = ("o", "s", "^", "D", "v", "P")
    for axis, (dataset, title) in zip(axes, datasets):
        results = dataset_results(dataset)
        for method, color, marker in zip(METHODS, colors, markers):
            axis.plot(
                METRICS,
                results[method],
                label=method,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4.2,
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Metric")
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.8)
        axis.set_ylim(0.75, 1.005)
        axis.tick_params(axis="x", labelrotation=25, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    figure.subplots_adjust(top=0.78, bottom=0.22, left=0.07, right=0.99, wspace=0.08)
    output = ROOT / "docs" / "figures" / "rq1_balanced_five_metrics.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
