#!/usr/bin/env python3
"""Collect paper-facing evaluation evidence without inventing missing values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("quo_vadis", "zenodo_11079764")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="datasets_50k")
    parser.add_argument(
        "--output",
        default="datasets_50k/evaluation_evidence.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_json(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def collect_metrics_tree(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("**/metrics.json")):
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        records.append({"source": str(path), "metrics": value})
    return records


def collect_dataset(root: Path, dataset: str) -> dict:
    dataset_root = root / dataset
    stats_path = next((dataset_root / "data").glob("*.stats.json"))
    stats = load_json(stats_path)
    with (dataset_root / "results" / "baseline_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        result_rows = list(csv.DictReader(handle))
    results = {}
    for row in result_rows:
        results[row["method"]] = {
            key: float(row[key])
            for key in (
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_auc",
                "runtime_seconds",
            )
            if row.get(key)
        }
        results[row["method"]]["status"] = row.get("status", "")
        results[row["method"]]["implementation"] = row.get("implementation", "")

    evidence = {
        "stats_source": str(stats_path),
        "dataset": stats,
        "diagnostic_rq1_source": str(dataset_root / "results" / "baseline_results.csv"),
        "diagnostic_rq1": results,
        "diagnostic_warning": (
            "These rows use the preliminary embedded split and do not constitute "
            "confirmatory EsCapturer-full evidence."
        ),
    }
    safe_root = dataset_root / "leakage_safe"
    evidence["leakage_safe"] = {
        "manifest": optional_json(safe_root / "manifest.json"),
        "dataset_quality": optional_json(safe_root / "dataset_quality.json"),
        "leakage_audit": optional_json(safe_root / "leakage_audit.json"),
    }
    result_root = dataset_root / "results"
    evidence["confirmatory_runs"] = {
        "full_model": collect_metrics_tree(result_root / "full_model"),
        "standard_baselines": collect_metrics_tree(result_root / "standard_baselines"),
        "graph_baselines": collect_metrics_tree(result_root / "graph_baselines"),
        "graph_baselines": collect_metrics_tree(result_root / "graph_baselines"),
        "ablation": collect_metrics_tree(result_root / "ablation"),
        "unknown_family": collect_metrics_tree(result_root / "unknown_family"),
        "profile": collect_metrics_tree(result_root / "profile"),
    }
    extraction_path = dataset_root / "results" / "llm_behavior_extraction_summary.json"
    cache_path = dataset_root / "results" / "llm_behavior_template_cache.json"
    if extraction_path.exists():
        evidence["semantic_extraction"] = load_json(extraction_path)
    if cache_path.exists():
        cache = load_json(cache_path)
        evidence["semantic_cache"] = {
            "source": str(cache_path),
            "entries": len(cache),
            "bytes": cache_path.stat().st_size,
        }
    return evidence


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    evidence = {
        "datasets": {
            dataset: collect_dataset(root, dataset)
            for dataset in DATASETS
        },
        "missing_experiments": {
            "leakage_safe_splits": "[TODO: materialize verified splits under each dataset/leakage_safe directory]",
            "rq1_full_dual_view": "[TODO: run EsCapturer-full on both leakage-safe datasets with seeds 7,17,27]",
            "rq1_baselines": "[TODO: rerun every baseline on identical leakage-safe splits with seeds 7,17,27]",
            "rq2_unknown_family": "[TODO: run Quo Vadis and Zenodo family-disjoint folds]",
            "rq3_ablation": "[TODO: run ten variants on both datasets with three seeds]",
            "rq3_extraction_quality": "[TODO: complete two-annotator quality study]",
            "rq4_stage_efficiency": "[TODO: run instrumented full-model profiling on common hardware]",
            "rq5_view_weights": "[TODO: analyze exported per-pair view weights with group-level resampling]",
        },
        "script_entries": {
            "leakage_safe": "experiments/data_preparation/prepare_leakage_safe_splits.py",
            "family_disjoint": "experiments/data_preparation/prepare_family_disjoint_protocol.py",
            "standard_baselines": "experiments/runners/run_sequence_baselines.py",
            "graph_baselines": "experiments/runners/run_graph_baselines.py",
            "graph_baselines": "experiments/runners/run_graph_baselines.py",
            "full_model": "experiments/runners/run_escapture_evaluation.py",
            "paired_comparison": "experiments/reporting/compare_predictions.py",
            "view_weight_analysis": "experiments/reporting/analyze_view_weights.py",
            "extraction_annotation": "experiments/reporting/sample_extraction_annotation.py",
        },
        "consistency_warnings": [
            "Current 'ours' rows are TF-IDF/SGD classifier results, not full dual-view model results.",
            "Legacy baselines do not all use the same threshold-selection policy.",
            "Zenodo source provenance must be reconciled with the pending-archive manifest.",
            "Some baseline READMEs still describe obsolete 500/500 pilot protocols.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
