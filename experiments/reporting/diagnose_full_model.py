#!/usr/bin/env python3
"""Diagnose EsCapturer-full against an API n-gram SVM on one fixed split.

The script is deliberately read-only with respect to model outputs. It joins
predictions to the compact dataset by sample_id, reconstructs the model's
behavior units, and writes machine-readable JSON/CSV evidence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from escapture.llm_behavior_extractor import FrozenTemplateBehaviorExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--full-predictions", required=True)
    parser.add_argument("--svm-predictions", required=True)
    parser.add_argument("--full-metrics")
    parser.add_argument("--svm-metrics")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-units", type=int, required=True)
    parser.add_argument(
        "--unit-selection",
        choices=["prefix", "uniform-cover"],
        required=True,
    )
    parser.add_argument("--representative-count", type=int, default=10)
    parser.add_argument(
        "--reference-diagnosis",
        help="Optional diagnosis.json from the pre-fix run; writes p2_comparison JSON/CSV.",
    )
    return parser.parse_args()


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_test_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "test":
                rows[str(row["sample_id"])] = row
    return rows


def load_metrics(path: str | None, prediction_rows: list[dict]) -> dict:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload.get("test", payload)
    labels = np.asarray([int(row["label"]) for row in prediction_rows])
    predictions = np.asarray([int(row["prediction"]) for row in prediction_rows])
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": None}


def confusion(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["label"]) for row in rows])
    predictions = np.asarray([int(row["prediction"]) for row in rows])
    return {
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
    }


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "q1": None,
                "median": None, "q3": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def graph_density(groups: list[dict]) -> float:
    edge_total = 0
    possible_total = 0
    for group in groups:
        calls = list(group.get("syscalls", []))
        nodes = set(calls)
        edges = set(zip(calls[:-1], calls[1:]))
        edge_total += len(edges)
        possible_total += len(nodes) * len(nodes)
    return edge_total / max(possible_total, 1)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_rows = load_test_rows(Path(args.data))
    full_rows = read_jsonl_gz(Path(args.full_predictions))
    svm_rows = read_jsonl_gz(Path(args.svm_predictions))
    full = {str(row["sample_id"]): row for row in full_rows}
    svm = {str(row["sample_id"]): row for row in svm_rows}
    ids = sorted(set(data_rows) & set(full) & set(svm))
    if set(ids) != set(data_rows) or set(ids) != set(full) or set(ids) != set(svm):
        raise ValueError(
            "Data and prediction artifacts must contain identical test sample IDs "
            f"(data={len(data_rows)}, full={len(full)}, svm={len(svm)}, joined={len(ids)})"
        )

    extractor = FrozenTemplateBehaviorExtractor()
    samples = []
    paired_errors = []
    for sample_id in ids:
        source = data_rows[sample_id]
        full_row = full[sample_id]
        svm_row = svm[sample_id]
        label = int(source["label"] == "malware")
        if label != int(full_row["label"]) or label != int(svm_row["label"]):
            raise ValueError(f"Label mismatch for {sample_id}")
        seq = list(source.get("api_seq", []))
        model_seq = seq[: args.max_seq_len]
        elements = extractor.extract_sequence(
            model_seq, sample_id=sample_id, max_len=args.max_seq_len
        )
        units = extractor.build_units_from_elements(
            elements,
            sample_id=sample_id,
            max_units=args.max_units,
            unit_selection=args.unit_selection,
        )
        groups = [unit.to_group() for unit in units]
        retained = sum(len(group["syscalls"]) for group in groups)
        full_correct = int(full_row["prediction"]) == label
        svm_correct = int(svm_row["prediction"]) == label
        if not full_correct and svm_correct:
            paired_status = "full_wrong_svm_right"
        elif full_correct and not svm_correct:
            paired_status = "full_right_svm_wrong"
        elif full_correct:
            paired_status = "both_right"
        else:
            paired_status = "both_wrong"
        record = {
            "sample_id": sample_id,
            "family": source.get("family", "unknown"),
            "label": label,
            "trace_call_count": len(seq),
            "model_input_call_count": len(model_seq),
            "retained_call_count": retained,
            "retained_call_coverage": retained / max(len(model_seq), 1),
            "full_trace_coverage": retained / max(len(seq), 1),
            "unit_count": len(groups),
            "graph_density": graph_density(groups),
            "full_score": float(full_row["score"]),
            "full_prediction": int(full_row["prediction"]),
            "svm_score": float(svm_row["score"]),
            "svm_prediction": int(svm_row["prediction"]),
            "paired_status": paired_status,
        }
        samples.append(record)
        if paired_status in {"full_wrong_svm_right", "full_right_svm_wrong"}:
            paired_errors.append(record)

    family_rows = []
    families = sorted({row["family"] for row in samples})
    for family in families:
        subset = [row for row in samples if row["family"] == family]
        for method in ("full", "svm"):
            correct = sum(
                int(row[f"{method}_prediction"] == row["label"]) for row in subset
            )
            positive = [row for row in subset if row["label"] == 1]
            recall = (
                sum(int(row[f"{method}_prediction"] == 1) for row in positive)
                / len(positive)
                if positive
                else None
            )
            family_rows.append(
                {
                    "family": family,
                    "method": method,
                    "support": len(subset),
                    "malware_support": len(positive),
                    "correct": correct,
                    "accuracy": correct / len(subset),
                    "recall": recall,
                }
            )

    full_metrics = load_metrics(args.full_metrics, full_rows)
    svm_metrics = load_metrics(args.svm_metrics, svm_rows)
    thresholds = {
        "full": full_metrics.get("threshold"),
        "svm": svm_metrics.get("threshold"),
    }
    score_distribution = {}
    for method in ("full", "svm"):
        threshold = thresholds[method]
        score_distribution[method] = {
            "threshold": threshold,
            "overall": quantiles([row[f"{method}_score"] for row in samples]),
            "benign": quantiles(
                [row[f"{method}_score"] for row in samples if row["label"] == 0]
            ),
            "malware": quantiles(
                [row[f"{method}_score"] for row in samples if row["label"] == 1]
            ),
            "margin_from_threshold": (
                quantiles(
                    [row[f"{method}_score"] - float(threshold) for row in samples]
                )
                if threshold is not None and math.isfinite(float(threshold))
                else None
            ),
        }

    representative = []
    full_threshold = thresholds["full"]
    if full_threshold is not None:
        false_positives = sorted(
            [
                row for row in samples
                if row["label"] == 0 and row["full_prediction"] == 1
            ],
            key=lambda row: row["full_score"] - float(full_threshold),
            reverse=True,
        )[: args.representative_count]
        false_negatives = sorted(
            [
                row for row in samples
                if row["label"] == 1 and row["full_prediction"] == 0
            ],
            key=lambda row: float(full_threshold) - row["full_score"],
            reverse=True,
        )[: args.representative_count]
        for error_type, rows in (("FP", false_positives), ("FN", false_negatives)):
            for row in rows:
                enriched = dict(row)
                enriched["error_type"] = error_type
                enriched["api_preview"] = " | ".join(
                    data_rows[row["sample_id"]].get("api_seq", [])[:25]
                )
                representative.append(enriched)

    paired_counts = defaultdict(int)
    for row in samples:
        paired_counts[row["paired_status"]] += 1
    by_label = {}
    for label_name, label in (("benign", 0), ("malware", 1)):
        subset = [row for row in samples if row["label"] == label]
        by_label[label_name] = {
            "retained_call_coverage": quantiles(
                [row["retained_call_coverage"] for row in subset]
            ),
            "unit_count": quantiles([row["unit_count"] for row in subset]),
            "graph_density": quantiles([row["graph_density"] for row in subset]),
        }

    summary = {
        "config": {
            "data": args.data,
            "full_predictions": args.full_predictions,
            "svm_predictions": args.svm_predictions,
            "max_seq_len": args.max_seq_len,
            "max_units": args.max_units,
            "unit_selection": args.unit_selection,
        },
        "sample_count": len(samples),
        "confusion_matrix": {
            "full": confusion(full_rows),
            "svm": confusion(svm_rows),
        },
        "metrics": {
            "full": {
                key: full_metrics.get(key)
                for key in (
                    "accuracy", "precision", "recall", "f1", "auc",
                    "threshold", "tp", "fp", "fn", "tn",
                )
                if key in full_metrics
            },
            "svm": {
                key: svm_metrics.get(key)
                for key in (
                    "accuracy", "precision", "recall", "f1", "auc",
                    "threshold", "tp", "fp", "fn", "tn",
                )
                if key in svm_metrics
            },
        },
        "family_recall": family_rows,
        "paired_error_counts": dict(paired_counts),
        "retained_call_coverage": quantiles(
            [row["retained_call_coverage"] for row in samples]
        ),
        "unit_count": quantiles([row["unit_count"] for row in samples]),
        "graph_density": quantiles([row["graph_density"] for row in samples]),
        "by_label": by_label,
        "score_distribution": score_distribution,
        "artifacts": {
            "sample_diagnostics": "sample_diagnostics.csv",
            "family_recall": "family_recall.csv",
            "paired_errors": "paired_errors.csv",
            "representative_errors": "representative_errors.csv",
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "sample_diagnostics.csv", samples)
    write_csv(out_dir / "family_recall.csv", family_rows)
    write_csv(out_dir / "paired_errors.csv", paired_errors)
    write_csv(out_dir / "representative_errors.csv", representative)
    (out_dir / "diagnosis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    if args.reference_diagnosis:
        reference = json.loads(
            Path(args.reference_diagnosis).read_text(encoding="utf-8")
        )
        comparison_rows = []
        reference_metrics = reference.get("metrics", {}).get("full", {})
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            before = reference_metrics.get(metric)
            after = summary["metrics"]["full"].get(metric)
            comparison_rows.append(
                {
                    "scope": "overall",
                    "measure": metric,
                    "before": before,
                    "after": after,
                    "delta": (
                        float(after) - float(before)
                        if before is not None and after is not None
                        else None
                    ),
                }
            )
        for measure, before, after in (
            (
                "retained_call_median_coverage",
                reference["retained_call_coverage"]["median"],
                summary["retained_call_coverage"]["median"],
            ),
            (
                "false_positives",
                reference["confusion_matrix"]["full"]["fp"],
                summary["confusion_matrix"]["full"]["fp"],
            ),
            (
                "false_negatives",
                reference["confusion_matrix"]["full"]["fn"],
                summary["confusion_matrix"]["full"]["fn"],
            ),
        ):
            comparison_rows.append(
                {
                    "scope": "overall",
                    "measure": measure,
                    "before": before,
                    "after": after,
                    "delta": float(after) - float(before),
                }
            )
        reference_family = {
            (row["family"], row["method"]): row
            for row in reference.get("family_recall", [])
        }
        for row in family_rows:
            if row["method"] != "full" or row["recall"] is None:
                continue
            before_row = reference_family.get((row["family"], "full"))
            before = before_row.get("recall") if before_row else None
            comparison_rows.append(
                {
                    "scope": f"family:{row['family']}",
                    "measure": "recall",
                    "before": before,
                    "after": row["recall"],
                    "delta": (
                        float(row["recall"]) - float(before)
                        if before is not None
                        else None
                    ),
                }
            )
        comparison = {
            "reference_diagnosis": args.reference_diagnosis,
            "current_diagnosis": str(out_dir / "diagnosis.json"),
            "rows": comparison_rows,
        }
        (out_dir / "p2_comparison.json").write_text(
            json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8"
        )
        write_csv(
            out_dir / "p2_comparison.csv",
            comparison_rows,
            ["scope", "measure", "before", "after", "delta"],
        )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
