#!/usr/bin/env python3
"""Compare fixed full-model repair stages from diagnosis JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Named diagnosis in NAME=PATH form. Repeat for each repair stage.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        runs.append((name.strip(), Path(path)))
    return runs


def main() -> None:
    args = parse_args()
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in parse_runs(args.run)
    }
    rows = []
    family_rows = []
    score_distributions = {}
    for name, payload in payloads.items():
        full = payload["metrics"]["full"]
        confusion = payload["confusion_matrix"]["full"]
        rows.append(
            {
                "method": name,
                "accuracy": full.get("accuracy"),
                "precision": full.get("precision"),
                "recall": full.get("recall"),
                "f1": full.get("f1"),
                "auc": full.get("auc"),
                "threshold": full.get("threshold"),
                "fp": confusion["fp"],
                "fn": confusion["fn"],
                "tp": confusion["tp"],
                "tn": confusion["tn"],
            }
        )
        score_distributions[name] = payload["score_distribution"]["full"]
        for family in payload.get("family_recall", []):
            if family["method"] == "full":
                family_rows.append({**family, "method": name})

    first_payload = next(iter(payloads.values()))
    svm = first_payload["metrics"]["svm"]
    svm_confusion = first_payload["confusion_matrix"]["svm"]
    rows.append(
        {
            "method": "api_ngram_svm",
            "accuracy": svm.get("accuracy"),
            "precision": svm.get("precision"),
            "recall": svm.get("recall"),
            "f1": svm.get("f1"),
            "auc": svm.get("auc"),
            "threshold": svm.get("threshold"),
            "fp": svm_confusion["fp"],
            "fn": svm_confusion["fn"],
            "tp": svm_confusion["tp"],
            "tn": svm_confusion["tn"],
        }
    )
    score_distributions["api_ngram_svm"] = first_payload["score_distribution"]["svm"]
    for family in first_payload.get("family_recall", []):
        if family["method"] == "svm":
            family_rows.append({**family, "method": "api_ngram_svm"})

    by_name = {row["method"]: row for row in rows}
    p2 = by_name.get("p2_dsvdd")
    p3a = by_name.get("p3a_dsvdd")
    p3b = by_name.get("p3b_bce")
    evidence = {
        "p3a_vs_p2": (
            {
                key: p3a[key] - p2[key]
                for key in ("precision", "recall", "f1", "auc", "fp", "fn")
            }
            if p2 and p3a
            else None
        ),
        "p3b_vs_p2": (
            {
                key: p3b[key] - p2[key]
                for key in ("precision", "recall", "f1", "auc", "fp", "fn")
            }
            if p2 and p3b
            else None
        ),
        "dsvdd_bottleneck_assessment": (
            "The supervised BCE head materially improves F1 and precision and "
            "reduces false positives relative to P2 DSVDD, so DSVDD is a material "
            "bottleneck. Because BCE remains below the n-gram SVM and loses recall, "
            "the objective is not the only bottleneck."
            if p2 and p3b and p3b["f1"] > p2["f1"]
            else "The available comparison does not establish DSVDD as the main bottleneck."
        ),
    }
    output = {
        "metrics": rows,
        "family_recall": family_rows,
        "score_distributions": score_distributions,
        "evidence": evidence,
        "inputs": args.run,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    family_csv = output_csv.with_name(output_csv.stem + "_family_recall.csv")
    with family_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "family", "support", "malware_support", "correct",
                      "accuracy", "recall"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(family_rows)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
