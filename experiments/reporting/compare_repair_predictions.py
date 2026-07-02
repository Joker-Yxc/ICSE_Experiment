#!/usr/bin/env python3
"""Create paired correctness/error transitions between two repair runs."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--before-name", required=True)
    parser.add_argument("--after-name", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path: str) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            str(row["sample_id"]): row
            for row in map(json.loads, handle)
        }


def main() -> None:
    args = parse_args()
    before = load(args.before)
    after = load(args.after)
    if set(before) != set(after):
        raise ValueError("Prediction files do not contain identical sample IDs")
    transitions = Counter()
    label_transitions = {"benign": Counter(), "malware": Counter()}
    examples = []
    for sample_id in sorted(before):
        old = before[sample_id]
        new = after[sample_id]
        label = int(old["label"])
        if label != int(new["label"]):
            raise ValueError(f"Label mismatch for {sample_id}")
        old_correct = int(old["prediction"]) == label
        new_correct = int(new["prediction"]) == label
        if old_correct and new_correct:
            transition = "both_right"
        elif old_correct:
            transition = "before_right_after_wrong"
        elif new_correct:
            transition = "before_wrong_after_right"
        else:
            transition = "both_wrong"
        transitions[transition] += 1
        label_transitions["malware" if label else "benign"][transition] += 1
        if transition in {
            "before_right_after_wrong", "before_wrong_after_right"
        }:
            examples.append(
                {
                    "sample_id": sample_id,
                    "family": old.get("family", "unknown"),
                    "label": label,
                    "transition": transition,
                    "before_prediction": int(old["prediction"]),
                    "after_prediction": int(new["prediction"]),
                    "before_score": float(old["score"]),
                    "after_score": float(new["score"]),
                }
            )
    result = {
        "before_name": args.before_name,
        "after_name": args.after_name,
        "sample_count": len(before),
        "transitions": dict(transitions),
        "by_label": {
            label: dict(counts) for label, counts in label_transitions.items()
        },
        "changed_examples": examples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items()
                      if key != "changed_examples"}, indent=2))


if __name__ == "__main__":
    main()
