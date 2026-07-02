#!/usr/bin/env python3
"""Create fixed leave-one-family-out compact datasets for all baselines."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets_50k/quo_vadis/unknown_family_protocol",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Generate only selected families; repeat as needed. Default: all malware families.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def select_rows(rows: list[dict], family: str) -> list[dict]:
    selected = []
    for row in rows:
        if row["split"] in {"train", "val"}:
            if row["label"] == "malware" and row.get("family") == family:
                continue
            selected.append(row)
        elif row["split"] == "test":
            if row["label"] == "benign" or row.get("family") == family:
                selected.append(row)
    return selected


def summarize(rows: list[dict], held_out_family: str) -> dict:
    split_counts = Counter(row["split"] for row in rows)
    split_label_counts = Counter((row["split"], row["label"]) for row in rows)
    held_out_counts = Counter(
        row["split"] for row in rows if row.get("family") == held_out_family
    )
    return {
        "held_out_family": held_out_family,
        "protocol": (
            "held-out family removed from train and validation; test contains "
            "original benign test samples plus original held-out-family test samples"
        ),
        "split_counts": dict(split_counts),
        "split_label_counts": {
            f"{split}:{label}": count
            for (split, label), count in sorted(split_label_counts.items())
        },
        "held_out_family_counts": dict(held_out_counts),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.data)
    rows = load_rows(source)
    available = sorted(
        {
            str(row.get("family", "unknown"))
            for row in rows
            if row["label"] == "malware"
        }
    )
    families = args.family or available
    unknown = sorted(set(families) - set(available))
    if unknown:
        raise ValueError(f"Families not present in the dataset: {unknown}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source),
        "source_rows": len(rows),
        "families": {},
    }
    for family in families:
        selected = select_rows(rows, family)
        output = out_dir / f"quo_vadis_heldout_{family}.jsonl.gz"
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        details = summarize(selected, family)
        details["output"] = str(output)
        manifest["families"][family] = details
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
