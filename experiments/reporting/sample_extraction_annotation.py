#!/usr/bin/env python3
"""Sample semantic elements and create a two-annotator CSV template."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from escapture.llm_behavior_extractor import FrozenTemplateBehaviorExtractor


FIELDS = ("subject", "operation", "object", "resource", "context", "goal", "template_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-per-record", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FrozenTemplateBehaviorExtractor()
    strata = defaultdict(list)
    with gzip.open(args.data, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            elements = extractor.extract_sequence(
                row["api_seq"], sample_id=str(row["sample_id"])
            )
            for element in elements[: args.max_per_record]:
                strata[(row["label"], element.resource, element.template_id)].append(
                    (row, element)
                )
    rng = random.Random(args.seed)
    keys = list(strata)
    rng.shuffle(keys)
    selected = []
    cursors = defaultdict(int)
    while len(selected) < args.sample_size:
        progressed = False
        for key in keys:
            candidates = strata[key]
            index = cursors[key]
            if index >= len(candidates):
                continue
            if index == 0:
                rng.shuffle(candidates)
            selected.append(candidates[index])
            cursors[key] += 1
            progressed = True
            if len(selected) >= args.sample_size:
                break
        if not progressed:
            break

    columns = [
        "sample_id", "family", "binary_label", "api_index", "api_name",
        *[f"pred_{field}" for field in FIELDS],
        *[f"ann1_{field}" for field in FIELDS],
        *[f"ann2_{field}" for field in FIELDS],
        *[f"gold_{field}" for field in FIELDS],
        *[f"supported_{field}" for field in FIELDS],
        "ann1_id", "ann2_id", "adjudicator_id", "notes",
    ]
    csv_path = out_dir / "semantic_extraction_annotation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row, element in selected:
            record = {
                "sample_id": row["sample_id"],
                "family": row.get("family", "unknown"),
                "binary_label": row["label"],
                "api_index": element.index,
                "api_name": element.api_name,
            }
            for field in FIELDS:
                record[f"pred_{field}"] = getattr(element, field)
            writer.writerow(record)

    codebook = """# Semantic Extraction Annotation Codebook

Annotators must not inspect detector predictions or malware scores.

For each field, record the normalized value and mark `supported_<field>` as
`yes`, `no`, or `underspecified`. A prediction is not correct merely because it
is plausible; it must be supported by the API name and available trace context.

- subject: process/module identity supported by the trace
- operation: normalized action performed by the API
- object: normalized target object
- resource: file, registry, network, process, memory, permission, IPC, or system
- context: execution condition directly supported by available input
- goal: local behavioral goal; do not infer malware intent from the label
- template_id: best frozen-template category

Annotators work independently. Cohen's kappa is computed before adjudication.
"""
    (out_dir / "annotation_codebook.md").write_text(codebook, encoding="utf-8")
    manifest = {
        "data": args.data,
        "sample_size_requested": args.sample_size,
        "sample_size_written": len(selected),
        "seed": args.seed,
        "strata": "binary label x resource x template_id",
        "csv": str(csv_path),
        "codebook": str(out_dir / "annotation_codebook.md"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
