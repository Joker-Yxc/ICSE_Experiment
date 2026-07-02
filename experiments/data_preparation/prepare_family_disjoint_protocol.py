#!/usr/bin/env python3
"""Create held-out-family folds from a leakage-safe compact dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--dataset",
        choices=("quo_vadis", "zenodo_11079764"),
        required=True,
    )
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Select the top N eligible malware families by distinct source groups.",
    )
    parser.add_argument(
        "--min-groups",
        type=int,
        default=20,
        help="Minimum distinct source groups required for an eligible family.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def source_group(row: dict, dataset: str) -> str:
    sample_id = str(row["sample_id"])
    if dataset == "zenodo_11079764":
        return sample_id.split(":", 1)[0].lower()
    for suffix in (".dat.json", ".json", ".dat"):
        if sample_id.lower().endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
            break
    return sample_id.lower()


def sequence_hash(row: dict) -> str:
    digest = hashlib.sha256()
    for call in row["api_seq"]:
        encoded = str(call).encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True))
                    handle.write("\n")


def validate_source_dataset(rows: list[dict], dataset: str) -> None:
    split_groups = defaultdict(set)
    split_sequences = defaultdict(set)
    for row in rows:
        split_groups[row["split"]].add(source_group(row, dataset))
        split_sequences[row["split"]].add(sequence_hash(row))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_groups[left] & split_groups[right]:
            raise ValueError(f"Input is not group-disjoint: {left}/{right}")
        if split_sequences[left] & split_sequences[right]:
            raise ValueError(f"Input is not exact-sequence-disjoint: {left}/{right}")


def available_families(rows: list[dict], dataset: str) -> list[dict]:
    family_groups = defaultdict(set)
    family_rows = Counter()
    for row in rows:
        if row["label"] != "malware":
            continue
        family = str(row.get("family", "unknown"))
        family_groups[family].add(source_group(row, dataset))
        family_rows[family] += 1
    return [
        {
            "family": family,
            "distinct_source_groups": len(family_groups[family]),
            "rows": family_rows[family],
        }
        for family in sorted(
            family_groups,
            key=lambda value: (-len(family_groups[value]), value),
        )
    ]


def fold_rows(rows: list[dict], family: str) -> list[dict]:
    output = []
    for row in rows:
        if row["split"] in {"train", "val"}:
            if row["label"] == "malware" and row.get("family") == family:
                continue
            output.append(row)
        elif row["label"] == "benign" or row.get("family") == family:
            output.append(row)
    return output


def audit_fold(rows: list[dict], family: str, dataset: str) -> dict:
    train_val = [
        row for row in rows if row["split"] in {"train", "val"}
    ]
    held_out_in_development = sum(
        row["label"] == "malware" and row.get("family") == family
        for row in train_val
    )
    test = [row for row in rows if row["split"] == "test"]
    unexpected_test_malware = sum(
        row["label"] == "malware" and row.get("family") != family
        for row in test
    )
    development_groups = {source_group(row, dataset) for row in train_val}
    development_sequences = {sequence_hash(row) for row in train_val}
    return {
        "held_out_family": family,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_label_counts": {
            f"{split}:{label}": count
            for (split, label), count in sorted(
                Counter((row["split"], row["label"]) for row in rows).items()
            )
        },
        "held_out_test_malware": sum(
            row["split"] == "test"
            and row["label"] == "malware"
            and row.get("family") == family
            for row in rows
        ),
        "benign_test": sum(
            row["split"] == "test" and row["label"] == "benign" for row in rows
        ),
        "held_out_in_development": held_out_in_development,
        "unexpected_test_malware": unexpected_test_malware,
        "test_rows_with_development_group": sum(
            source_group(row, dataset) in development_groups for row in test
        ),
        "test_rows_with_development_sequence": sum(
            sequence_hash(row) in development_sequences for row in test
        ),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(source)
    validate_source_dataset(rows, args.dataset)

    inventory = available_families(rows, args.dataset)
    eligible = [
        item for item in inventory if item["distinct_source_groups"] >= args.min_groups
    ]
    if args.family:
        selected = args.family
    elif args.top_n:
        selected = [item["family"] for item in eligible[: args.top_n]]
    else:
        selected = [item["family"] for item in eligible]
    known = {item["family"] for item in inventory}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")
    if not selected:
        raise ValueError("No held-out families selected")

    manifest = {
        "dataset": args.dataset,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "selection": {
            "min_distinct_source_groups": args.min_groups,
            "top_n": args.top_n,
            "explicit_families": args.family,
            "selected_families": selected,
        },
        "family_inventory": inventory,
        "folds": {},
    }
    for family in selected:
        selected_rows = fold_rows(rows, family)
        audit = audit_fold(selected_rows, family, args.dataset)
        required_zero = (
            audit["held_out_in_development"],
            audit["unexpected_test_malware"],
            audit["test_rows_with_development_group"],
            audit["test_rows_with_development_sequence"],
        )
        if any(required_zero) or not audit["held_out_test_malware"]:
            raise RuntimeError(f"Fold invariant failed for {family}: {audit}")
        output = out_dir / f"{args.dataset}_heldout_{family}.jsonl.gz"
        write_rows(output, selected_rows)
        audit["output"] = str(output)
        audit["output_sha256"] = sha256_file(output)
        manifest["folds"][family] = audit
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(out_dir / "manifest.json")


if __name__ == "__main__":
    main()
