#!/usr/bin/env python3
"""Build exact-sequence-deduplicated, group-disjoint compact datasets.

The input must use the compact JSONL.GZ schema:

    sample_id, source, label, family, api_seq, split

Quo Vadis records are grouped by report identity; Zenodo windows are grouped
by the source SHA encoded before the first colon in ``sample_id``. Exact API
sequences associated with conflicting binary labels are removed as complete
groups. Remaining duplicate sequences are represented once before assigning
source groups atomically to train, validation, and test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input compact JSONL.GZ.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--dataset",
        choices=("auto", "quo_vadis", "zenodo_11079764"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(row: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for call in row["api_seq"]:
        encoded = str(call).encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def detect_dataset(path: Path, rows: list[dict[str, Any]]) -> str:
    text = str(path).lower()
    if "zenodo" in text:
        return "zenodo_11079764"
    if "quo" in text or "vadis" in text:
        return "quo_vadis"
    if rows and ":" in str(rows[0].get("sample_id", "")):
        return "zenodo_11079764"
    return "quo_vadis"


def group_key(row: dict[str, Any], dataset: str) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if dataset == "zenodo_11079764":
        return sample_id.split(":", 1)[0].lower()

    match = re.search(r"(?i)([0-9a-f]{64})", sample_id)
    if match:
        return match.group(1).lower()
    for suffix in (".dat.json", ".json", ".dat"):
        if sample_id.lower().endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
            break
    return sample_id.lower()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            missing = {
                key
                for key in ("sample_id", "label", "family", "api_seq", "split")
                if key not in row
            }
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing required fields: {sorted(missing)}"
                )
            if row["split"] not in SPLITS:
                raise ValueError(
                    f"{path}:{line_number} has invalid split {row['split']!r}"
                )
            if row["label"] not in {"benign", "malware"}:
                raise ValueError(
                    f"{path}:{line_number} has invalid label {row['label']!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_source_stats(path: Path) -> dict[str, Any] | None:
    candidates = [
        path.with_suffix(path.suffix + ".stats.json"),
        Path(str(path) + ".stats.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def overlap_audit(
    rows: list[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    by_split = {split: [] for split in SPLITS}
    for row in rows:
        by_split[row["split"]].append(row)

    seq_sets = {
        split: {sequence_hash(row) for row in split_rows}
        for split, split_rows in by_split.items()
    }
    group_sets = {
        split: {group_key(row, dataset) for row in split_rows}
        for split, split_rows in by_split.items()
    }
    pairwise = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pairwise[f"{left}_{right}"] = {
            "exact_sequence_groups": len(seq_sets[left] & seq_sets[right]),
            "source_groups": len(group_sets[left] & group_sets[right]),
        }

    train_sequences = seq_sets["train"]
    train_groups = group_sets["train"]
    test_rows = by_split["test"]
    return {
        "pairwise_overlap": pairwise,
        "test_rows_with_train_sequence": sum(
            sequence_hash(row) in train_sequences for row in test_rows
        ),
        "test_rows_with_train_group": sum(
            group_key(row, dataset) in train_groups for row in test_rows
        ),
    }


def deduplicate(
    rows: list[dict[str, Any]], dataset: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequence_groups[sequence_hash(row)].append(row)

    duplicate_groups = {
        key: values for key, values in sequence_groups.items() if len(values) > 1
    }
    conflict_groups = {
        key: values
        for key, values in sequence_groups.items()
        if len({row["label"] for row in values}) > 1
    }

    retained = []
    duplicate_rows_removed = 0
    conflict_rows_removed = 0
    for seq_hash in sorted(sequence_groups):
        group = sequence_groups[seq_hash]
        if seq_hash in conflict_groups:
            conflict_rows_removed += len(group)
            continue
        representative = min(
            group,
            key=lambda row: (
                group_key(row, dataset),
                str(row.get("sample_id", "")),
                str(row.get("family", "")),
            ),
        )
        retained.append(dict(representative))
        duplicate_rows_removed += len(group) - 1

    return retained, {
        "exact_sequence_groups": len(sequence_groups),
        "duplicate_sequence_groups": len(duplicate_groups),
        "duplicate_rows_beyond_first": sum(
            len(values) - 1 for values in duplicate_groups.values()
        ),
        "conflicting_label_sequence_groups": len(conflict_groups),
        "conflicting_label_rows_removed": conflict_rows_removed,
        "duplicate_rows_removed_after_conflicts": duplicate_rows_removed,
        "rows_retained_after_deduplication": len(retained),
    }


def assign_group_disjoint_splits(
    rows: list[dict[str, Any]],
    dataset: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be positive and sum to 1")

    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_groups[group_key(row, dataset)].append(row)

    mixed_label_groups = {
        key: values
        for key, values in source_groups.items()
        if len({row["label"] for row in values}) > 1
    }
    if mixed_label_groups:
        examples = sorted(mixed_label_groups)[:5]
        raise ValueError(
            "Source groups contain mixed binary labels after sequence "
            f"deduplication; examples: {examples}"
        )

    grouped_by_label: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
        "benign": [],
        "malware": [],
    }
    for key, values in source_groups.items():
        grouped_by_label[values[0]["label"]].append((key, values))

    assignments: dict[str, str] = {}
    targets_by_label = {}
    counts_by_label = {}
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    for label, groups in grouped_by_label.items():
        total_rows = sum(len(values) for _, values in groups)
        targets = {split: ratios[split] * total_rows for split in SPLITS}
        targets_by_label[label] = targets
        counts = {split: 0 for split in SPLITS}
        counts_by_label[label] = counts

        shuffled = list(groups)
        random.Random(f"{seed}:{label}").shuffle(shuffled)
        shuffled.sort(key=lambda item: len(item[1]), reverse=True)
        for key, values in shuffled:
            candidates = []
            for split in SPLITS:
                target = targets[split]
                projected = counts[split] + len(values)
                normalized_overflow = (projected - target) / max(target, 1.0)
                fill_ratio = counts[split] / max(target, 1.0)
                candidates.append((normalized_overflow, fill_ratio, SPLITS.index(split), split))
            chosen = min(candidates)[-1]
            assignments[key] = chosen
            counts[chosen] += len(values)

    output = []
    for row in rows:
        updated = dict(row)
        updated["split"] = assignments[group_key(row, dataset)]
        output.append(updated)
    random.Random(seed).shuffle(output)

    return output, {
        "unique_source_groups": len(source_groups),
        "target_rows_by_label": targets_by_label,
        "assigned_rows_by_label": counts_by_label,
        "split_policy": (
            "binary-label-balanced greedy assignment of atomic report groups"
            if dataset == "quo_vadis"
            else "binary-label-balanced greedy assignment of atomic SHA groups"
        ),
    }


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(row["api_seq"]) for row in rows]
    malware_families = Counter(
        str(row.get("family", "unknown"))
        for row in rows
        if row["label"] == "malware"
    )
    malware_total = sum(malware_families.values())
    return {
        "total": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "family_counts": dict(Counter(str(row.get("family", "unknown")) for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_label_counts": {
            f"{split}:{label}": count
            for (split, label), count in sorted(
                Counter((row["split"], row["label"]) for row in rows).items()
            )
        },
        "avg_seq_len": statistics.mean(lengths),
        "median_seq_len": statistics.median(lengths),
        "min_seq_len": min(lengths),
        "max_seq_len": max(lengths),
        "malware_family_count": len(malware_families),
        "largest_malware_family": (
            malware_families.most_common(1)[0][0] if malware_families else None
        ),
        "largest_malware_family_count": (
            malware_families.most_common(1)[0][1] if malware_families else 0
        ),
        "largest_malware_family_share": (
            malware_families.most_common(1)[0][1] / malware_total
            if malware_total
            else None
        ),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
                    handle.write("\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    source_stats = load_source_stats(input_path)
    dataset = (
        detect_dataset(input_path, rows)
        if args.dataset == "auto"
        else args.dataset
    )
    preliminary_overlap = overlap_audit(rows, dataset)
    deduplicated, dedup_stats = deduplicate(rows, dataset)
    final_rows, split_stats = assign_group_disjoint_splits(
        deduplicated,
        dataset,
        args.seed,
        args.train_ratio,
        args.val_ratio,
    )
    final_overlap = overlap_audit(final_rows, dataset)

    overlap_values = [
        value
        for pair in final_overlap["pairwise_overlap"].values()
        for value in pair.values()
    ]
    overlap_values.extend(
        [
            final_overlap["test_rows_with_train_sequence"],
            final_overlap["test_rows_with_train_group"],
        ]
    )
    if any(overlap_values):
        raise RuntimeError(f"Final overlap invariant failed: {final_overlap}")

    output_path = out_dir / f"{dataset}_leakage_safe.jsonl.gz"
    write_jsonl_gz(output_path, final_rows)

    input_checksum = sha256_file(input_path)
    output_checksum = sha256_file(output_path)
    dataset_quality = {
        "dataset": dataset,
        "input": str(input_path),
        "output": str(output_path),
        "source_stats": source_stats,
        "preliminary": distribution(rows),
        "confirmatory": distribution(final_rows),
        "removed_total": len(rows) - len(final_rows),
        "deduplication": dedup_stats,
    }
    leakage_audit = {
        "dataset": dataset,
        "sequence_hash": "sha256 over length-prefixed UTF-8 API names",
        "grouping_key": (
            "normalized Quo Vadis report identity/SHA"
            if dataset == "quo_vadis"
            else "source SHA before the first colon in sample_id"
        ),
        "preliminary": preliminary_overlap,
        "confirmatory": final_overlap,
        "invariants": {
            "exact_sequence_overlap_is_zero": True,
            "source_group_overlap_is_zero": True,
        },
    }
    manifest = {
        "dataset": dataset,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "input": str(input_path),
        "input_sha256": input_checksum,
        "output": str(output_path),
        "output_sha256": output_checksum,
        "split_assignment": split_stats,
        "artifacts": {
            "dataset_quality": str(out_dir / "dataset_quality.json"),
            "leakage_audit": str(out_dir / "leakage_audit.json"),
        },
    }
    write_json(out_dir / "dataset_quality.json", dataset_quality)
    write_json(out_dir / "leakage_audit.json", leakage_audit)
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
