#!/usr/bin/env python3
"""Build compact Windows API malware experiment subsets.

The script intentionally avoids materializing full raw datasets. It writes
selected samples only, with one compact JSON object per line in gzip files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import statistics
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SEED = 7
MAX_API_SEQ_LEN = 5000
BENIGN_FAMILIES = {"benign", "clean", "normal", "goodware", "windows_syswow64"}
SUBSETS = {
    "pilot_1k": (500, 500),
    "main_10k": (5000, 5000),
    "extended_20k": (10000, 10000),
    "main_50k": (25000, 25000),
}


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    source: str
    label: str
    family: str
    path: Path


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:32]


def clean_api_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "*" or len(text) > 160:
        return None
    return " ".join(text.split())


def extract_api_seq(obj: Any, limit: int = MAX_API_SEQ_LEN) -> list[str]:
    """Extract API names in encounter order from nested Speakeasy-like JSON."""
    seq: list[str] = []

    def walk(value: Any) -> None:
        if len(seq) >= limit:
            return
        if isinstance(value, dict):
            api_name = value.get("api_name")
            if api_name is not None:
                cleaned = clean_api_name(api_name)
                if cleaned:
                    seq.append(cleaned)
                    if len(seq) >= limit:
                        return
            apis = value.get("apis")
            if isinstance(apis, list):
                for item in apis:
                    walk(item)
                    if len(seq) >= limit:
                        return
            for key, child in value.items():
                if key in {"api_name", "apis", "args", "ret_val", "pc", "error", "regs", "stack"}:
                    continue
                if isinstance(child, (dict, list)):
                    walk(child)
                    if len(seq) >= limit:
                        return
        elif isinstance(value, list):
            for item in value:
                walk(item)
                if len(seq) >= limit:
                    return

    walk(obj)
    return seq[:limit]


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def split_items(items: list[Any], seed: int = SEED) -> dict[str, list[Any]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def stratified_split(items: list[Any], key_fn, seed: int = SEED) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        groups[str(key_fn(item))].append(item)
    out = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        parts = split_items(groups[key], seed)
        for split, rows in parts.items():
            out[split].extend(rows)
    for split in out:
        random.Random(seed).shuffle(out[split])
    return out


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
            count += 1
    return count


def stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(row["api_seq"]) for row in rows]
    family_counts = Counter(row["family"] for row in rows)
    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    return {
        "total": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "family_counts": dict(sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "split_counts": dict(sorted(split_counts.items())),
        "avg_seq_len": round(statistics.mean(lengths), 3) if lengths else 0,
        "max_seq_len": max(lengths) if lengths else 0,
        "min_seq_len": min(lengths) if lengths else 0,
    }


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_quo_vadis(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("example-"):
            continue
        family = "unknown"
        for part in reversed(path.parts):
            if part.startswith("report_"):
                family = part.removeprefix("report_")
                break
        label = "benign" if family in {"clean", "windows_syswow64"} else "malware"
        normalized_family = "benign" if label == "benign" else family
        sample_id = path.stem
        candidates.append(Candidate(sample_id, "quo_vadis", label, normalized_family, path))
    return candidates


def materialize_candidate(candidate: Candidate) -> dict[str, Any] | None:
    try:
        seq = extract_api_seq(load_json_file(candidate.path))
    except Exception as exc:
        print(f"[warn] failed to parse {candidate.path}: {exc}")
        return None
    if len(seq) < 2:
        return None
    return {
        "sample_id": candidate.sample_id,
        "source": candidate.source,
        "label": candidate.label,
        "family": candidate.family,
        "api_seq": seq,
    }


def select_valid_rows(candidates: list[Candidate], target: int, seed: int = SEED) -> tuple[list[dict[str, Any]], int]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    rows: list[dict[str, Any]] = []
    invalid = 0
    for candidate in shuffled:
        row = materialize_candidate(candidate)
        if row is None:
            invalid += 1
            continue
        rows.append(row)
        if len(rows) >= target:
            break
    return rows, invalid


def select_balanced_binary(candidates: list[Candidate], benign_n: int, malware_n: int, seed: int = SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benign = [c for c in candidates if c.label == "benign"]
    malware = [c for c in candidates if c.label == "malware"]
    benign_rows, invalid_benign = select_valid_rows(benign, benign_n, seed)
    malware_rows, invalid_malware = select_valid_rows(malware, malware_n, seed)
    notes = {
        "candidate_counts": {"benign": len(benign), "malware": len(malware)},
        "invalid_or_too_short_skipped": {"benign": invalid_benign, "malware": invalid_malware},
        "shortfall": {
            "benign": max(benign_n - len(benign_rows), 0),
            "malware": max(malware_n - len(malware_rows), 0),
        },
    }
    return benign_rows + malware_rows, notes


def build_binary_subset(name: str, candidates: list[Candidate], output_dir: Path) -> dict[str, Any]:
    benign_n, malware_n = SUBSETS[name]
    rows, selection_notes = select_balanced_binary(candidates, benign_n, malware_n)
    split = stratified_split(rows, key_fn=lambda row: row["label"])
    rows = []
    for split_name, split_rows in split.items():
        for row in split_rows:
            row = dict(row)
            row["split"] = split_name
            rows.append(row)
    random.Random(SEED).shuffle(rows)
    out_path = output_dir / f"quo_vadis_{name}.jsonl.gz"
    write_jsonl_gz(out_path, rows)
    stats = stats_for_rows(rows)
    stats.update(
        {
            "dataset": "quo_vadis_malware_emulation",
            "subset": name,
            "output": str(out_path),
            "requested": {"benign": benign_n, "malware": malware_n},
            "selection_notes": selection_notes,
            "seed": SEED,
            "split_policy": "stratified by binary label, 70/15/15",
            "max_api_seq_len": MAX_API_SEQ_LEN,
        }
    )
    write_stats(out_path.with_suffix(out_path.suffix + ".stats.json"), stats)
    return stats


def build_family_subset(candidates: list[Candidate], output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    malware_by_family: dict[str, list[Candidate]] = defaultdict(list)
    benign = [c for c in candidates if c.label == "benign"]
    for candidate in candidates:
        if candidate.label == "malware":
            malware_by_family[candidate.family].append(candidate)
    top5 = [family for family, rows in sorted(malware_by_family.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]]
    selected_rows: list[dict[str, Any]] = []
    invalid_by_family: dict[str, int] = {}
    for family in top5:
        rows = list(malware_by_family[family])
        valid_rows, invalid = select_valid_rows(rows, 500, seed)
        selected_rows.extend(valid_rows)
        invalid_by_family[family] = invalid
    benign_rows = list(benign)
    benign_target = 1000 if len(benign_rows) >= 1000 else min(500, len(benign_rows))
    valid_benign_rows, invalid_benign = select_valid_rows(benign_rows, benign_target, seed)
    selected_rows.extend(valid_benign_rows)
    split = stratified_split(selected_rows, key_fn=lambda row: row["family"])
    rows = []
    for split_name, split_rows in split.items():
        for row in split_rows:
            row = dict(row)
            row["split"] = split_name
            rows.append(row)
    random.Random(seed).shuffle(rows)
    out_path = output_dir / "quo_vadis_family_top5.jsonl.gz"
    write_jsonl_gz(out_path, rows)
    stats = stats_for_rows(rows)
    stats.update(
        {
            "dataset": "quo_vadis_malware_emulation",
            "subset": "family_top5",
            "output": str(out_path),
            "top5_malware_families": top5,
            "malware_per_family_target": 500,
            "benign_target": benign_target,
            "invalid_or_too_short_skipped": {**invalid_by_family, "benign": invalid_benign},
            "seed": seed,
            "split_policy": "stratified by family, 70/15/15",
            "max_api_seq_len": MAX_API_SEQ_LEN,
        }
    )
    write_stats(out_path.with_suffix(out_path.suffix + ".stats.json"), stats)
    return stats


def write_zenodo_manifest(zenodo_dir: Path, output_dir: Path) -> dict[str, Any]:
    record_path = zenodo_dir / "record_metadata.json"
    family_path = zenodo_dir / "shas_by_families.json"
    manifest: dict[str, Any] = {
        "dataset": "zenodo_11079764",
        "status": "pending_raw_trace_archive",
        "reason": "Only record_metadata.json and shas_by_families.json are present locally; API trace archive is not present.",
        "expected_local_archive": str(zenodo_dir / "API_traces_malware_detection.tar.xz"),
        "streaming_supported": False,
        "note": "The archive is not downloaded or extracted automatically. Add a dataset-specific streaming reader before generating Zenodo compact subsets.",
    }
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        manifest["record_id"] = record.get("id")
        manifest["doi"] = record.get("doi")
        manifest["files"] = record.get("files", [])
    if family_path.exists():
        families = json.loads(family_path.read_text(encoding="utf-8"))
        counts = sorted(((family, len(shas)) for family, shas in families.items()), key=lambda kv: (-kv[1], kv[0]))
        manifest["family_mapping_file"] = str(family_path)
        manifest["family_count"] = len(counts)
        manifest["sha_reference_count"] = sum(count for _, count in counts)
        manifest["top20_families"] = dict(counts[:20])
    out_path = output_dir / "zenodo_11079764_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def can_attempt_extended(output_dir: Path) -> bool:
    usage = __import__("shutil").disk_usage(output_dir.resolve().anchor or "/")
    return usage.free >= 20 * 1024**3


def write_readme(output_dir: Path, all_stats: list[dict[str, Any]], zenodo_manifest: dict[str, Any]) -> None:
    lines = [
        "# Windows API Malware Baseline Experiment Setting",
        "",
        "## Data Policy",
        "",
        "- Full raw datasets are not expanded or copied into experiment artifacts.",
        "- Compact subsets store only `sample_id`, `source`, `label`, `family`, `api_seq`, and `split`.",
        "- `api_seq` contains API names only and is truncated to at most 5000 calls.",
        "- All methods must consume these files directly so they share the exact same split.",
        "- Split policy: seed=7, stratified 70/15/15.",
        "",
        "## Zenodo 11079764",
        "",
        "Zenodo record metadata is present locally, but the API trace archive is not present in this workspace.",
        "The original archive is intentionally not downloaded or extracted automatically because storage is limited.",
        "The compact Quo Vadis subsets below are ready. Zenodo compact subsets are pending until the trace archive is available and a dataset-specific streaming reader is enabled; the manifest records the exact Zenodo file URL and checksum.",
        "",
        "## Quo Vadis Malware Emulation",
        "",
    ]
    for stats in all_stats:
        lines.extend(
            [
                f"### {stats['subset']}",
                "",
                f"- Output: `{stats['output']}`",
                f"- Total: {stats['total']}",
                f"- Labels: `{stats['label_counts']}`",
                f"- Splits: `{stats['split_counts']}`",
                f"- Average API sequence length: {stats['avg_seq_len']}",
                f"- Max API sequence length: {stats['max_seq_len']}",
                f"- Family distribution: `{stats['family_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Main Experimental Protocol",
            "",
            "Due to the potentially large expanded size of Zenodo 11079764, experiments use stratified balanced compact subsets.",
            "The primary experiment is the 10k balanced subset; pilot_1k is for smoke tests and development; extended_20k is generated only when storage allows.",
            "All baselines and the proposed method must use the same compact subset files and the embedded `split` field.",
            "",
            "## Files",
            "",
            "- `*.jsonl.gz`: compact samples.",
            "- `*.jsonl.gz.stats.json`: subset statistics.",
            "- `zenodo_11079764_manifest.json`: Zenodo file metadata and download pointer.",
        ]
    )
    (output_dir / "README_experiment_setting.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quo-vadis-root", default="datasets_50k/quo_vadis/data/raw")
    parser.add_argument("--zenodo-root", default="datasets_50k/zenodo_11079764/data/raw")
    parser.add_argument("--output-dir", default="datasets_50k/quo_vadis/data")
    parser.add_argument("--skip-extended", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: list[dict[str, Any]] = []
    qv_root = Path(args.quo_vadis_root)
    if qv_root.exists():
        candidates = discover_quo_vadis(qv_root)
        for subset_name in ["main_50k", "extended_20k", "main_10k"]:
            stats.append(build_binary_subset(subset_name, candidates, output_dir))
        stats.append(build_family_subset(candidates, output_dir))
    else:
        print(f"[warn] Quo Vadis root not found: {qv_root}")

    zenodo_manifest = write_zenodo_manifest(Path(args.zenodo_root), output_dir)
    write_readme(output_dir, stats, zenodo_manifest)
    summary = {"quo_vadis_subsets": stats, "zenodo": zenodo_manifest}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
