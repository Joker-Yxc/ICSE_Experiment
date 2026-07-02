#!/usr/bin/env python3
"""Stream Zenodo API traces into the compact 50k experiment format."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import statistics
import tarfile
from collections import Counter
from pathlib import Path


SEED = 7
WINDOW = 200
STRIDE = 100
TARGET_PER_LABEL = 25_000


def clean_call(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "*" or len(text) > 160:
        return None
    return re.sub(r"\s+", "_", text)


def segment(trace, window=WINDOW, stride=STRIDE):
    if len(trace) < 2:
        return []
    if len(trace) <= window:
        return [trace]
    return [trace[i : i + window] for i in range(0, len(trace) - window + 1, stride)]


def extract_calls(fileobj):
    calls = []
    for raw in fileobj:
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if event.get("msg") and event.get("msg") != "Monitored function called":
            continue
        call = clean_call(event.get("vmi_FunctionName") or event.get("vmi_Function"))
        if call:
            calls.append(call)
    return calls


def load_labels(metadata_path):
    by_family = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    labels = {}
    families = {}
    for family, shas in by_family.items():
        label = "benign" if family == "benign" else "malware"
        for sha in shas:
            if sha not in labels or labels[sha] != "benign":
                labels[sha] = label
                families[sha] = family if label == "malware" else "benign"
    return labels, families, by_family


def split_rows(rows):
    groups = {"benign": [], "malware": []}
    for row in rows:
        groups[row["label"]].append(row)
    out = []
    rng = random.Random(SEED)
    for label in sorted(groups):
        items = list(groups[label])
        rng.shuffle(items)
        n_train = int(len(items) * 0.70)
        n_val = int(len(items) * 0.15)
        for split, part in (
            ("train", items[:n_train]),
            ("val", items[n_train : n_train + n_val]),
            ("test", items[n_train + n_val :]),
        ):
            for row in part:
                row = dict(row)
                row["split"] = split
                out.append(row)
    rng.shuffle(out)
    return out


def write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def stats(rows, metadata_summary, files_used):
    lengths = [len(row["api_seq"]) for row in rows]
    return {
        "total": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "family_counts": dict(Counter(row["family"] for row in rows).most_common()),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "avg_seq_len": round(statistics.mean(lengths), 3),
        "max_seq_len": max(lengths),
        "min_seq_len": min(lengths),
        "dataset": "zenodo_11079764_api_traces",
        "subset": "main_50k",
        "requested": {"benign": TARGET_PER_LABEL, "malware": TARGET_PER_LABEL},
        "seed": SEED,
        "split_policy": "stratified by binary label, 70/15/15",
        "window": WINDOW,
        "stride": STRIDE,
        "files_used": files_used,
        "metadata_summary": metadata_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="API_traces_malware_detection.tar.xz")
    parser.add_argument(
        "--metadata",
        default="datasets_50k/zenodo_11079764/data/raw/shas_by_families.json",
    )
    parser.add_argument("--output", default="datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz")
    args = parser.parse_args()

    labels, families, by_family = load_labels(args.metadata)
    metadata_summary = {
        "benign_files": len(by_family.get("benign", [])),
        "malware_files": sum(len(v) for k, v in by_family.items() if k != "benign"),
        "families": len(by_family),
    }
    rows = []
    counts = Counter()
    files_used = Counter()
    candidate_shas = set(labels)

    with tarfile.open(args.archive, "r:xz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            sha = Path(member.name).stem
            if sha not in candidate_shas:
                continue
            label = labels[sha]
            if counts[label] >= TARGET_PER_LABEL:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            calls = extract_calls(f)
            parts = segment(calls)
            if not parts:
                continue
            files_used[label] += 1
            for seg_id, seq in enumerate(parts):
                if counts[label] >= TARGET_PER_LABEL:
                    break
                rows.append(
                    {
                        "sample_id": f"{sha}:{seg_id}",
                        "source": "zenodo_11079764",
                        "label": label,
                        "family": families.get(sha, label),
                        "api_seq": seq,
                    }
                )
                counts[label] += 1
            if counts["benign"] >= TARGET_PER_LABEL and counts["malware"] >= TARGET_PER_LABEL:
                break

    if counts["benign"] < TARGET_PER_LABEL or counts["malware"] < TARGET_PER_LABEL:
        raise RuntimeError(f"shortfall: {dict(counts)}")

    rows = split_rows(rows)
    output = Path(args.output)
    write_jsonl_gz(output, rows)
    summary = stats(rows, metadata_summary, dict(files_used))
    summary["output"] = str(output)
    output.with_suffix(output.suffix + ".stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
