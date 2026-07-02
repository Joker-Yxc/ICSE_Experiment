#!/usr/bin/env python3
"""Compare two prediction files with group-level paired resampling."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dataset", choices=("quo_vadis", "zenodo_11079764"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in map(json.loads, handle)}


def group_id(sample_id: str, dataset: str) -> str:
    if dataset == "zenodo_11079764":
        return sample_id.split(":", 1)[0]
    for suffix in (".dat.json", ".json", ".dat"):
        if sample_id.lower().endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def paired_arrays(full: dict[str, dict], baseline: dict[str, dict]):
    ids = sorted(set(full) & set(baseline))
    if set(full) != set(baseline):
        raise ValueError("Prediction files do not contain identical sample IDs")
    labels = np.asarray([int(full[key]["label"]) for key in ids])
    if any(int(full[key]["label"]) != int(baseline[key]["label"]) for key in ids):
        raise ValueError("Prediction files disagree on labels")
    full_pred = np.asarray([int(full[key]["prediction"]) for key in ids])
    baseline_pred = np.asarray([int(baseline[key]["prediction"]) for key in ids])
    return ids, labels, full_pred, baseline_pred


def score(labels, pred) -> float:
    return float(f1_score(labels, pred, zero_division=0))


def main() -> None:
    args = parse_args()
    full = load(Path(args.full))
    baseline = load(Path(args.baseline))
    ids, labels, full_pred, baseline_pred = paired_arrays(full, baseline)
    groups = defaultdict(list)
    for index, sample_id in enumerate(ids):
        groups[group_id(sample_id, args.dataset)].append(index)
    group_indices = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    observed = score(labels, full_pred) - score(labels, baseline_pred)
    rng = np.random.default_rng(args.seed)

    boot = []
    for _ in range(args.iterations):
        sampled = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[index] for index in sampled])
        boot.append(score(labels[indices], full_pred[indices]) - score(labels[indices], baseline_pred[indices]))
    ci = np.quantile(boot, [0.025, 0.975]).tolist()

    extreme = 0
    for _ in range(args.iterations):
        perm_full = full_pred.copy()
        perm_baseline = baseline_pred.copy()
        for indices in group_indices:
            if rng.random() < 0.5:
                perm_full[indices], perm_baseline[indices] = (
                    perm_baseline[indices].copy(),
                    perm_full[indices].copy(),
                )
        delta = score(labels, perm_full) - score(labels, perm_baseline)
        extreme += abs(delta) >= abs(observed)
    p_value = (extreme + 1) / (args.iterations + 1)
    result = {
        "dataset": args.dataset,
        "samples": len(ids),
        "groups": len(group_indices),
        "full_f1": score(labels, full_pred),
        "baseline_f1": score(labels, baseline_pred),
        "delta_f1": observed,
        "bootstrap_95_ci": ci,
        "paired_randomization_p_value": p_value,
        "iterations": args.iterations,
        "seed": args.seed,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
