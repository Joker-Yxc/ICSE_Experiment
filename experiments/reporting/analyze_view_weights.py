#!/usr/bin/env python3
"""Analyze graph-view weights conditioned on the interleaving indicator."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--dataset", choices=("quo_vadis", "zenodo_11079764"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def group_id(sample_id: str, dataset: str) -> str:
    if dataset == "zenodo_11079764":
        return sample_id.split(":", 1)[0]
    for suffix in (".dat.json", ".json", ".dat"):
        if sample_id.lower().endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def main() -> None:
    args = parse_args()
    values = {0: [], 1: []}
    by_group = defaultdict(lambda: {0: [], 1: []})
    for pattern in args.inputs:
        paths = (
            [Path(value) for value in sorted(glob.glob(pattern))]
            if any(char in pattern for char in "*?[")
            else [Path(pattern)]
        )
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    indicator = int(row["c_ij"])
                    weight = float(row["w_graph"])
                    values[indicator].append(weight)
                    by_group[group_id(row["sample_id"], args.dataset)][indicator].append(weight)
    if not values[0] or not values[1]:
        raise ValueError("Both C_ij groups must contain observations")

    group_differences = np.asarray(
        [
            np.median(bucket[1]) - np.median(bucket[0])
            for bucket in by_group.values()
            if bucket[0] and bucket[1]
        ],
        dtype=np.float64,
    )
    if not group_differences.size:
        raise ValueError("No source group contains both C_ij values")
    rng = np.random.default_rng(args.seed)
    boot = np.asarray(
        [
            np.median(rng.choice(group_differences, size=len(group_differences), replace=True))
            for _ in range(args.iterations)
        ]
    )
    observed = float(np.median(group_differences))
    observed_mean = float(np.mean(group_differences))
    permuted = np.asarray(
        [
            np.mean(group_differences * rng.choice([-1.0, 1.0], size=len(group_differences)))
            for _ in range(args.iterations)
        ]
    )
    result = {
        "dataset": args.dataset,
        "c0": summarize(values[0]),
        "c1": summarize(values[1]),
        "groups_with_both": int(group_differences.size),
        "group_median_difference": observed,
        "group_mean_difference_for_test": observed_mean,
        "bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "paired_sign_permutation_p_value": float(
            (np.sum(np.abs(permuted) >= abs(observed_mean)) + 1)
            / (args.iterations + 1)
        ),
        "matched_pairs_rank_biserial": float(
            np.mean(group_differences > 0) - np.mean(group_differences < 0)
        ),
        "iterations": args.iterations,
        "seed": args.seed,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.plot:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(4.6, 3.2))
        axis.violinplot([values[0], values[1]], showmedians=True, showextrema=False)
        axis.set_xticks([1, 2], [r"$C_{ij}=0$", r"$C_{ij}=1$"])
        axis.set_ylabel(r"Graph-view weight $w^g_{ij}$")
        axis.set_title(args.dataset.replace("_", " "))
        figure.tight_layout()
        figure.savefig(args.plot, dpi=200)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
