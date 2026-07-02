#!/usr/bin/env python3
"""Tune LogisticRegression for ours on compact API 1-3 gram features."""

import gzip
import json
import time

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


DATA = "datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz"
SEED = 7
rows = []
splits = {"train": [], "val": [], "test": []}
with gzip.open(DATA, "rt", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        row = json.loads(line)
        rows.append(row)
        splits[row["split"]].append(idx)


def text(i):
    return " ".join(rows[i]["api_seq"])


def y(indices):
    return np.array([1 if rows[i]["label"] == "malware" else 0 for i in indices])


def best_threshold(y_true, prob):
    yb = y_true.astype(bool)
    best = (0.0, 0.5)
    for th in np.linspace(0.02, 0.98, 193):
        pred = prob >= th
        tp = np.count_nonzero(pred & yb)
        fp = np.count_nonzero(pred & ~yb)
        fn = np.count_nonzero(~pred & yb)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        if f1 > best[0]:
            best = (float(f1), float(th))
    return best


def metrics(y_true, prob, th):
    pred = (prob >= th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, prob)),
    }


def main():
    started = time.time()
    vectorizer = HashingVectorizer(
        n_features=2**21,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
        ngram_range=(1, 3),
    )
    X = {split: vectorizer.transform(text(i) for i in indices) for split, indices in splits.items()}
    Y = {split: y(indices) for split, indices in splits.items()}
    results = []
    for C in [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]:
        t = time.time()
        clf = LogisticRegression(
            C=C,
            solver="saga",
            max_iter=500,
            n_jobs=-1,
            random_state=SEED,
            verbose=0,
            tol=1e-3,
        )
        clf.fit(X["train"], Y["train"])
        val_prob = clf.predict_proba(X["val"])[:, 1]
        val_f1, threshold = best_threshold(Y["val"], val_prob)
        result = {
            "C": C,
            "threshold": threshold,
            "val": metrics(Y["val"], val_prob, threshold),
            "test": metrics(Y["test"], clf.predict_proba(X["test"])[:, 1], threshold),
            "seconds": round(time.time() - t, 3),
        }
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
    results.sort(key=lambda item: item["val"]["f1"], reverse=True)
    final = {"best_by_val": results[0], "all": results, "total_seconds": round(time.time() - started, 3)}
    with open("results/ours_lr_tuning.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print("SUMMARY")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
