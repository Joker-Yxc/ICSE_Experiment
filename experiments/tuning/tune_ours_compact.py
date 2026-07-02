#!/usr/bin/env python3
"""Validation-only tuning for the compact 'ours' adapter."""

import gzip
import json
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


SEED = 7
DATA = "datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz"


rows = []
splits = {"train": [], "val": [], "test": []}
with gzip.open(DATA, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        rows.append(row)
        splits[row["split"]].append(i)


def y(idxs):
    return np.array([1 if rows[i]["label"] == "malware" else 0 for i in idxs])


def seq(row):
    return " ".join(row["api_seq"])


def edge(row):
    s = row["api_seq"]
    tokens = [f"N={x}" for x in s]
    tokens += [f"E={a}->{b}" for a, b in zip(s, s[1:])]
    return " ".join(tokens)


def path(row):
    s = row["api_seq"]
    out = []
    for n in (2, 3, 4):
        out += ["_".join(s[i : i + n]) for i in range(max(0, len(s) - n + 1))]
    return " ".join(out)


def hybrid(row):
    return seq(row) + " " + edge(row) + " " + path(row)


def len_scaled(idxs):
    feats = []
    for i in idxs:
        s = rows[i]["api_seq"]
        length = len(s)
        unique = len(set(s))
        repeat_ratio = 1.0 - unique / max(length, 1)
        feats.append([np.log1p(length) / 10.0, np.log1p(unique) / 10.0, repeat_ratio])
    return sparse.csr_matrix(np.array(feats, dtype=np.float32))


def best_thresh(ytrue, prob):
    best = (0.0, 0.5, 0.0, 0.0)
    for th in np.linspace(0.05, 0.95, 181):
        pred = (prob >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(ytrue, pred, average="binary", zero_division=0)
        if f1 > best[0]:
            best = (float(f1), float(th), float(p), float(r))
    return best


def eval_metrics(ytrue, prob, th):
    pred = (prob >= th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(ytrue, pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(ytrue, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(roc_auc_score(ytrue, prob)),
    }


def run(name, fn, ngram=(1, 2), n_features=2**20, add_len=False, alpha=1e-5):
    started = time.time()
    print(f"\nRUN {name}", flush=True)
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
        ngram_range=ngram,
    )
    matrices = {}
    for split, idxs in splits.items():
        X = vectorizer.transform(fn(rows[i]) for i in idxs)
        if add_len:
            X = sparse.hstack([X, len_scaled(idxs)], format="csr")
        matrices[split] = X
    y_train, y_val, y_test = y(splits["train"]), y(splits["val"]), y(splits["test"])
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        average=True,
        random_state=SEED,
    )
    classes = np.array([0, 1])
    rng = np.random.default_rng(SEED)
    best_state = None
    best_f1 = -1.0
    stale = 0
    best_epoch = 0
    for epoch in range(1, 31):
        order = rng.permutation(matrices["train"].shape[0])
        clf.partial_fit(matrices["train"][order], y_train[order], classes=classes)
        val_prob = clf.predict_proba(matrices["val"])[:, 1]
        val_f1, _, _, _ = best_thresh(y_val, val_prob)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = (clf.coef_.copy(), clf.intercept_.copy())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 5:
            break
    clf.coef_, clf.intercept_ = best_state
    val_prob = clf.predict_proba(matrices["val"])[:, 1]
    _, threshold, _, _ = best_thresh(y_val, val_prob)
    val_metrics = eval_metrics(y_val, val_prob, threshold)
    test_metrics = eval_metrics(y_test, clf.predict_proba(matrices["test"])[:, 1], threshold)
    result = {
        "name": name,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "val": val_metrics,
        "test": test_metrics,
        "seconds": round(time.time() - started, 3),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    configs = [
        ("seq123_no_len", seq, (1, 3), 2**20, False, 5e-6),
        ("seq123_scaled_len", seq, (1, 3), 2**20, True, 5e-6),
        ("edge_no_len", edge, (1, 1), 2**20, False, 1e-5),
        ("path_no_len", path, (1, 1), 2**20, False, 1e-5),
        ("hybrid_no_len", hybrid, (1, 1), 2**20, False, 1e-5),
        ("hybrid_scaled_len", hybrid, (1, 1), 2**20, True, 1e-5),
    ]
    results = [run(*cfg) for cfg in configs]
    results.sort(key=lambda item: item["val"]["f1"], reverse=True)
    print("\nSUMMARY", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
