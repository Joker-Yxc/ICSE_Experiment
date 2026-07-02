#!/usr/bin/env python3
"""Tune an ensemble for the compact ours model using validation only."""

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


def labels(indices):
    return np.array([1 if rows[i]["label"] == "malware" else 0 for i in indices], dtype=np.int64)


def seq_text(row):
    return " ".join(row["api_seq"])


def path_text(row):
    seq = row["api_seq"]
    out = []
    for n in (2, 3, 4):
        out.extend("_".join(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1)))
    return " ".join(out)


def edge_text(row):
    seq = row["api_seq"]
    out = [f"N={api}" for api in seq]
    out.extend(f"E={a}->{b}" for a, b in zip(seq, seq[1:]))
    return " ".join(out)


def len_features(indices):
    feats = []
    for i in indices:
        seq = rows[i]["api_seq"]
        length = len(seq)
        unique = len(set(seq))
        repeat_ratio = 1.0 - unique / max(length, 1)
        feats.append([np.log1p(length) / 10.0, np.log1p(unique) / 10.0, repeat_ratio])
    return sparse.csr_matrix(np.asarray(feats, dtype=np.float32))


def vectorize(text_fn, ngram_range=(1, 1), n_features=2**20, add_lengths=False):
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
        ngram_range=ngram_range,
    )
    mats = {}
    for split, idxs in splits.items():
        X = vectorizer.transform(text_fn(rows[i]) for i in idxs)
        if add_lengths:
            X = sparse.hstack([X, len_features(idxs)], format="csr")
        mats[split] = X
    return mats


def best_threshold(y_true, prob):
    y_true = np.asarray(y_true, dtype=bool)
    best = {"f1": -1.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for threshold in np.linspace(0.02, 0.98, 193):
        pred = prob >= threshold
        tp = np.count_nonzero(pred & y_true)
        fp = np.count_nonzero(pred & ~y_true)
        fn = np.count_nonzero(~pred & y_true)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, prob)),
    }


def train_view(name, text_fn, ngram_range, n_features, add_lengths, alpha):
    started = time.time()
    print(f"[view] {name}", flush=True)
    mats = vectorize(text_fn, ngram_range, n_features, add_lengths)
    y_train = labels(splits["train"])
    y_val = labels(splits["val"])
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        average=True,
        random_state=SEED,
    )
    classes = np.array([0, 1], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    best_state = None
    best_f1 = -1.0
    stale = 0
    best_epoch = 0
    for epoch in range(1, 31):
        order = rng.permutation(mats["train"].shape[0])
        clf.partial_fit(mats["train"][order], y_train[order], classes=classes)
        val_prob = clf.predict_proba(mats["val"])[:, 1]
        info = best_threshold(y_val, val_prob)
        if info["f1"] > best_f1:
            best_f1 = info["f1"]
            best_state = (clf.coef_.copy(), clf.intercept_.copy())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 5:
            break
    clf.coef_, clf.intercept_ = best_state
    out = {
        "name": name,
        "best_epoch": best_epoch,
        "val_prob": clf.predict_proba(mats["val"])[:, 1],
        "test_prob": clf.predict_proba(mats["test"])[:, 1],
        "seconds": round(time.time() - started, 3),
    }
    print(f"[view-done] {name} epoch={best_epoch} seconds={out['seconds']}", flush=True)
    return out


def main():
    y_val = labels(splits["val"])
    y_test = labels(splits["test"])
    views = [
        train_view("seq123", seq_text, (1, 3), 2**20, False, 5e-6),
        train_view("path234", path_text, (1, 1), 2**20, False, 1e-5),
        train_view("edge", edge_text, (1, 1), 2**20, False, 1e-5),
        train_view("edge_len", edge_text, (1, 1), 2**20, True, 1e-5),
    ]
    weight_candidates = []
    base = np.eye(len(views))
    weight_candidates.extend(base)
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            for a in (0.25, 0.4, 0.5, 0.6, 0.75):
                w = np.zeros(len(views), dtype=np.float64)
                w[i] = a
                w[j] = 1.0 - a
                weight_candidates.append(w)
    for raw in ([0.5, 0.3, 0.2, 0.0], [0.6, 0.25, 0.15, 0.0], [0.7, 0.2, 0.1, 0.0],
                [0.5, 0.25, 0.15, 0.1], [0.4, 0.3, 0.2, 0.1]):
        weight_candidates.append(np.asarray(raw, dtype=np.float64))
    candidates = []
    for weights in weight_candidates:
        weights = weights / weights.sum()
        val_prob = sum(w * view["val_prob"] for w, view in zip(weights, views))
        threshold_info = best_threshold(y_val, val_prob)
        candidates.append((threshold_info["f1"], threshold_info["threshold"], weights, threshold_info))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_f1, threshold, weights, threshold_info = candidates[0]
    val_prob = sum(w * view["val_prob"] for w, view in zip(weights, views))
    test_prob = sum(w * view["test_prob"] for w, view in zip(weights, views))
    result = {
        "views": [{k: v for k, v in view.items() if not k.endswith("_prob")} for view in views],
        "weights": {view["name"]: float(w) for view, w in zip(views, weights)},
        "threshold": float(threshold),
        "val": metrics(y_val, val_prob, threshold),
        "test": metrics(y_test, test_prob, threshold),
    }
    print(json.dumps(result, indent=2), flush=True)
    with open("results/ours_ensemble_tuning.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
