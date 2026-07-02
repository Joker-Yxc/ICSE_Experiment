#!/usr/bin/env python3
"""Tune TF-IDF features for the compact ours model."""

import gzip
import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


DATA = "datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz"
SEED = 7

rows = []
splits = {"train": [], "val": [], "test": []}
with gzip.open(DATA, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        rows.append(row)
        splits[row["split"]].append(i)


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
    train_texts = [text(i) for i in splits["train"]]
    val_texts = [text(i) for i in splits["val"]]
    test_texts = [text(i) for i in splits["test"]]
    Y = {split: y(indices) for split, indices in splits.items()}
    configs = [
        ("tfidf_word123", dict(analyzer="word", ngram_range=(1, 3), min_df=2, max_features=800000, sublinear_tf=True)),
        ("tfidf_word124", dict(analyzer="word", ngram_range=(1, 4), min_df=2, max_features=1000000, sublinear_tf=True)),
        ("tfidf_char", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=800000, sublinear_tf=True)),
    ]
    results = []
    for name, cfg in configs:
        print(f"[features] {name}", flush=True)
        vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\S+\b", **cfg)
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        X_test = vectorizer.transform(test_texts)
        print(f"[shape] {name} {X_train.shape}", flush=True)
        model_configs = [
            ("lr_C8", LogisticRegression(C=8.0, solver="saga", max_iter=500, random_state=SEED, tol=1e-3)),
            ("lr_C16", LogisticRegression(C=16.0, solver="saga", max_iter=500, random_state=SEED, tol=1e-3)),
            ("sgd_a1e-6", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-6, average=True, random_state=SEED)),
        ]
        for model_name, clf in model_configs:
            t = time.time()
            clf.fit(X_train, Y["train"])
            val_prob = clf.predict_proba(X_val)[:, 1]
            val_f1, threshold = best_threshold(Y["val"], val_prob)
            result = {
                "features": name,
                "model": model_name,
                "threshold": threshold,
                "val": metrics(Y["val"], val_prob, threshold),
                "test": metrics(Y["test"], clf.predict_proba(X_test)[:, 1], threshold),
                "seconds": round(time.time() - t, 3),
            }
            print(json.dumps(result, indent=2), flush=True)
            results.append(result)
    results.sort(key=lambda item: item["val"]["f1"], reverse=True)
    final = {"best_by_val": results[0], "all": results, "total_seconds": round(time.time() - started, 3)}
    with open("results/ours_tfidf_tuning.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print("SUMMARY")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
