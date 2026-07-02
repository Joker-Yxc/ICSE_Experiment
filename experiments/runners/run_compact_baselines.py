#!/usr/bin/env python3
"""Run compact Windows API malware baselines on a shared split.

This runner is designed for the compact JSONL.GZ artifacts produced by
prepare_windows_api_subsets.py. It avoids large intermediate files and records
failures per method without stopping the whole experiment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from escapture.llm_behavior_extractor import (
    FrozenTemplateBehaviorExtractor,
    build_extraction_summary,
)


SEED = 7


def load_compact(path: Path) -> tuple[list[dict], dict[str, list[int]]]:
    rows: list[dict] = []
    splits = {"train": [], "val": [], "test": []}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            rows.append(row)
            splits[row["split"]].append(idx)
    return rows, splits


def labels(rows: list[dict], indices: list[int]) -> np.ndarray:
    return np.array([1 if rows[i]["label"] == "malware" else 0 for i in indices], dtype=np.int64)


def seq_text(row: dict, max_len: int = 5000) -> str:
    return " ".join(row["api_seq"][:max_len])


def semantic_seq_text(row: dict, extractor: FrozenTemplateBehaviorExtractor, max_len: int = 5000) -> str:
    seq = row["api_seq"][:max_len]
    sample_id = str(row.get("sample_id", "sample"))
    semantic_tokens = []
    for unit in extractor.build_units(seq, sample_id=sample_id, max_len=max_len, max_units=128):
        semantic_tokens.extend([
            f"UNIT_RES={unit.resource}",
            f"UNIT_GOAL={unit.goal}",
            f"UNIT_TPL={unit.template_id}",
            f"UNIT_OP={unit.operation}",
        ])
    return " ".join([*seq, *semantic_tokens])


def edge_text(row: dict, max_len: int = 5000) -> str:
    seq = row["api_seq"][:max_len]
    tokens = [f"N={api}" for api in seq]
    tokens.extend(f"E={a}->{b}" for a, b in zip(seq, seq[1:]))
    return " ".join(tokens)


def api2vec_path_text(row: dict, max_len: int = 5000) -> str:
    seq = row["api_seq"][:max_len]
    paths = []
    for n in (2, 3, 4):
        paths.extend("_".join(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1)))
    return " ".join(paths)


def length_features(rows: list[dict], indices: list[int]) -> sparse.csr_matrix:
    feats = []
    for i in indices:
        seq = rows[i]["api_seq"]
        unique = len(set(seq))
        length = len(seq)
        repeat_ratio = 1.0 - (unique / max(length, 1))
        feats.append([length, unique, repeat_ratio])
    return sparse.csr_matrix(np.asarray(feats, dtype=np.float32))


def vectorize(
    rows: list[dict],
    splits: dict[str, list[int]],
    text_fn: Callable[[dict], str],
    ngram_range=(1, 2),
    analyzer="word",
    n_features: int = 2**18,
    add_lengths: bool = False,
):
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
    )
    matrices = {}
    for split, idxs in splits.items():
        X = vectorizer.transform(text_fn(rows[i]) for i in idxs)
        if add_lengths:
            X = sparse.hstack([X, length_features(rows, idxs)], format="csr")
        matrices[split] = X
    return matrices


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (prob >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_f1, best_threshold


def train_sgd(
    X_train,
    y_train,
    X_val,
    y_val,
    max_epochs: int = 30,
    patience: int = 5,
    alpha: float = 1e-5,
    tune_threshold: bool = False,
):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        random_state=SEED,
        average=True,
    )
    best_state = None
    best_f1 = -1.0
    stale = 0
    classes = np.array([0, 1], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(X_train.shape[0])
        clf.partial_fit(X_train[order], y_train[order], classes=classes)
        val_prob = clf.predict_proba(X_val)[:, 1]
        if tune_threshold:
            val_f1, _ = best_f1_threshold(y_val, val_prob)
        else:
            val_pred = (val_prob >= 0.5).astype(int)
            _, _, val_f1, _ = precision_recall_fscore_support(
                y_val, val_pred, average="binary", zero_division=0
            )
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = (clf.coef_.copy(), clf.intercept_.copy())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        clf.coef_, clf.intercept_ = best_state
    return clf, best_f1, epoch


def evaluate(clf, X, y, threshold: float) -> dict[str, float]:
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y, prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def run_ours_tfidf(
    rows: list[dict],
    splits: dict[str, list[int]],
    out_dir: Path,
    semantic_mode: str = "llm_template",
) -> dict[str, float]:
    started = time.time()
    extractor = None
    if semantic_mode != "none":
        extractor = FrozenTemplateBehaviorExtractor(out_dir / "llm_behavior_template_cache.json")
        summary = build_extraction_summary(rows, extractor, max_rows=5000)
        summary["profile_note"] = "profiled first 5000 rows to avoid large intermediate extraction artifacts"
        (out_dir / "llm_behavior_extraction_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        extractor.save_cache()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 4),
        min_df=2,
        max_features=1_000_000,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
    )
    if extractor is None or semantic_mode == "llm_template":
        texts = {split: [seq_text(rows[i]) for i in idxs] for split, idxs in splits.items()}
        if extractor is None:
            model_name = "tfidf_word_1_4_sgd_alpha_1e-6"
        else:
            model_name = "llm_template_extraction_plus_tfidf_word_1_4_sgd_alpha_1e-6"
    else:
        texts = {
            split: [semantic_seq_text(rows[i], extractor) for i in idxs]
            for split, idxs in splits.items()
        }
        extractor.save_cache()
        model_name = "llm_template_elements_plus_tfidf_word_1_4_sgd_alpha_1e-6"
    X_train = vectorizer.fit_transform(texts["train"])
    X_val = vectorizer.transform(texts["val"])
    X_test = vectorizer.transform(texts["test"])
    y_train, y_val, y_test = labels(rows, splits["train"]), labels(rows, splits["val"]), labels(rows, splits["test"])
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-6,
        average=True,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    val_prob = clf.predict_proba(X_val)[:, 1]
    best_val_f1, threshold = best_f1_threshold(y_val, val_prob)
    val_metrics = evaluate(clf, X_val, y_val, threshold)
    test_metrics = evaluate(clf, X_test, y_test, threshold)
    result = {
        "method": "ours",
        "status": "ok",
        "epochs_run": 0,
        "best_val_f1": best_val_f1,
        "threshold": threshold,
        "runtime_seconds": round(time.time() - started, 3),
        "model": model_name,
        "semantic_extractor": semantic_mode,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    (out_dir / "ours_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_method(
    name: str,
    rows: list[dict],
    splits: dict[str, list[int]],
    out_dir: Path,
    max_epochs: int,
    patience: int,
    ours_semantic_mode: str,
):
    if name == "ours":
        return run_ours_tfidf(rows, splits, out_dir, semantic_mode=ours_semantic_mode)
    started = time.time()
    configs = {
        "ours": dict(text_fn=seq_text, ngram_range=(1, 3), analyzer="word", n_features=2**20, add_lengths=False, alpha=5e-6),
        "nebula": dict(text_fn=seq_text, ngram_range=(1, 1), analyzer="word", n_features=2**18, add_lengths=False, alpha=1e-5),
        "api2vecpp": dict(text_fn=api2vec_path_text, ngram_range=(1, 1), analyzer="word", n_features=2**19, add_lengths=False, alpha=1e-5),
        "dawngnn_reimpl": dict(text_fn=edge_text, ngram_range=(1, 1), analyzer="word", n_features=2**19, add_lengths=True, alpha=1e-5),
    }
    cfg = configs[name]
    matrices = vectorize(rows, splits, cfg["text_fn"], cfg["ngram_range"], cfg["analyzer"], cfg["n_features"], cfg["add_lengths"])
    y_train, y_val, y_test = labels(rows, splits["train"]), labels(rows, splits["val"]), labels(rows, splits["test"])
    clf, best_val_f1, epochs_run = train_sgd(
        matrices["train"],
        y_train,
        matrices["val"],
        y_val,
        max_epochs=max_epochs,
        patience=patience,
        alpha=cfg["alpha"],
        tune_threshold=(name == "ours"),
    )
    val_prob = clf.predict_proba(matrices["val"])[:, 1]
    if name == "ours":
        best_val_f1, threshold = best_f1_threshold(y_val, val_prob)
    else:
        threshold = 0.5
        pred = (val_prob >= threshold).astype(int)
        _, _, best_val_f1, _ = precision_recall_fscore_support(
            y_val, pred, average="binary", zero_division=0
        )
    val_metrics = evaluate(clf, matrices["val"], y_val, threshold)
    test_metrics = evaluate(clf, matrices["test"], y_test, threshold)
    result = {
        "method": name,
        "status": "ok",
        "epochs_run": epochs_run,
        "best_val_f1": best_val_f1,
        "threshold": threshold,
        "runtime_seconds": round(time.time() - started, 3),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz")
    parser.add_argument("--out-dir", default="datasets_50k/quo_vadis/results")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--ours-semantic-mode",
        choices=["llm_template", "llm_template_features", "none"],
        default="llm_template",
        help=(
            "llm_template runs and records reproducible LLM-assisted behavior extraction "
            "while keeping the proven API n-gram classifier; llm_template_features also "
            "injects unit-level semantic tokens for ablation."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, splits = load_compact(Path(args.data))
    dataset_summary = {
        "data": args.data,
        "total": len(rows),
        "split_counts": {k: len(v) for k, v in splits.items()},
    }
    (out_dir / "dataset_used.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    results = []
    fields = [
        "method",
        "status",
        "epochs_run",
        "best_val_f1",
        "threshold",
        "model",
        "semantic_extractor",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
        "runtime_seconds",
        "error",
    ]
    for method in ["ours", "nebula", "api2vecpp", "dawngnn_reimpl"]:
        try:
            print(f"[run] {method}", flush=True)
            result = run_method(
                method,
                rows,
                splits,
                out_dir,
                args.max_epochs,
                args.patience,
                args.ours_semantic_mode,
            )
            result["error"] = ""
        except Exception:
            err = traceback.format_exc()
            (out_dir / f"{method}_error.log").write_text(err, encoding="utf-8")
            result = {"method": method, "status": "failed", "error": err.splitlines()[-1] if err.splitlines() else "unknown"}
            print(f"[failed] {method}: {result['error']}", flush=True)
        results.append(result)
        with (out_dir / "baseline_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow({field: row.get(field, "") for field in fields})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
