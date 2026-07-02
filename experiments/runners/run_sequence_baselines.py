#!/usr/bin/env python3
"""Run standard simple and neural sequence baselines on compact splits."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.svm import LinearSVC

MODELS = (
    "api_frequency_lr",
    "api_frequency_rf",
    "api_frequency_xgboost",
    "api_ngram_lr",
    "api_ngram_svm",
    "api_ngram_xgboost",
    "lstm",
    "gru",
    "transformer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--limit-per-split", type=int, default=0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rows(path: Path, limit: int) -> dict[str, list[dict]]:
    rows = {"train": [], "val": [], "test": []}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = row["split"]
            if limit and len(rows[split]) >= limit:
                continue
            rows[split].append(row)
    if any(not rows[split] for split in rows):
        raise ValueError(f"Every split must be non-empty: { {k: len(v) for k, v in rows.items()} }")
    return rows


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([int(row["label"] == "malware") for row in rows], dtype=np.int64)


def threshold_for_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 301)))
    best = (-1.0, 0.5)
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int64)
        f1 = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )[2]
        if f1 > best[0]:
            best = (float(f1), float(threshold))
    return best[1], best[0]


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
    }


def save_predictions(path: Path, rows: list[dict], scores: np.ndarray, threshold: float) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row, score in zip(rows, scores):
            record = {
                "sample_id": row["sample_id"],
                "family": row.get("family", "unknown"),
                "label": int(row["label"] == "malware"),
                "score": float(score),
                "prediction": int(score >= threshold),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def text(row: dict, max_length: int) -> str:
    return " ".join(str(call) for call in row["api_seq"][:max_length])


def estimator_scores(model, matrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    raw = np.asarray(model.decision_function(matrix), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))


def run_classical(name: str, rows: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    frequency = name.startswith("api_frequency")
    vectorizer = (
        CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\S+\b",
            binary=False,
        )
        if frequency
        else TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\S+\b",
            ngram_range=(1, 4),
            min_df=2,
            max_features=1_000_000,
            sublinear_tf=True,
        )
    )
    matrices = {}
    matrices["train"] = vectorizer.fit_transform(
        text(row, args.max_length) for row in rows["train"]
    )
    matrices["val"] = vectorizer.transform(text(row, args.max_length) for row in rows["val"])
    matrices["test"] = vectorizer.transform(text(row, args.max_length) for row in rows["test"])
    y_train = labels(rows["train"])

    if name.endswith("_lr"):
        model = LogisticRegression(max_iter=2000, random_state=args.seed, n_jobs=-1)
    elif name.endswith("_rf"):
        model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )
    elif name.endswith("_svm"):
        model = LinearSVC(class_weight="balanced", random_state=args.seed)
    elif name.endswith("_xgboost"):
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "xgboost is unavailable; install xgboost and its native runtime "
                "(libomp on macOS)"
            ) from exc
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:  # pragma: no cover
        raise ValueError(name)

    model.fit(matrices["train"], y_train)
    val_scores = estimator_scores(model, matrices["val"])
    test_scores = estimator_scores(model, matrices["test"])
    threshold, _ = threshold_for_f1(labels(rows["val"]), val_scores)
    parameter_count = (
        int(np.prod(model.coef_.shape) + np.prod(model.intercept_.shape))
        if hasattr(model, "coef_")
        else None
    )
    return {
        "validation_scores": val_scores,
        "test_scores": test_scores,
        "parameter_count": parameter_count,
        "feature_count": len(vectorizer.vocabulary_),
        "epochs_run": None,
    }


class Vocabulary:
    def __init__(self, train_rows: list[dict], max_length: int):
        counts = Counter(
            str(call)
            for row in train_rows
            for call in row["api_seq"][:max_length]
        )
        self.to_id = {"<PAD>": 0, "<UNK>": 1}
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            self.to_id[token] = len(self.to_id)

    def encode(self, row: dict, max_length: int) -> list[int]:
        return [
            self.to_id.get(str(call), 1)
            for call in row["api_seq"][:max_length]
        ]

    def __len__(self) -> int:
        return len(self.to_id)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], vocab: Vocabulary, max_length: int):
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        ids = self.vocab.encode(row, self.max_length) or [1]
        return torch.tensor(ids), int(row["label"] == "malware")


def collate(batch):
    sequences, labels_batch = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels_batch, dtype=torch.float32)


class NeuralSequenceModel(nn.Module):
    def __init__(self, kind: str, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if kind == "lstm":
            self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            output_dim = hidden_dim * 2
        elif kind == "gru":
            self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            output_dim = hidden_dim * 2
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            output_dim = embed_dim
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(output_dim, 1))

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        if self.kind in {"lstm", "gru"}:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.encoder(packed)
            if self.kind == "lstm":
                hidden = hidden[0]
            pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            mask = tokens.eq(0)
            encoded = self.encoder(embedded, src_key_padding_mask=mask)
            valid = (~mask).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled).squeeze(-1)


def neural_scores(model, loader, device) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for tokens, lengths, _ in loader:
            logits = model(tokens.to(device), lengths.to(device))
            values.extend(torch.sigmoid(logits).cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def run_neural(name: str, rows: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    device = get_device(args.device)
    vocab = Vocabulary(rows["train"], args.max_length)
    loaders = {
        split: torch.utils.data.DataLoader(
            SequenceDataset(split_rows, vocab, args.max_length),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split, split_rows in rows.items()
    }
    model = NeuralSequenceModel(
        name, len(vocab), args.embedding_dim, args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_f1 = -1.0
    stale = 0
    epochs_run = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for tokens, lengths, target in loaders["train"]:
            optimizer.zero_grad()
            logits = model(tokens.to(device), lengths.to(device))
            loss = criterion(logits, target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_scores = neural_scores(model, loaders["val"], device)
        _, val_f1 = threshold_for_f1(labels(rows["val"]), val_scores)
        epochs_run = epoch
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Neural training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "validation_scores": neural_scores(model, loaders["val"], device),
        "test_scores": neural_scores(model, loaders["test"], device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_count": len(vocab),
        "epochs_run": epochs_run,
        "state_dict": best_state,
    }


def run_one(name: str, rows: dict[str, list[dict]], args: argparse.Namespace, out_dir: Path) -> dict:
    seed_everything(args.seed)
    started = time.perf_counter()
    payload = (
        run_classical(name, rows, args)
        if name.startswith("api_")
        else run_neural(name, rows, args)
    )
    threshold, best_val_f1 = threshold_for_f1(
        labels(rows["val"]), payload["validation_scores"]
    )
    run_dir = out_dir / name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(
        run_dir / "validation_predictions.jsonl.gz",
        rows["val"],
        payload["validation_scores"],
        threshold,
    )
    save_predictions(
        run_dir / "test_predictions.jsonl.gz",
        rows["test"],
        payload["test_scores"],
        threshold,
    )
    if "state_dict" in payload:
        torch.save(payload["state_dict"], run_dir / "best_model.pt")
    result = {
        "method": name,
        "seed": args.seed,
        "data": args.data,
        "data_sha256": file_sha256(Path(args.data)),
        "split_counts": {split: len(values) for split, values in rows.items()},
        "max_length": args.max_length,
        "threshold": threshold,
        "best_validation_f1": best_val_f1,
        "validation": compute_metrics(
            labels(rows["val"]), payload["validation_scores"], threshold
        ),
        "test": compute_metrics(labels(rows["test"]), payload["test_scores"], threshold),
        "parameter_count": payload["parameter_count"],
        "feature_or_vocabulary_count": payload["feature_count"],
        "epochs_run": payload["epochs_run"],
        "runtime_seconds": time.perf_counter() - started,
        "config": vars(args),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    requested = MODELS if args.models == "all" else tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    unknown = sorted(set(requested) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    rows = load_rows(Path(args.data), args.limit_per_split)
    out_dir = Path(args.out_dir)
    results = []
    for name in requested:
        print(f"[run] {name} seed={args.seed}", flush=True)
        results.append(run_one(name, rows, args, out_dir))
    (out_dir / f"standard_baselines_seed_{args.seed}.json").write_text(
        json.dumps(results, indent=2, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
