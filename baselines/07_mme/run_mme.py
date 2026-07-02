#!/usr/bin/env python3
"""MME-style knowledge-enhanced TextCNN for Windows API sequences.

The official MME repository currently publishes the paper metadata and links to
precomputed data/knowledge graph files, but no training source. This runner
therefore implements a documented fallback:

* transition/co-occurrence graph built from the training split only;
* API-name/category resource encoding;
* graph-smoothed embedding initialization;
* TextCNN binary detector with supervised contrastive and family losses.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_DATASETS = {
    "quo_vadis": WORKSPACE_ROOT / "datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz",
    "zenodo_11079764": WORKSPACE_ROOT
    / "datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz",
}

RESOURCE_CATEGORIES = (
    "file",
    "registry",
    "process",
    "thread",
    "memory",
    "network",
    "service",
    "security",
    "sync",
    "system",
    "library",
    "other",
)

CATEGORY_HINTS = {
    "file": ("file", "directory", "path", "volume", "device", "io", "pipe"),
    "registry": ("reg", "key", "valuekey"),
    "process": ("process", "job", "debug"),
    "thread": ("thread", "fiber", "tls"),
    "memory": ("memory", "virtual", "heap", "section", "mapview", "protect"),
    "network": ("socket", "http", "internet", "connect", "send", "recv", "dns", "wininet"),
    "service": ("service", "scmanager"),
    "security": ("token", "privilege", "security", "crypt", "cert", "acl", "sid"),
    "sync": ("mutex", "mutant", "event", "semaphore", "wait", "criticalsection"),
    "system": ("system", "time", "performance", "computer", "environment", "locale"),
    "library": ("library", "module", "ldr", "procaddress", "resource"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_DATASETS), required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-api-frequency", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--filters", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--family-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k-neighbors", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Smoke-test limit; 0 uses all rows.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not append baseline_results.csv.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_api(name: str) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower()) or "<unk>"


def api_category(name: str) -> int:
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    for category, hints in CATEGORY_HINTS.items():
        if any(hint in compact for hint in hints):
            return RESOURCE_CATEGORIES.index(category)
    return RESOURCE_CATEGORIES.index("other")


def read_rows(path: Path, limit_per_split: int = 0) -> dict[str, list[dict]]:
    rows = {"train": [], "val": [], "test": []}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = row.get("split")
            if split not in rows:
                raise ValueError(f"Unexpected split {split!r} in {path}")
            if limit_per_split and len(rows[split]) >= limit_per_split:
                continue
            if not isinstance(row.get("api_seq"), list) or not row["api_seq"]:
                continue
            rows[split].append(row)
    if any(not split_rows for split_rows in rows.values()):
        raise ValueError(f"Every split must contain samples: {path}")
    return rows


def build_vocabulary(train_rows: list[dict], min_frequency: int) -> tuple[dict[str, int], list[str]]:
    counts = Counter(normalize_api(api) for row in train_rows for api in row["api_seq"])
    tokens = [api for api, count in counts.most_common() if count >= min_frequency]
    id_to_api = ["<pad>", "<unk>"] + tokens
    return {api: idx for idx, api in enumerate(id_to_api)}, id_to_api


def build_graph_and_resources(
    train_rows: list[dict],
    api_to_id: dict[str, int],
    id_to_api: list[str],
    top_k: int,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    edge_counts: dict[int, Counter] = defaultdict(Counter)
    for row in train_rows:
        ids = [api_to_id.get(normalize_api(api), 1) for api in row["api_seq"]]
        for left, right in zip(ids, ids[1:]):
            if left > 1 and right > 1 and left != right:
                edge_counts[left][right] += 1
                edge_counts[right][left] += 1

    category_count = len(RESOURCE_CATEGORIES)
    lexical_dim = 16
    resources = np.zeros((len(id_to_api), category_count + lexical_dim), dtype=np.float32)
    for idx, api in enumerate(id_to_api[2:], start=2):
        resources[idx, api_category(api)] = 1.0
        for chunk in re.findall(r"[a-z]+|\d+", api):
            bucket = int(hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:8], 16) % lexical_dim
            resources[idx, category_count + bucket] += 1.0
        norm = np.linalg.norm(resources[idx])
        if norm:
            resources[idx] /= norm

    graph_context = resources.copy()
    serialized_edges = []
    for src, neighbors in edge_counts.items():
        selected = neighbors.most_common(top_k)
        if not selected:
            continue
        weights = np.asarray([np.log1p(weight) for _, weight in selected], dtype=np.float32)
        weights /= weights.sum()
        graph_context[src] = 0.5 * resources[src] + 0.5 * sum(
            weight * resources[dst] for weight, (dst, _) in zip(weights, selected)
        )
        serialized_edges.extend(
            {"source": id_to_api[src], "target": id_to_api[dst], "count": count}
            for dst, count in selected
            if src < dst
        )

    features = np.concatenate([resources, graph_context], axis=1)
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, 1 / np.sqrt(features.shape[1]), (features.shape[1], embedding_dim))
    embeddings = (features @ projection).astype(np.float32)
    embeddings[0] = 0
    graph = {
        "type": "fallback_train_transition_cooccurrence",
        "top_k_neighbors": top_k,
        "node_count": len(id_to_api) - 2,
        "edge_count": len(serialized_edges),
        "resource_categories": list(RESOURCE_CATEGORIES),
        "edges": serialized_edges,
    }
    return embeddings, graph


def encode_rows(
    rows: list[dict],
    api_to_id: dict[str, int],
    family_to_id: dict[str, int],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), max_length), dtype=np.int32)
    y = np.zeros(len(rows), dtype=np.int64)
    y_family = np.full(len(rows), -1, dtype=np.int64)
    sample_ids = np.empty(len(rows), dtype=f"<U{max(1, max(len(str(r['sample_id'])) for r in rows))}")
    for idx, row in enumerate(rows):
        ids = [api_to_id.get(normalize_api(api), 1) for api in row["api_seq"][:max_length]]
        x[idx, : len(ids)] = ids
        y[idx] = int(str(row["label"]).lower() == "malware")
        if y[idx]:
            y_family[idx] = family_to_id.get(str(row.get("family", "")).lower(), -1)
        sample_ids[idx] = str(row["sample_id"])
    return x, y, y_family, sample_ids


def prepare_cache(args: argparse.Namespace, data_path: Path, cache_dir: Path) -> tuple[dict, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    expected = {
        "data_path": str(data_path.resolve()),
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "max_length": args.max_length,
        "min_api_frequency": args.min_api_frequency,
        "seed": args.seed,
        "limit_per_split": args.limit_per_split,
    }
    files = {split: cache_dir / f"{split}.npz" for split in ("train", "val", "test")}
    if not args.rebuild_cache and manifest_path.exists() and all(path.exists() for path in files.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            arrays = {split: dict(np.load(path)) for split, path in files.items()}
            return arrays, manifest

    rows = read_rows(data_path, args.limit_per_split)
    api_to_id, id_to_api = build_vocabulary(rows["train"], args.min_api_frequency)
    malware_families = sorted(
        {str(row.get("family", "")).lower() for row in rows["train"] if row["label"] == "malware"}
    )
    family_to_id = {family: idx for idx, family in enumerate(malware_families)}
    embeddings, graph = build_graph_and_resources(
        rows["train"], api_to_id, id_to_api, args.top_k_neighbors, args.embedding_dim, args.seed
    )

    arrays = {}
    for split, split_rows in rows.items():
        x, y, y_family, sample_ids = encode_rows(
            split_rows, api_to_id, family_to_id, args.max_length
        )
        np.savez_compressed(files[split], x=x, y=y, y_family=y_family, sample_id=sample_ids)
        arrays[split] = {"x": x, "y": y, "y_family": y_family, "sample_id": sample_ids}

    np.save(cache_dir / "fallback_embedding_init.npy", embeddings)
    (cache_dir / "fallback_knowledge_graph.json").write_text(
        json.dumps(graph, indent=2), encoding="utf-8"
    )
    (cache_dir / "vocabulary.json").write_text(
        json.dumps({"id_to_api": id_to_api, "family_to_id": family_to_id}, indent=2),
        encoding="utf-8",
    )
    manifest = {
        **expected,
        "split_counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "vocab_size": len(id_to_api),
        "family_count": len(family_to_id),
        "knowledge_graph": "fallback_knowledge_graph.json",
        "resource_encoding": "API name/category deterministic hashed encoding",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return arrays, manifest


class MMETextCNN(nn.Module):
    def __init__(
        self,
        embedding_init: np.ndarray,
        filters: int,
        family_count: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        vocab_size, embedding_dim = embedding_init.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_init))
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, filters, kernel_size=size) for size in (3, 4, 5)]
        )
        hidden_dim = filters * len(self.convs)
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.family_head = nn.Linear(hidden_dim, family_count) if family_count else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).transpose(1, 2)
        pooled = [F.adaptive_max_pool1d(F.relu(conv(embedded)), 1).squeeze(-1) for conv in self.convs]
        return self.dropout(torch.cat(pooled, dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        features = self.encode(x)
        binary = self.binary_head(features).squeeze(1)
        family = self.family_head(features) if self.family_head is not None else None
        return binary, family, features


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits.masked_fill(identity, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    return -((log_prob * positive).sum(dim=1) / positive_count).mean()


def make_loader(arrays: dict, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays["x"]).long(),
        torch.from_numpy(arrays["y"]).float(),
        torch.from_numpy(arrays["y_family"]).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict(model: MMETextCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, ...]:
    model.eval()
    probabilities, labels, families, family_scores = [], [], [], []
    for x, y, y_family in loader:
        binary_logits, family_logits, _ = model(x.to(device))
        probabilities.append(torch.sigmoid(binary_logits).cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
        families.append(y_family.numpy())
        if family_logits is not None:
            family_scores.append(torch.softmax(family_logits, dim=1).cpu().numpy())
    scores = np.concatenate(family_scores) if family_scores else np.empty((len(np.concatenate(labels)), 0))
    return np.concatenate(probabilities), np.concatenate(labels), np.concatenate(families), scores


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = probabilities >= threshold
        f1 = precision_recall_fscore_support(
            y_true, prediction, average="binary", zero_division=0
        )[2]
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)
    return best_threshold


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    prediction = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, probabilities)),
    }


def family_metrics(y_family: np.ndarray, scores: np.ndarray) -> dict | None:
    mask = y_family >= 0
    if not mask.any() or scores.shape[1] == 0:
        return None
    truth = y_family[mask]
    prediction = scores[mask].argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="macro", zero_division=0
    )
    result = {
        "sample_count": int(mask.sum()),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    per_class_auc = []
    for class_id in range(scores.shape[1]):
        binary_truth = truth == class_id
        if binary_truth.any() and (~binary_truth).any():
            per_class_auc.append(roc_auc_score(binary_truth, scores[mask, class_id]))
    result["macro_auc_ovr"] = float(np.mean(per_class_auc)) if per_class_auc else None
    result["auc_class_count"] = len(per_class_auc)
    return result


def train(
    args: argparse.Namespace,
    arrays: dict,
    embedding_init: np.ndarray,
    family_count: int,
    device: torch.device,
    log,
) -> tuple[MMETextCNN, int, float, dict]:
    train_loader = make_loader(arrays["train"], args.batch_size, True, args.num_workers)
    val_loader = make_loader(arrays["val"], args.batch_size * 2, False, args.num_workers)
    model = MMETextCNN(embedding_init, args.filters, family_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state, best_epoch, best_f1, stale = None, 0, -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y, y_family in train_loader:
            x, y, y_family = x.to(device), y.to(device), y_family.to(device)
            optimizer.zero_grad(set_to_none=True)
            binary_logits, family_logits, features = model(x)
            loss = F.binary_cross_entropy_with_logits(binary_logits, y)
            loss = loss + args.contrastive_weight * supervised_contrastive_loss(
                features, y.long(), args.temperature
            )
            family_mask = y_family >= 0
            if family_logits is not None and family_mask.any():
                loss = loss + args.family_weight * F.cross_entropy(
                    family_logits[family_mask], y_family[family_mask]
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(x)

        val_prob, val_y, _, _ = predict(model, val_loader, device)
        threshold = choose_threshold(val_y, val_prob)
        metrics = binary_metrics(val_y, val_prob, threshold)
        entry = {"epoch": epoch, "train_loss": epoch_loss / len(train_loader.dataset), **metrics}
        history.append(entry)
        log(json.dumps(entry, sort_keys=True))
        if metrics["f1"] > best_f1 + 1e-6:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, best_f1, {"epochs": history}


def append_result_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            fieldnames = next(csv.reader(handle))
    else:
        fieldnames = list(row)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_path = args.data_path or DEFAULT_DATASETS[args.dataset]
    dataset_root = data_path.parents[1]
    output_path = args.output_path or dataset_root / "results/mme_metrics.json"
    cache_dir = (
        args.cache_dir
        or WORKSPACE_ROOT / "datasets_50k" / args.dataset / "artifacts" / "mme"
    )
    log_path = args.log_path or BASELINE_ROOT / "logs" / f"{args.dataset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        def log(message: str) -> None:
            print(message, flush=True)
            print(message, file=log_handle, flush=True)

        started = time.time()
        device = get_device(args.device)
        log(f"dataset={args.dataset} data={data_path} device={device}")
        arrays, manifest = prepare_cache(args, data_path, cache_dir)
        embedding_init = np.load(cache_dir / "fallback_embedding_init.npy")
        family_count = int(manifest["family_count"])
        log(f"split_counts={manifest['split_counts']} vocab={manifest['vocab_size']} families={family_count}")

        model, best_epoch, best_val_f1, training = train(
            args, arrays, embedding_init, family_count, device, log
        )
        val_loader = make_loader(arrays["val"], args.batch_size * 2, False, args.num_workers)
        test_loader = make_loader(arrays["test"], args.batch_size * 2, False, args.num_workers)
        val_prob, val_y, val_family, val_family_scores = predict(model, val_loader, device)
        threshold = choose_threshold(val_y, val_prob)
        test_prob, test_y, test_family, test_family_scores = predict(model, test_loader, device)
        val_result = binary_metrics(val_y, val_prob, threshold)
        test_result = binary_metrics(test_y, test_prob, threshold)
        runtime = time.time() - started

        result = {
            "method": "mme_textcnn_fallback",
            "status": "ok",
            "dataset": args.dataset,
            "seed": args.seed,
            "split_policy": "used existing split field without resplitting",
            "input": "api_seq",
            "official_code_commit": "8a055f0",
            "official_training_code_available": False,
            "knowledge_graph": "fallback training-only API transition/co-occurrence graph",
            "resource_encoding": "fallback API name/category encoding",
            "model": "MME-TextCNN-style multitask classifier",
            "contrastive_learning": "supervised contrastive auxiliary loss",
            "epochs_run": len(training["epochs"]),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "threshold": threshold,
            "val": val_result,
            "test": test_result,
            "family_val": family_metrics(val_family, val_family_scores),
            "family_test": family_metrics(test_family, test_family_scores),
            "runtime_seconds": runtime,
            "config": vars(args) | {"data_path": str(data_path), "cache_dir": str(cache_dir)},
            "cache_manifest": manifest,
            "training_history": training["epochs"],
        }
        output_path.write_text(
            json.dumps(result, indent=2, default=str, allow_nan=False), encoding="utf-8"
        )
        torch.save(
            {"state_dict": model.state_dict(), "config": result["config"]},
            cache_dir / "mme_textcnn_best.pt",
        )

        csv_row = {
            "method": "mme_textcnn_fallback",
            "status": "ok",
            "epochs_run": result["epochs_run"],
            "best_val_f1": best_val_f1,
            "threshold": threshold,
            "model": result["model"],
            "semantic_extractor": "fallback_api_knowledge_graph_resource_encoding",
            **{f"val_{key}": value for key, value in val_result.items()},
            **{f"test_{key}": value for key, value in test_result.items()},
            "runtime_seconds": runtime,
            "error": "",
        }
        if not args.skip_csv:
            append_result_csv(dataset_root / "results/baseline_results.csv", csv_row)
        log(json.dumps({"output": str(output_path), "test": test_result, "runtime_seconds": runtime}))


if __name__ == "__main__":
    main()
