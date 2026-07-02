#!/usr/bin/env python3
"""Run GAT and R-GCN baselines on compact API-transition graphs."""

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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


MODELS = ("gat", "rgcn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default="gat,rgcn")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--limit-per-split", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
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
    if any(not values for values in rows.values()):
        raise ValueError(f"Every split must be non-empty: { {k: len(v) for k, v in rows.items()} }")
    return rows


def resource_type(api: str) -> int:
    text = api.lower()
    if any(key in text for key in ("file", "directory", "path", "read", "write")):
        return 0
    if any(key in text for key in ("reg", "key", "value")):
        return 1
    if any(key in text for key in ("process", "thread", "token")):
        return 2
    if any(key in text for key in ("socket", "connect", "send", "recv", "http", "dns")):
        return 3
    if any(key in text for key in ("alloc", "protect", "map", "heap", "library")):
        return 4
    return 5


class Vocabulary:
    def __init__(self, rows: list[dict], max_length: int):
        counts = Counter(
            str(call)
            for row in rows
            for call in row["api_seq"][:max_length]
        )
        self.to_id = {"<UNK>": 0}
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            self.to_id[token] = len(self.to_id)

    def encode(self, token: str) -> int:
        return self.to_id.get(str(token), 0)

    def __len__(self) -> int:
        return len(self.to_id)


def to_graph(row: dict, vocab: Vocabulary, max_length: int) -> dict:
    sequence = [str(call) for call in row["api_seq"][:max_length]]
    if not sequence:
        sequence = ["<UNK>"]
    nodes = list(dict.fromkeys(sequence))
    local = {node: index for index, node in enumerate(nodes)}
    transitions = Counter(zip(sequence[:-1], sequence[1:]))
    edges = []
    weights = []
    relations = []
    for (source, target), count in transitions.items():
        edges.append((local[source], local[target]))
        weights.append(np.log1p(count))
        source_type = resource_type(source)
        target_type = resource_type(target)
        relations.append(source_type * 6 + target_type)
    for node in nodes:
        edges.append((local[node], local[node]))
        weights.append(1.0)
        node_type = resource_type(node)
        relations.append(36 + node_type)
    return {
        "node_ids": torch.tensor([vocab.encode(node) for node in nodes], dtype=torch.long),
        "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous(),
        "edge_weight": torch.tensor(weights, dtype=torch.float32),
        "edge_type": torch.tensor(relations, dtype=torch.long),
        "label": int(row["label"] == "malware"),
        "sample_id": row["sample_id"],
        "family": row.get("family", "unknown"),
    }


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], vocab: Vocabulary, max_length: int):
        self.graphs = [to_graph(row, vocab, max_length) for row in rows]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def collate_graphs(graphs: list[dict]) -> dict:
    node_ids, edge_indices, weights, types, batch, labels = [], [], [], [], [], []
    sample_ids, families = [], []
    offset = 0
    for graph_index, graph in enumerate(graphs):
        node_count = len(graph["node_ids"])
        node_ids.append(graph["node_ids"])
        edge_indices.append(graph["edge_index"] + offset)
        weights.append(graph["edge_weight"])
        types.append(graph["edge_type"])
        batch.append(torch.full((node_count,), graph_index, dtype=torch.long))
        labels.append(graph["label"])
        sample_ids.append(graph["sample_id"])
        families.append(graph["family"])
        offset += node_count
    return {
        "node_ids": torch.cat(node_ids),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_weight": torch.cat(weights),
        "edge_type": torch.cat(types),
        "batch": torch.cat(batch),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "sample_ids": sample_ids,
        "families": families,
    }


def segment_softmax(scores: torch.Tensor, destinations: torch.Tensor, node_count: int) -> torch.Tensor:
    maximum = torch.full(
        (node_count, scores.size(1)), -torch.inf, device=scores.device
    )
    maximum.scatter_reduce_(
        0,
        destinations[:, None].expand_as(scores),
        scores,
        reduce="amax",
        include_self=True,
    )
    exponent = torch.exp(scores - maximum[destinations])
    denominator = torch.zeros(
        node_count, scores.size(1), device=scores.device
    ).index_add_(0, destinations, exponent)
    return exponent / denominator[destinations].clamp_min(1e-12)


class GATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, heads * output_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(heads, output_dim))
        self.target_attention = nn.Parameter(torch.empty(heads, output_dim))
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, features, edge_index, edge_weight):
        node_count = features.size(0)
        projected = self.projection(features).view(node_count, self.heads, self.output_dim)
        source, target = edge_index
        scores = (
            (projected[source] * self.source_attention).sum(-1)
            + (projected[target] * self.target_attention).sum(-1)
            + edge_weight[:, None]
        )
        attention = segment_softmax(F.leaky_relu(scores, 0.2), target, node_count)
        output = torch.zeros(
            node_count, self.heads, self.output_dim, device=features.device
        )
        output.index_add_(0, target, projected[source] * attention[:, :, None])
        return output.flatten(1)


class RGCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, relation_count: int):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(relation_count, input_dim, output_dim)
        )
        self.root = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, features, edge_index, edge_type, edge_weight):
        source, target = edge_index
        transformed = torch.bmm(
            features[source].unsqueeze(1), self.weights[edge_type]
        ).squeeze(1)
        transformed = transformed * edge_weight[:, None]
        output = torch.zeros(
            features.size(0), transformed.size(1), device=features.device
        )
        output.index_add_(0, target, transformed)
        degree = torch.zeros(features.size(0), device=features.device).index_add_(
            0, target, edge_weight
        )
        return output / degree[:, None].clamp_min(1.0) + self.root(features)


class GraphClassifier(nn.Module):
    def __init__(self, kind: str, vocab_size: int, embedding_dim: int, hidden_dim: int, heads: int):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if kind == "gat":
            if hidden_dim % heads:
                raise ValueError("hidden_dim must be divisible by heads")
            self.layer1 = GATLayer(embedding_dim, hidden_dim // heads, heads)
            self.layer2 = GATLayer(hidden_dim, hidden_dim // heads, heads)
        else:
            self.layer1 = RGCNLayer(embedding_dim, hidden_dim, 42)
            self.layer2 = RGCNLayer(hidden_dim, hidden_dim, 42)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        features = self.embedding(batch["node_ids"])
        if self.kind == "gat":
            features = F.elu(self.layer1(features, batch["edge_index"], batch["edge_weight"]))
            features = F.elu(self.layer2(features, batch["edge_index"], batch["edge_weight"]))
        else:
            features = F.relu(
                self.layer1(features, batch["edge_index"], batch["edge_type"], batch["edge_weight"])
            )
            features = F.relu(
                self.layer2(features, batch["edge_index"], batch["edge_type"], batch["edge_weight"])
            )
        graph_count = int(batch["batch"].max().item()) + 1
        pooled = torch.zeros(graph_count, features.size(1), device=features.device)
        pooled.index_add_(0, batch["batch"], features)
        counts = torch.bincount(batch["batch"], minlength=graph_count).to(features.device)
        pooled = pooled / counts[:, None].clamp_min(1)
        return self.classifier(pooled).squeeze(-1)


def move(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 301)))
    best = (-1.0, 0.5)
    for threshold in candidates:
        predictions = scores >= threshold
        f1 = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )[2]
        if f1 > best[0]:
            best = (float(f1), float(threshold))
    return best[1], best[0]


def metrics(labels, scores, threshold):
    predictions = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(labels, scores)),
        "threshold": float(threshold),
    }


def predict(model, loader, device):
    model.eval()
    scores, labels, records = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move(raw_batch, device)
            probabilities = torch.sigmoid(model(batch)).cpu().tolist()
            scores.extend(probabilities)
            labels.extend(raw_batch["labels"].long().tolist())
            records.extend(
                zip(raw_batch["sample_ids"], raw_batch["families"], labels[-len(probabilities):], probabilities)
            )
    return np.asarray(labels), np.asarray(scores), records


def save_predictions(path: Path, records, threshold):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for sample_id, family, label, score in records:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "family": family,
                        "label": int(label),
                        "score": float(score),
                        "prediction": int(score >= threshold),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def run_model(kind, rows, vocab, args, out_dir):
    set_seed(args.seed)
    device = get_device(args.device)
    datasets = {
        split: GraphDataset(values, vocab, args.max_length)
        for split, values in rows.items()
    }
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_graphs,
        )
        for split, dataset in datasets.items()
    }
    model = GraphClassifier(
        kind, len(vocab), args.embedding_dim, args.hidden_dim, args.heads
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_state, best_f1, stale, epochs_run = None, -1.0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for raw_batch in loaders["train"]:
            batch = move(raw_batch, device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_labels, val_scores, _ = predict(model, loaders["val"], device)
        _, val_f1 = choose_threshold(val_labels, val_scores)
        epochs_run = epoch
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    val_labels, val_scores, val_records = predict(model, loaders["val"], device)
    threshold, best_f1 = choose_threshold(val_labels, val_scores)
    test_labels, test_scores, test_records = predict(model, loaders["test"], device)
    run_dir = out_dir / kind / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "best_model.pt")
    save_predictions(run_dir / "validation_predictions.jsonl.gz", val_records, threshold)
    save_predictions(run_dir / "test_predictions.jsonl.gz", test_records, threshold)
    result = {
        "method": kind,
        "seed": args.seed,
        "data": args.data,
        "data_sha256": sha256_file(Path(args.data)),
        "split_counts": {split: len(values) for split, values in rows.items()},
        "max_length": args.max_length,
        "threshold": threshold,
        "best_validation_f1": best_f1,
        "validation": metrics(val_labels, val_scores, threshold),
        "test": metrics(test_labels, test_scores, threshold),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_count": len(vocab),
        "epochs_run": epochs_run,
        "runtime_seconds": time.perf_counter() - started,
        "config": vars(args),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main():
    args = parse_args()
    requested = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unknown = sorted(set(requested) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    rows = load_rows(Path(args.data), args.limit_per_split)
    vocab = Vocabulary(rows["train"], args.max_length)
    out_dir = Path(args.out_dir)
    results = []
    for kind in requested:
        print(f"[run] {kind} seed={args.seed}", flush=True)
        results.append(run_model(kind, rows, vocab, args, out_dir))
    (out_dir / f"graph_baselines_seed_{args.seed}.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
