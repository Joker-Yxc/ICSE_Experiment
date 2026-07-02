#!/usr/bin/env python3
"""DEEPCAPA-adapted baseline for Windows API sequence classification."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


BASELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASELINE_ROOT.parents[1]
OFFICIAL_REPOSITORY = "https://github.com/ucsb-seclab/DeepCapa"
OFFICIAL_COMMIT = "f14f03ce3710c279d8c603d1451559e202fdc3ca"
DATASETS = {
    "quo_vadis": WORKSPACE_ROOT / "datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz",
    "zenodo_11079764": WORKSPACE_ROOT
    / "datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz",
}
CAPABILITY_NAMES = [
    "pad",
    "file",
    "registry",
    "process",
    "memory",
    "network",
    "service",
    "crypto",
    "system",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--fallback-max-length", type=int, default=512)
    parser.add_argument("--min-api-frequency", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-pool-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--cnn-filters", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--family-weight", type=float, default=0.15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Smoke-test limit only; 0 uses all rows in the existing splits.",
    )
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


def normalize_api(value: str) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower()) or "<unk>"


def capability_category(api_name: str) -> int:
    name = normalize_api(api_name)
    rules = (
        (1, ("file", "directory", "path", "drive", "volume", "section", "mapview")),
        (2, ("reg", "key", "valuekey")),
        (3, ("process", "thread", "token", "job", "debug", "snapshot")),
        (4, ("memory", "heap", "virtual", "alloc", "protect", "readprocess", "writeprocess")),
        (5, ("socket", "internet", "http", "dns", "connect", "send", "recv", "network", "url")),
        (6, ("service", "scmanager", "driver", "deviceio")),
        (7, ("crypt", "hash", "cert", "encrypt", "decrypt", "random")),
        (8, ("system", "computer", "environment", "time", "performance", "library", "module")),
    )
    for category, keywords in rules:
        if any(keyword in name for keyword in keywords):
            return category
    return 9


def load_rows(path: Path, limit_per_split: int) -> dict[str, list[dict]]:
    rows = {"train": [], "val": [], "test": []}
    total_labels: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            required = {"sample_id", "source", "label", "family", "api_seq", "split"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            split = str(row["split"])
            if split not in rows:
                raise ValueError(f"{path}:{line_number} unexpected split {split!r}")
            label = str(row["label"]).lower()
            if label not in {"benign", "malware"}:
                raise ValueError(f"{path}:{line_number} unexpected label {label!r}")
            if not isinstance(row["api_seq"], list) or not row["api_seq"]:
                raise ValueError(f"{path}:{line_number} api_seq must be a non-empty list")
            total_labels[label] += 1
            if not limit_per_split or len(rows[split]) < limit_per_split:
                rows[split].append(row)

    expected = {"train": 35000, "val": 7500, "test": 7500}
    counts = {split: len(items) for split, items in rows.items()}
    if not limit_per_split and counts != expected:
        raise ValueError(f"Expected existing split counts {expected}, found {counts}")
    if total_labels != Counter({"benign": 25000, "malware": 25000}):
        raise ValueError(f"Expected balanced 50k labels, found {dict(total_labels)}")
    if any(not items for items in rows.values()):
        raise ValueError("Every existing split must contain samples")
    return rows


def build_vocabulary(
    train_rows: list[dict], min_frequency: int
) -> tuple[dict[str, int], list[str], np.ndarray]:
    counts = Counter(normalize_api(api) for row in train_rows for api in row["api_seq"])
    tokens = [token for token, count in counts.most_common() if count >= min_frequency]
    id_to_api = ["<pad>", "<unk>"] + tokens
    api_to_id = {token: index for index, token in enumerate(id_to_api)}
    categories = np.array(
        [0, capability_category("<unk>")] + [capability_category(token) for token in tokens],
        dtype=np.int64,
    )
    return api_to_id, id_to_api, categories


def encode_rows(
    rows: list[dict],
    api_to_id: dict[str, int],
    family_to_id: dict[str, int],
    max_length: int,
) -> dict[str, np.ndarray]:
    x = np.zeros((len(rows), max_length), dtype=np.int32)
    lengths = np.zeros(len(rows), dtype=np.int32)
    y = np.zeros(len(rows), dtype=np.float32)
    y_family = np.full(len(rows), -1, dtype=np.int64)
    sample_ids = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        token_ids = [
            api_to_id.get(normalize_api(api), 1) for api in row["api_seq"][:max_length]
        ]
        x[index, : len(token_ids)] = token_ids
        lengths[index] = len(token_ids)
        malware = str(row["label"]).lower() == "malware"
        y[index] = float(malware)
        if malware:
            y_family[index] = family_to_id.get(str(row["family"]).lower(), -1)
        sample_ids[index] = str(row["sample_id"])
    return {
        "x": x,
        "lengths": lengths,
        "y": y,
        "y_family": y_family,
        "sample_id": sample_ids,
    }


def prepare_cache(
    args: argparse.Namespace, data_path: Path, cache_dir: Path, max_length: int, log
) -> tuple[dict[str, dict[str, np.ndarray]], dict, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    vocabulary_path = cache_dir / "vocabulary.json"
    split_paths = {split: cache_dir / f"{split}.npz" for split in ("train", "val", "test")}
    expected = {
        "data_path": str(data_path.resolve()),
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "seed": args.seed,
        "max_length": max_length,
        "min_api_frequency": args.min_api_frequency,
        "limit_per_split": args.limit_per_split,
    }
    if (
        not args.rebuild_cache
        and manifest_path.exists()
        and vocabulary_path.exists()
        and all(path.exists() for path in split_paths.values())
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            log(f"Loading cached arrays from {cache_dir}")
            arrays = {
                split: dict(np.load(path, allow_pickle=True))
                for split, path in split_paths.items()
            }
            vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
            return arrays, manifest, np.asarray(vocabulary["api_categories"], dtype=np.int64)

    log(f"Building train-only vocabulary and encoded arrays at max_length={max_length}")
    rows = load_rows(data_path, args.limit_per_split)
    api_to_id, id_to_api, api_categories = build_vocabulary(
        rows["train"], args.min_api_frequency
    )
    families = sorted(
        {
            str(row["family"]).lower()
            for row in rows["train"]
            if str(row["label"]).lower() == "malware"
        }
    )
    family_to_id = {family: index for index, family in enumerate(families)}
    arrays = {}
    for split, split_rows in rows.items():
        arrays[split] = encode_rows(split_rows, api_to_id, family_to_id, max_length)
        np.savez_compressed(split_paths[split], **arrays[split])

    vocabulary_path.write_text(
        json.dumps(
            {
                "id_to_api": id_to_api,
                "family_to_id": family_to_id,
                "capability_names": CAPABILITY_NAMES,
                "api_categories": api_categories.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        **expected,
        "split_policy": "existing split field used verbatim; no resplitting",
        "split_counts": {split: len(items) for split, items in rows.items()},
        "vocab_size": len(id_to_api),
        "family_count": len(family_to_id),
        "capability_categories": CAPABILITY_NAMES[1:],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return arrays, manifest, api_categories


class SequenceArrays(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.x = arrays["x"]
        self.lengths = arrays["lengths"]
        self.y = arrays["y"]
        self.y_family = arrays["y_family"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, float, int]:
        return self.x[index], int(self.lengths[index]), float(self.y[index]), int(
            self.y_family[index]
        )


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle large buckets, then batch similarly sized sequences together."""

    def __init__(
        self, lengths: np.ndarray, batch_size: int, shuffle: bool, seed: int
    ) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        if not self.shuffle:
            indices = np.argsort(self.lengths, kind="stable")
        else:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(len(self.lengths))
            bucket_size = self.batch_size * 20
            buckets = []
            for start in range(0, len(indices), bucket_size):
                bucket = indices[start : start + bucket_size]
                buckets.append(bucket[np.argsort(self.lengths[bucket], kind="stable")])
            rng.shuffle(buckets)
            indices = np.concatenate(buckets)
            self.epoch += 1
        batches = [
            indices[start : start + self.batch_size].tolist()
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def dynamic_collate(
    batch: list[tuple[np.ndarray, int, float, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(item[1] for item in batch)
    x = np.stack([item[0][:max_length] for item in batch])
    y = np.asarray([item[2] for item in batch], dtype=np.float32)
    y_family = np.asarray([item[3] for item in batch], dtype=np.int64)
    return torch.from_numpy(x).long(), torch.from_numpy(y), torch.from_numpy(y_family)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[:, : x.shape[1]])


class DeepCapaAdapted(nn.Module):
    """Lightweight DeepCapa-style Transformer, attention, CNN, and adapted heads."""

    def __init__(
        self,
        vocab_size: int,
        api_categories: np.ndarray,
        family_count: int,
        max_length: int,
        embedding_dim: int,
        attention_heads: int,
        attention_pool_heads: int,
        ff_dim: int,
        transformer_layers: int,
        cnn_filters: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.api_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.capability_embedding = nn.Embedding(
            len(CAPABILITY_NAMES), embedding_dim, padding_idx=0
        )
        self.register_buffer(
            "api_categories", torch.from_numpy(api_categories).long(), persistent=True
        )
        self.position = PositionalEncoding(embedding_dim, max_length, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.attention_scores = nn.Linear(embedding_dim, attention_pool_heads)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, cnn_filters, kernel_size=kernel, padding=kernel // 2)
                for kernel in (3, 5, 7)
            ]
        )
        feature_dim = embedding_dim * attention_pool_heads + cnn_filters * 3
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(128, 1)
        self.family_head = nn.Linear(128, family_count) if family_count else None

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        padding_mask = token_ids.eq(0)
        categories = self.api_categories[token_ids]
        encoded = self.api_embedding(token_ids) + self.capability_embedding(categories)
        encoded = self.position(encoded)
        encoded = self.transformer(encoded, src_key_padding_mask=padding_mask)

        attention_logits = self.attention_scores(encoded).masked_fill(
            padding_mask.unsqueeze(-1), -1e4
        )
        attention = torch.softmax(attention_logits, dim=1)
        attention_features = torch.einsum("blh,bld->bhd", attention, encoded).flatten(1)

        convolution_input = encoded.transpose(1, 2)
        cnn_features = []
        for convolution in self.convolutions:
            values = F.gelu(convolution(convolution_input))
            values = values.masked_fill(padding_mask.unsqueeze(1), -1e4)
            cnn_features.append(values.amax(dim=2))
        features = self.projection(torch.cat([attention_features, *cnn_features], dim=1))
        binary_logits = self.binary_head(features).squeeze(-1)
        family_logits = self.family_head(features) if self.family_head is not None else None
        return binary_logits, family_logits, attention


def make_model(
    args: argparse.Namespace,
    manifest: dict,
    api_categories: np.ndarray,
    max_length: int,
) -> DeepCapaAdapted:
    return DeepCapaAdapted(
        vocab_size=int(manifest["vocab_size"]),
        api_categories=api_categories,
        family_count=int(manifest["family_count"]),
        max_length=max_length,
        embedding_dim=args.embedding_dim,
        attention_heads=args.attention_heads,
        attention_pool_heads=args.attention_pool_heads,
        ff_dim=args.ff_dim,
        transformer_layers=args.transformer_layers,
        cnn_filters=args.cnn_filters,
        dropout=args.dropout,
    )


def probe_memory(
    args: argparse.Namespace,
    manifest: dict,
    api_categories: np.ndarray,
    max_length: int,
    device: torch.device,
    log,
) -> bool:
    if device.type == "cpu" or max_length <= args.fallback_max_length:
        return True
    probe_batch = min(args.batch_size, 16)
    model = make_model(args, manifest, api_categories, max_length).to(device)
    try:
        model.train()
        x = torch.randint(
            1,
            int(manifest["vocab_size"]),
            (probe_batch, max_length),
            device=device,
        )
        binary, family, _ = model(x)
        loss = binary.square().mean()
        if family is not None:
            loss = loss + family.square().mean()
        loss.backward()
        del x, binary, family, loss, model
        if device.type == "mps":
            torch.mps.empty_cache()
        log(f"Memory probe passed: batch={probe_batch}, max_length={max_length}")
        return True
    except RuntimeError as exc:
        message = str(exc).lower()
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        if "memory" in message or "alloc" in message or "mps" in message:
            log(f"Memory probe failed at max_length={max_length}: {exc}")
            return False
        raise


def make_loader(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int = 7,
) -> DataLoader:
    sampler = LengthBucketBatchSampler(arrays["lengths"], batch_size, shuffle, seed)
    return DataLoader(
        SequenceArrays(arrays),
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dynamic_collate,
    )


@torch.inference_mode()
def predict(
    model: DeepCapaAdapted, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels, families, family_scores = [], [], [], []
    for x, y, y_family in loader:
        binary_logits, family_logits, _ = model(x.to(device))
        probabilities.append(torch.sigmoid(binary_logits).cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
        families.append(y_family.numpy())
        if family_logits is not None:
            family_scores.append(torch.softmax(family_logits, dim=1).cpu().numpy())
    all_labels = np.concatenate(labels)
    scores = np.concatenate(family_scores) if family_scores else np.empty((len(all_labels), 0))
    return np.concatenate(probabilities), all_labels, np.concatenate(families), scores


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = probability >= threshold
        f1 = precision_recall_fscore_support(
            y_true, prediction, average="binary", zero_division=0
        )[2]
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, probability)),
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
    auc_values = []
    for class_id in range(scores.shape[1]):
        binary_truth = truth == class_id
        if binary_truth.any() and (~binary_truth).any():
            auc_values.append(roc_auc_score(binary_truth, scores[mask, class_id]))
    return {
        "sample_count": int(mask.sum()),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_auc_ovr": float(np.mean(auc_values)) if auc_values else None,
        "auc_class_count": len(auc_values),
    }


def train_model(
    args: argparse.Namespace,
    arrays: dict[str, dict[str, np.ndarray]],
    manifest: dict,
    api_categories: np.ndarray,
    max_length: int,
    device: torch.device,
    checkpoint_path: Path,
    log,
) -> tuple[DeepCapaAdapted, int, float, list[dict]]:
    train_loader = make_loader(
        arrays["train"], args.batch_size, True, args.num_workers, args.seed
    )
    val_loader = make_loader(
        arrays["val"], args.batch_size * 2, False, args.num_workers, args.seed
    )
    model = make_model(args, manifest, api_categories, max_length).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_epoch, best_f1, stale = 0, -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y, y_family in train_loader:
            x, y, y_family = x.to(device), y.to(device), y_family.to(device)
            optimizer.zero_grad(set_to_none=True)
            binary_logits, family_logits, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(binary_logits, y)
            family_mask = y_family >= 0
            if family_logits is not None and family_mask.any():
                loss = loss + args.family_weight * F.cross_entropy(
                    family_logits[family_mask], y_family[family_mask]
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(x)

        val_probability, val_y, _, _ = predict(model, val_loader, device)
        threshold = choose_threshold(val_y, val_probability)
        metrics = binary_metrics(val_y, val_probability, threshold)
        entry = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader.dataset),
            "threshold": threshold,
            **metrics,
        }
        history.append(entry)
        log(json.dumps(entry, sort_keys=True))
        if metrics["f1"] > best_f1 + 1e-6:
            best_epoch, best_f1, stale = epoch, metrics["f1"], 0
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "best_val_f1": best_f1,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                log(f"Early stopping after epoch {epoch}; best epoch={best_epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, best_epoch, best_f1, history


def upsert_result_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or list(row)
            existing = [item for item in reader if item.get("method") != row["method"]]
    else:
        fieldnames = list(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_path = args.data_path or DATASETS[args.dataset]
    dataset_root = data_path.parents[1]
    output_path = args.output_path or dataset_root / "results/deepcapa_adapted_metrics.json"
    root_cache_dir = (
        args.cache_dir
        or WORKSPACE_ROOT / "datasets_50k" / args.dataset / "artifacts" / "deepcapa_adapted"
    )
    log_path = args.log_path or BASELINE_ROOT / "logs" / f"{args.dataset}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        def log(message: str) -> None:
            print(message, flush=True)
            print(message, file=log_handle, flush=True)

        started = time.time()
        device = get_device(args.device)
        max_length = args.max_length
        cache_dir = root_cache_dir / f"len_{max_length}"
        log(f"dataset={args.dataset} data={data_path} device={device}")
        log("split_policy=use existing split field verbatim; never resplit")
        arrays, manifest, api_categories = prepare_cache(
            args, data_path, cache_dir, max_length, log
        )
        if not probe_memory(args, manifest, api_categories, max_length, device, log):
            max_length = args.fallback_max_length
            cache_dir = root_cache_dir / f"len_{max_length}"
            arrays, manifest, api_categories = prepare_cache(
                args, data_path, cache_dir, max_length, log
            )
            log(f"Fell back to max_length={max_length}")

        checkpoint_path = cache_dir / "deepcapa_adapted_best.pt"
        log(
            f"split_counts={manifest['split_counts']} vocab={manifest['vocab_size']} "
            f"families={manifest['family_count']} max_length={max_length}"
        )
        model, best_epoch, best_val_f1, history = train_model(
            args,
            arrays,
            manifest,
            api_categories,
            max_length,
            device,
            checkpoint_path,
            log,
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
            "method": "deepcapa_adapted",
            "status": "ok",
            "dataset": args.dataset,
            "seed": args.seed,
            "paper": "DEEPCAPA: Identifying Malicious Capabilities in Windows Malware",
            "venue": "ACSAC 2024",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_code_commit": OFFICIAL_COMMIT,
            "official_code_run": False,
            "official_code_reason": (
                "Official fine-tuning requires per-technique MITRE labels, its dataset layout, "
                "and a compatible pretraining checkpoint; these datasets contain API traces, "
                "binary labels, and family labels only."
            ),
            "implementation": "DeepCapa-style fallback reimplementation",
            "original_task": "malicious capability identification mapped to MITRE ATT&CK techniques",
            "adapted_task": "benign/malware detection with malware family classification",
            "input": "api_seq Windows API-call sequence",
            "split_policy": "used existing split field without resplitting",
            "model": (
                "API and capability embeddings + positional encoding + Transformer + "
                "multi-head attention pooling + multi-kernel CNN + adapted heads"
            ),
            "capability_labels_are_ground_truth": False,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "threshold": threshold,
            "val": val_result,
            "test": test_result,
            "family_val": family_metrics(val_family, val_family_scores),
            "family_test": family_metrics(test_family, test_family_scores),
            "runtime_seconds": runtime,
            "config": vars(args)
            | {
                "data_path": str(data_path),
                "cache_dir": str(cache_dir),
                "device_used": str(device),
                "max_length_used": max_length,
            },
            "cache_manifest": manifest,
            "training_history": history,
        }
        output_path.write_text(
            json.dumps(result, indent=2, default=str, allow_nan=False),
            encoding="utf-8",
        )
        csv_row = {
            "method": "deepcapa_adapted",
            "status": "ok",
            "epochs_run": len(history),
            "best_val_f1": best_val_f1,
            "threshold": threshold,
            "model": result["model"],
            "semantic_extractor": "capability-category heuristic",
            **{f"val_{key}": value for key, value in val_result.items()},
            **{f"test_{key}": value for key, value in test_result.items()},
            "runtime_seconds": runtime,
            "error": "",
        }
        if not args.skip_csv:
            upsert_result_csv(dataset_root / "results/baseline_results.csv", csv_row)
        log(json.dumps({"output": str(output_path), "test": test_result, "runtime": runtime}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DEEPCAPA-adapted failed: {exc}", file=sys.stderr)
        raise
