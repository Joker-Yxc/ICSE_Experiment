#!/usr/bin/env python3
"""APILI-adapted malware detector for compact Windows API sequence datasets.

The official APILI task predicts MITRE ATT&CK techniques and uses attention to
locate relevant API calls in dynamic traces. This adaptation keeps frozen BERT
semantic representations plus API/resource attention, while replacing the
technique head with binary malware detection and an optional family head.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
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
from transformers import AutoModel, AutoTokenizer


BASELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASELINE_ROOT.parents[1]
OFFICIAL_ROOT = BASELINE_ROOT / "official_code"
OFFICIAL_COMMIT = "84a18757896347f032f49aec4dd1907bbae7bb07"
DATASETS = {
    "quo_vadis": WORKSPACE_ROOT / "datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz",
    "zenodo_11079764": WORKSPACE_ROOT
    / "datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--bert-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-api-frequency", type=int, default=2)
    parser.add_argument("--bert-batch-size", type=int, default=128)
    parser.add_argument("--bert-max-tokens", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--family-weight", type=float, default=0.15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Smoke-test limit only; 0 uses every row in the existing splits.",
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


def identifier_words(value: str) -> str:
    value = value.replace(".", " ")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value)
    return " ".join(value.lower().split())


def api_semantic_text(api: str) -> str:
    return f"Windows API call {identifier_words(api)}"


def resource_semantic_text(api: str) -> str:
    # No arguments/resources exist in the compact datasets. APILI's resource
    # branch is retained using the API name as the required simplified proxy.
    return f"Windows API resource accessed by {identifier_words(api)}"


def load_rows(path: Path, limit_per_split: int) -> dict[str, list[dict]]:
    rows = {"train": [], "val": [], "test": []}
    label_counts: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            missing = {"sample_id", "source", "label", "family", "api_seq", "split"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            split = str(row["split"])
            if split not in rows:
                raise ValueError(f"{path}:{line_number} unexpected split {split!r}")
            label = str(row["label"]).lower()
            if label not in {"benign", "malware"}:
                raise ValueError(f"{path}:{line_number} unexpected label {label!r}")
            if not isinstance(row["api_seq"], list):
                raise ValueError(f"{path}:{line_number} api_seq must be a list")
            if limit_per_split and len(rows[split]) >= limit_per_split:
                continue
            rows[split].append(row)
            label_counts[label] += 1

    expected = {"train": 35000, "val": 7500, "test": 7500}
    counts = {split: len(items) for split, items in rows.items()}
    if not limit_per_split and counts != expected:
        raise ValueError(f"Expected existing split counts {expected}, found {counts}")
    if not limit_per_split and label_counts != Counter({"benign": 25000, "malware": 25000}):
        raise ValueError(f"Expected balanced 50k labels, found {dict(label_counts)}")
    if any(not items for items in rows.values()):
        raise ValueError("Every existing split must contain samples")
    return rows


def build_vocabulary(train_rows: list[dict], min_frequency: int) -> tuple[dict[str, int], list[str]]:
    counts = Counter(normalize_api(api) for row in train_rows for api in row["api_seq"])
    tokens = [token for token, count in counts.most_common() if count >= min_frequency]
    id_to_api = ["<pad>", "<unk>"] + tokens
    return {token: index for index, token in enumerate(id_to_api)}, id_to_api


def encode_rows(
    rows: list[dict],
    api_to_id: dict[str, int],
    family_to_id: dict[str, int],
    max_length: int,
) -> dict[str, np.ndarray]:
    x = np.zeros((len(rows), max_length), dtype=np.int32)
    y = np.zeros(len(rows), dtype=np.float32)
    y_family = np.full(len(rows), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        tokens = [api_to_id.get(normalize_api(api), 1) for api in row["api_seq"][:max_length]]
        x[index, : len(tokens)] = tokens
        is_malware = str(row["label"]).lower() == "malware"
        y[index] = float(is_malware)
        if is_malware:
            y_family[index] = family_to_id.get(str(row["family"]).lower(), -1)
    return {"x": x, "y": y, "y_family": y_family}


@torch.inference_mode()
def bert_encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_tokens: int,
    device: torch.device,
    log,
) -> np.ndarray:
    log(f"Loading frozen BERT semantic encoder: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as exc:
        log(f"Remote BERT lookup failed ({type(exc).__name__}); using local cache")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model = model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        chunks.append(pooled.float().cpu().numpy())
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return np.concatenate(chunks).astype(np.float32)


def hashed_unknown_vector(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        vector[value % dim] += 1.0 if value & 1 else -1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def prepare_cache(
    args: argparse.Namespace,
    data_path: Path,
    cache_dir: Path,
    device: torch.device,
    log,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    split_paths = {split: cache_dir / f"{split}.npz" for split in ("train", "val", "test")}
    semantic_path = cache_dir / "bert_semantics.npz"
    expected = {
        "data_path": str(data_path.resolve()),
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "seed": args.seed,
        "max_length": args.max_length,
        "min_api_frequency": args.min_api_frequency,
        "bert_model": args.bert_model,
        "bert_max_tokens": args.bert_max_tokens,
        "limit_per_split": args.limit_per_split,
    }
    if (
        not args.rebuild_cache
        and manifest_path.exists()
        and semantic_path.exists()
        and all(path.exists() for path in split_paths.values())
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            arrays = {split: dict(np.load(path)) for split, path in split_paths.items()}
            return arrays, manifest

    rows = load_rows(data_path, args.limit_per_split)
    api_to_id, id_to_api = build_vocabulary(rows["train"], args.min_api_frequency)
    train_families = sorted(
        {
            str(row["family"]).lower()
            for row in rows["train"]
            if str(row["label"]).lower() == "malware" and str(row["family"]).strip()
        }
    )
    family_to_id = {family: index for index, family in enumerate(train_families)}
    arrays = {}
    for split, split_rows in rows.items():
        arrays[split] = encode_rows(split_rows, api_to_id, family_to_id, args.max_length)
        np.savez_compressed(split_paths[split], **arrays[split])

    actual_apis = id_to_api[2:]
    api_vectors = bert_encode_texts(
        [api_semantic_text(api) for api in actual_apis],
        args.bert_model,
        args.bert_batch_size,
        args.bert_max_tokens,
        device,
        log,
    )
    resource_vectors = bert_encode_texts(
        [resource_semantic_text(api) for api in actual_apis],
        args.bert_model,
        args.bert_batch_size,
        args.bert_max_tokens,
        device,
        log,
    )
    dim = api_vectors.shape[1]
    api_semantics = np.zeros((len(id_to_api), dim), dtype=np.float32)
    resource_semantics = np.zeros_like(api_semantics)
    api_semantics[1] = hashed_unknown_vector("unknown Windows API call", dim)
    resource_semantics[1] = hashed_unknown_vector("unknown Windows API resource", dim)
    api_semantics[2:] = api_vectors
    resource_semantics[2:] = resource_vectors
    np.savez_compressed(
        semantic_path, api_semantics=api_semantics, resource_semantics=resource_semantics
    )
    (cache_dir / "vocabulary.json").write_text(
        json.dumps(
            {"id_to_api": id_to_api, "family_to_id": family_to_id},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        **expected,
        "split_policy": "existing split field used verbatim; no resplitting",
        "split_counts": {split: len(items) for split, items in rows.items()},
        "vocab_size": len(id_to_api),
        "family_count": len(family_to_id),
        "semantic_dim": dim,
        "resource_fallback": "API name converted to a simplified resource sentence",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return arrays, manifest


class APILIAdapted(nn.Module):
    """BERT semantics, resource attention, API attention, and adapted heads."""

    def __init__(
        self,
        api_semantics: np.ndarray,
        resource_semantics: np.ndarray,
        hidden_dim: int,
        family_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        vocab_size, semantic_dim = api_semantics.shape
        self.api_embedding = nn.Embedding(vocab_size, semantic_dim, padding_idx=0)
        self.resource_embedding = nn.Embedding(vocab_size, semantic_dim, padding_idx=0)
        self.api_embedding.weight.data.copy_(torch.from_numpy(api_semantics))
        self.resource_embedding.weight.data.copy_(torch.from_numpy(resource_semantics))
        self.api_embedding.weight.requires_grad_(False)
        self.resource_embedding.weight.requires_grad_(False)

        self.api_projection = nn.Linear(semantic_dim, hidden_dim)
        self.resource_projection = nn.Linear(semantic_dim, hidden_dim)
        self.resource_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.api_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_dim * 2, 1)
        self.family_head = nn.Linear(hidden_dim * 2, family_count) if family_count else None

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        mask = token_ids.ne(0)
        api = torch.tanh(self.api_projection(self.api_embedding(token_ids)))
        resource = torch.tanh(self.resource_projection(self.resource_embedding(token_ids)))
        resource_weights = torch.sigmoid(self.resource_attention(torch.cat([api, resource], dim=-1)))
        fused = api + resource_weights * resource
        sequence, _ = self.encoder(fused)
        api_scores = self.api_attention(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        api_weights = torch.softmax(api_scores, dim=1)
        context = torch.sum(sequence * api_weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)
        binary_logits = self.binary_head(context).squeeze(-1)
        family_logits = self.family_head(context) if self.family_head is not None else None
        return binary_logits, family_logits, api_weights, resource_weights.squeeze(-1)


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


@torch.inference_mode()
def predict(
    model: APILIAdapted, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels, families, family_scores = [], [], [], []
    for x, y, y_family in loader:
        logits, family_logits, _, _ = model(x.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
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
            best_f1, best_threshold = float(f1), float(threshold)
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
    arrays: dict,
    api_semantics: np.ndarray,
    resource_semantics: np.ndarray,
    family_count: int,
    device: torch.device,
    log,
) -> tuple[APILIAdapted, int, float, list[dict]]:
    train_loader = make_loader(arrays["train"], args.batch_size, True, args.num_workers)
    val_loader = make_loader(arrays["val"], args.batch_size * 2, False, args.num_workers)
    model = APILIAdapted(
        api_semantics,
        resource_semantics,
        args.hidden_dim,
        family_count,
        args.dropout,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state, best_epoch, best_f1, stale = None, 0, -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y, y_family in train_loader:
            x, y, y_family = x.to(device), y.to(device), y_family.to(device)
            optimizer.zero_grad(set_to_none=True)
            binary_logits, family_logits, _, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(binary_logits, y)
            family_mask = y_family >= 0
            if family_logits is not None and family_mask.any():
                loss = loss + args.family_weight * F.cross_entropy(
                    family_logits[family_mask], y_family[family_mask]
                )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
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
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, best_f1, history


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
    data_path = args.data_path or DATASETS[args.dataset]
    dataset_root = data_path.parents[1]
    output_path = args.output_path or dataset_root / "results/apili_adapted_metrics.json"
    cache_dir = (
        args.cache_dir
        or WORKSPACE_ROOT / "datasets_50k" / args.dataset / "artifacts" / "apili_adapted"
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
        log(f"dataset={args.dataset} data={data_path} device={device}")
        log("split_policy=use existing split field verbatim; never resplit")
        arrays, manifest = prepare_cache(args, data_path, cache_dir, device, log)
        semantics = np.load(cache_dir / "bert_semantics.npz")
        log(
            f"split_counts={manifest['split_counts']} vocab={manifest['vocab_size']} "
            f"families={manifest['family_count']}"
        )

        model, best_epoch, best_val_f1, history = train_model(
            args,
            arrays,
            semantics["api_semantics"],
            semantics["resource_semantics"],
            int(manifest["family_count"]),
            device,
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
            "method": "apili_adapted",
            "status": "ok",
            "dataset": args.dataset,
            "seed": args.seed,
            "original_task": "MITRE ATT&CK technique prediction and attention-based API locating",
            "adapted_task": "benign/malware detection with optional malware family classification",
            "split_policy": "used existing split field without resplitting",
            "input": "api_seq",
            "official_repository": (
                "https://github.com/Irish-kw/"
                "Attention-Based-API-Locating-for-Malware-Techniques"
            ),
            "official_code_commit": OFFICIAL_COMMIT,
            "official_training_code_in_clone": False,
            "model": "APILI-inspired frozen-BERT BiGRU with API/resource attention",
            "semantic_extractor": args.bert_model,
            "resource_representation": (
                "simplified API-name resource sentence because resource/argument labels are absent"
            ),
            "binary_head": True,
            "family_head": bool(manifest["family_count"]),
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
            },
            "cache_manifest": manifest,
            "training_history": history,
        }
        output_path.write_text(
            json.dumps(result, indent=2, default=str, allow_nan=False),
            encoding="utf-8",
        )
        torch.save(
            {"state_dict": model.state_dict(), "config": result["config"]},
            cache_dir / "apili_adapted_best.pt",
        )

        csv_row = {
            "method": "apili_adapted",
            "status": "ok",
            "epochs_run": len(history),
            "best_val_f1": best_val_f1,
            "threshold": threshold,
            "model": result["model"],
            "semantic_extractor": args.bert_model,
            **{f"val_{key}": value for key, value in val_result.items()},
            **{f"test_{key}": value for key, value in test_result.items()},
            "runtime_seconds": runtime,
            "error": "",
        }
        if not args.skip_csv:
            append_result_csv(dataset_root / "results/baseline_results.csv", csv_row)
        log(json.dumps({"output": str(output_path), "test": test_result, "runtime": runtime}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"APILI-adapted failed: {exc}", file=sys.stderr)
        raise
