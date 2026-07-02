#!/usr/bin/env python3
"""GPT-4 API-description + BERT/sentence-transformer embedding + CNN baseline.

The public official repository for "Prompt Engineering-assisted Malware Dynamic
Analysis Using GPT-4" only exposes the paper README. This script implements the
paper-described pipeline against the local compact 50k API sequence dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - exercised in dependency-poor envs.
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz"
DEFAULT_METRICS = ROOT / "datasets_50k/quo_vadis/results/gpt4_api_bert_cnn_metrics.json"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass
class Sample:
    sample_id: str
    label: str
    family: str
    api_seq: List[str]
    split: str


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_samples(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            samples.append(
                Sample(
                    sample_id=str(obj.get("sample_id", "")),
                    label=str(obj["label"]),
                    family=str(obj.get("family") or obj["label"]),
                    api_seq=list(obj.get("api_seq") or []),
                    split=str(obj["split"]),
                )
            )
    return samples


def validate_dataset(samples: Sequence[Sample]) -> None:
    split_counts = Counter(s.split for s in samples)
    label_counts = Counter(s.label for s in samples)
    logging.info("Dataset total=%d labels=%s splits=%s", len(samples), dict(label_counts), dict(split_counts))
    expected_splits = {"train": 35000, "val": 7500, "test": 7500}
    if len(samples) != 50000 or dict(split_counts) != expected_splits:
        logging.warning("Dataset does not match expected 50k split counts: %s", expected_splits)
    if label_counts.get("benign") != 25000 or label_counts.get("malware") != 25000:
        logging.warning("Dataset does not match expected benign/malware balance.")


def normalize_api(api_name: str) -> Tuple[str, str]:
    if "." in api_name:
        module, func = api_name.rsplit(".", 1)
    else:
        module, func = "Windows", api_name
    return module.strip(), func.strip()


def split_identifier(name: str) -> List[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"[^A-Za-z0-9]+", " ", parts)
    return [p.lower() for p in parts.split() if p]


def rule_template_description(api_name: str) -> str:
    module, func = normalize_api(api_name)
    words = split_identifier(func)
    text = " ".join(words) if words else func
    lower = set(words)
    hints = []
    categories = [
        ({"file", "directory", "path", "write", "read", "create", "delete", "copy", "move"}, "file-system access"),
        ({"reg", "registry", "key", "value"}, "registry manipulation"),
        ({"process", "thread", "module", "library", "load", "proc", "dll"}, "process or library management"),
        ({"socket", "connect", "send", "recv", "http", "url", "internet", "dns"}, "network communication"),
        ({"crypt", "hash", "cert", "encrypt", "decrypt"}, "cryptographic operation"),
        ({"service", "scm", "driver"}, "service or driver control"),
        ({"memory", "heap", "virtual", "alloc", "free", "protect"}, "memory management"),
        ({"window", "keyboard", "mouse", "hook", "message"}, "user-interface interaction"),
        ({"time", "tick", "performance", "sleep"}, "timing or synchronization"),
        ({"token", "privilege", "security", "acl", "sid"}, "security context management"),
    ]
    for keys, label in categories:
        if lower & keys:
            hints.append(label)
    if not hints:
        hints.append("general Windows runtime behavior")
    return (
        f"Windows API {api_name} from module {module}. "
        f"The function name indicates {text}. "
        f"It is commonly associated with {', '.join(dict.fromkeys(hints))}."
    )


def load_or_create_descriptions(
    train_apis: Iterable[str],
    cache_path: Path,
    description_source: str,
) -> Dict[str, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            descriptions = json.load(handle)
    else:
        descriptions = {}

    train_apis = sorted(set(train_apis))
    missing = [api for api in train_apis if api not in descriptions]
    logging.info("API descriptions cache=%s existing=%d missing=%d", cache_path, len(descriptions), len(missing))

    if missing and description_source in {"gpt4", "openai"}:
        descriptions.update(generate_openai_descriptions(missing))
        missing = [api for api in train_apis if api not in descriptions]

    if missing and description_source == "deepseek":
        descriptions.update(
            generate_deepseek_descriptions(missing, cache_path, descriptions)
        )
        missing = [api for api in train_apis if api not in descriptions]
        if missing:
            raise RuntimeError(
                f"DeepSeek did not return valid descriptions for {len(missing)} APIs. "
                "The successful batches were cached; rerun with a smaller "
                "DEEPSEEK_DESCRIPTION_BATCH_SIZE."
            )

    if missing and description_source in {"local-llm", "local"}:
        descriptions.update(generate_local_llm_descriptions(missing))
        missing = [api for api in train_apis if api not in descriptions]

    if missing:
        logging.info("Using deterministic rule templates for %d API descriptions.", len(missing))
        for api in missing:
            descriptions[api] = rule_template_description(api)

    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(descriptions, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return {api: descriptions[api] for api in train_apis}


def generate_openai_descriptions(api_names: Sequence[str]) -> Dict[str, str]:
    """Optional GPT-4 path. Falls back silently if dependencies or credentials are absent."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY not set; cannot call GPT-4 for API descriptions.")
        return {}
    try:
        from openai import OpenAI
    except Exception as exc:
        logging.warning("openai package unavailable: %s", exc)
        return {}

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    out: Dict[str, str] = {}
    for api in api_names:
        prompt = (
            "Briefly explain the Windows API call for malware dynamic analysis. "
            "Do not infer benign or malicious labels. API: " + api
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            logging.warning("GPT description failed for %s: %s", api, exc)
            break
        out[api] = text.strip() or rule_template_description(api)
    return out


def generate_deepseek_descriptions(
    api_names: Sequence[str],
    checkpoint_path: Path,
    existing_descriptions: Dict[str, str],
) -> Dict[str, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logging.warning("DEEPSEEK_API_KEY not set; cannot call DeepSeek.")
        return {}
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    batch_size = int(os.environ.get("DEEPSEEK_DESCRIPTION_BATCH_SIZE", "10"))
    workers = int(os.environ.get("DEEPSEEK_DESCRIPTION_WORKERS", "4"))
    out: Dict[str, str] = {}

    def request_batch(start: int) -> Tuple[int, List[str], Dict[str, str]]:
        batch = list(api_names[start : start + batch_size])
        prompt = (
            "For every Windows API name in the JSON array below, write one concise "
            "English sentence of at most 18 words describing its behavior for dynamic "
            "malware analysis. "
            "Do not infer benign/malicious labels. Return only one valid JSON object "
            "whose keys exactly match the input API names and whose values are the "
            f"descriptions.\n\nAPI names:\n{json.dumps(batch, ensure_ascii=False)}"
        )
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max(8000, batch_size * 120),
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        parsed: Dict[str, str] = {}
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(
                    base_url + "/chat/completions",
                    data=payload,
                    headers={
                        "Authorization": "Bearer " + api_key,
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.load(response)
                text = body["choices"][0]["message"]["content"].strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
                candidate = json.loads(text)
                parsed = {
                    api: str(candidate[api]).strip()
                    for api in batch
                    if api in candidate and str(candidate[api]).strip()
                }
                break
            except (urllib.error.URLError, KeyError, TypeError, json.JSONDecodeError) as exc:
                logging.warning(
                    "DeepSeek batch %d-%d attempt=%d failed: %s",
                    start,
                    start + len(batch),
                    attempt,
                    exc,
                )
                time.sleep(attempt * 2)
        return start, batch, parsed

    starts = list(range(0, len(api_names), batch_size))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(request_batch, start) for start in starts]
        for future in concurrent.futures.as_completed(futures):
            start, batch, parsed = future.result()
            out.update(parsed)
            checkpoint = dict(existing_descriptions)
            checkpoint.update(out)
            with checkpoint_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    checkpoint,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            logging.info(
                "DeepSeek descriptions generated=%d/%d batch=%d-%d",
                len(out),
                len(api_names),
                start,
                start + len(batch),
            )
    return out


def generate_local_llm_descriptions(api_names: Sequence[str]) -> Dict[str, str]:
    """Optional local Qwen/DeepSeek/Llama path through transformers text-generation."""
    model_name = os.environ.get("LOCAL_LLM_MODEL")
    if not model_name:
        logging.warning("LOCAL_LLM_MODEL not set; cannot use local LLM descriptions.")
        return {}
    try:
        from transformers import pipeline
    except Exception as exc:
        logging.warning("transformers unavailable for local LLM descriptions: %s", exc)
        return {}
    try:
        generator = pipeline("text-generation", model=model_name, device_map="auto")
    except Exception as exc:
        logging.warning("Failed to load local LLM %s: %s", model_name, exc)
        return {}
    out: Dict[str, str] = {}
    for api in api_names:
        prompt = f"Explain this Windows API call in one concise sentence for dynamic malware analysis: {api}"
        try:
            result = generator(prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"]
            text = str(result).replace(prompt, "").strip()
        except Exception as exc:
            logging.warning("Local LLM description failed for %s: %s", api, exc)
            break
        out[api] = text or rule_template_description(api)
    return out


def hash_embedding(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = split_identifier(text) + text.lower().split()
    if not tokens:
        tokens = [text]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        idx = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def encode_descriptions(
    descriptions: Dict[str, str],
    encoder: str,
    encoder_model: str,
    batch_size: int,
    hash_dim: int,
) -> Tuple[Dict[str, np.ndarray], str]:
    apis = sorted(descriptions)
    texts = [descriptions[api] for api in apis]
    requested = encoder

    if encoder in {"auto", "sentence-transformer"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(encoder_model)
            arr = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
            return {api: np.asarray(vec, dtype=np.float32) for api, vec in zip(apis, arr)}, "sentence-transformer"
        except Exception as exc:
            if requested == "sentence-transformer":
                raise
            logging.warning("sentence-transformer encoder unavailable, trying BERT/hash fallback: %s", exc)

    if encoder in {"auto", "bert"}:
        try:
            import torch as torch_mod
            from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizer

            device = "cuda" if torch_mod.cuda.is_available() else "cpu"
            if encoder_model == "prajjwal1/bert-tiny":
                hub = Path.home() / ".cache/huggingface/hub/models--prajjwal1--bert-tiny/snapshots"
                snapshots = sorted(path for path in hub.glob("*") if (path / "config.json").exists())
                if not snapshots:
                    raise FileNotFoundError("Cached prajjwal1/bert-tiny snapshot was not found.")
                snapshot = snapshots[-1]
                tokenizer = BertTokenizer(vocab=str(snapshot / "vocab.txt"))
                config = BertConfig.from_json_file(str(snapshot / "config.json"))
                model = BertModel.from_pretrained(snapshot, config=config).to(device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(encoder_model, use_fast=False)
                model = AutoModel.from_pretrained(encoder_model).to(device)
            model.eval()
            vectors = []
            with torch_mod.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    enc = tokenizer(batch, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
                    output = model(**enc).last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1)
                    pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    pooled = F.normalize(pooled, p=2, dim=1)
                    vectors.extend(pooled.cpu().numpy().astype(np.float32))
            return {api: vec for api, vec in zip(apis, vectors)}, "bert"
        except Exception as exc:
            if requested == "bert":
                raise
            logging.warning("BERT encoder unavailable, using hash fallback: %s", exc)

    logging.warning("Using deterministic hash embeddings. Install sentence-transformers/transformers for BERT encoding.")
    return {api: hash_embedding(descriptions[api], hash_dim) for api in apis}, "hash"


def build_sequence_array(
    samples: Sequence[Sample],
    api_embeddings: Dict[str, np.ndarray],
    max_len: int,
) -> np.ndarray:
    emb_dim = len(next(iter(api_embeddings.values())))
    arr = np.zeros((len(samples), max_len, emb_dim), dtype=np.float32)
    unknown_cache: Dict[str, np.ndarray] = {}
    for i, sample in enumerate(samples):
        seq = sample.api_seq[:max_len]
        for j, api in enumerate(seq):
            vec = api_embeddings.get(api)
            if vec is None:
                if api not in unknown_cache:
                    unknown_cache[api] = hash_embedding(rule_template_description(api), emb_dim)
                vec = unknown_cache[api]
            arr[i, j] = vec
    return arr


if torch is not None:

    class SequenceDataset(Dataset):
        def __init__(
            self,
            samples: Sequence[Sample],
            y: np.ndarray,
            api_embeddings: Dict[str, np.ndarray],
            max_len: int,
            emb_dim: int,
        ) -> None:
            self.samples = samples
            self.y = y.astype(np.int64)
            self.api_embeddings = api_embeddings
            self.max_len = max_len
            self.emb_dim = emb_dim
            self.unknown_cache: Dict[str, np.ndarray] = {}

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            arr = np.zeros((self.max_len, self.emb_dim), dtype=np.float32)
            for j, api in enumerate(self.samples[idx].api_seq[: self.max_len]):
                vec = self.api_embeddings.get(api)
                if vec is None:
                    if api not in self.unknown_cache:
                        self.unknown_cache[api] = hash_embedding(rule_template_description(api), self.emb_dim)
                    vec = self.unknown_cache[api]
                arr[j] = vec
            return torch.from_numpy(arr), torch.tensor(self.y[idx], dtype=torch.long)


    class ApiTextCNN(nn.Module):
        def __init__(
            self, emb_dim: int, num_classes: int, channels: int, kernels: Sequence[int], dropout: float
        ) -> None:
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in kernels])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(channels * len(kernels), num_classes)

        def forward(self, x):
            x = x.transpose(1, 2)
            features = []
            for conv in self.convs:
                h = F.relu(conv(x))
                features.append(F.adaptive_max_pool1d(h, 1).squeeze(-1))
            return self.fc(self.dropout(torch.cat(features, dim=1)))
else:
    SequenceDataset = None
    ApiTextCNN = None


def binary_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, y_true), key=lambda item: item[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        positives = sum(label for _, label in pairs[i : j + 1])
        rank_sum += positives * avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def multiclass_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    aucs = []
    for class_id in range(scores.shape[1]):
        binary_true = (y_true == class_id).astype(np.int64)
        if 0 < int(binary_true.sum()) < len(binary_true):
            aucs.append(binary_auc(binary_true.tolist(), scores[:, class_id].tolist()))
    return float(np.mean(aucs)) if aucs else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray], average: str) -> Dict[str, float]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    acc = float((y_true == y_pred).mean())
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    if average == "binary" and len(labels) == 2 and 1 in labels:
        idx = labels.index(1)
        precision, recall, f1 = precisions[idx], recalls[idx], f1s[idx]
    else:
        precision, recall, f1 = float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))
    if scores is None:
        auc = float("nan")
    elif scores.shape[1] == 2:
        auc = binary_auc(y_true.tolist(), scores[:, 1].tolist())
    else:
        auc = multiclass_auc(y_true, scores)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def evaluate(model, loader, device: str, average: str) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred, scores = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())
            scores.extend(prob.tolist())
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float32)
    return classification_metrics(y_true_arr, y_pred_arr, scores_arr, average), y_true_arr, y_pred_arr, scores_arr


def train_cnn(
    train_samples: Sequence[Sample],
    y_train: np.ndarray,
    val_samples: Sequence[Sample],
    y_val: np.ndarray,
    test_samples: Sequence[Sample],
    y_test: np.ndarray,
    api_embeddings: Dict[str, np.ndarray],
    num_classes: int,
    args: argparse.Namespace,
    task_name: str,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for CNN training: {TORCH_IMPORT_ERROR}")

    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logging.info("%s training device=%s", task_name, device)
    emb_dim = len(next(iter(api_embeddings.values())))
    model = ApiTextCNN(
        emb_dim=emb_dim,
        num_classes=num_classes,
        channels=args.channels,
        kernels=[int(k) for k in args.kernels.split(",")],
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        SequenceDataset(train_samples, y_train, api_embeddings, args.max_len, emb_dim),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SequenceDataset(val_samples, y_val, api_embeddings, args.max_len, emb_dim),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        SequenceDataset(test_samples, y_test, api_embeddings, args.max_len, emb_dim),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    best_f1 = -1.0
    best_state = None
    stale = 0
    average = "binary" if num_classes == 2 else "macro"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics, _, _, _ = evaluate(model, val_loader, device, average)
        logging.info(
            "%s epoch=%d train_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%s",
            task_name,
            epoch,
            float(np.mean(losses)) if losses else 0.0,
            val_metrics["accuracy"],
            val_metrics["f1"],
            f"{val_metrics['auc']:.4f}" if not math.isnan(val_metrics["auc"]) else "nan",
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                logging.info("%s early stopping at epoch=%d", task_name, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_dir = ROOT / "datasets_50k" / args.dataset_name / "artifacts" / "gpt4_api_bert_cnn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / f"{task_name}_best.pt")
    val_metrics, _, _, _ = evaluate(model, val_loader, device, average)
    test_metrics, _, _, _ = evaluate(model, test_loader, device, average)
    return val_metrics, test_metrics, epoch


def label_array(samples: Sequence[Sample], mapping: Dict[str, int], field: str) -> np.ndarray:
    if field == "label":
        return np.asarray([mapping[s.label] for s in samples], dtype=np.int64)
    return np.asarray([mapping[s.family] for s in samples], dtype=np.int64)


def append_baseline_result(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            fieldnames = next(csv.reader(handle))
    else:
        fieldnames = list(row)
    existing_rows = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    existing_rows = [item for item in existing_rows if item.get("method") != row["method"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--dataset-name", default="quo_vadis")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=ROOT / "datasets_50k/artifacts/gpt4_api_bert_cnn/api_descriptions.json",
    )
    parser.add_argument("--log-path", type=Path, default=BASELINE_DIR / "gpt4_api_bert_cnn.log")
    parser.add_argument(
        "--description-source",
        choices=["auto", "rule", "gpt4", "openai", "deepseek", "local-llm", "local"],
        default="auto",
    )
    parser.add_argument("--encoder", choices=["auto", "sentence-transformer", "bert", "hash"], default="auto")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--kernels", default="3,5,7")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-family", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split sample limit for smoke tests.")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after data, description, and embedding preparation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(BASELINE_DIR / ".env")
    setup_logging(args.log_path)
    set_seed(args.seed)

    if args.description_source == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            args.description_source = "gpt4"
        elif os.environ.get("DEEPSEEK_API_KEY"):
            args.description_source = "deepseek"
        elif os.environ.get("LOCAL_LLM_MODEL"):
            args.description_source = "local-llm"
        else:
            args.description_source = "rule"
    if args.encoder == "bert" and args.encoder_model == "sentence-transformers/all-MiniLM-L6-v2":
        args.encoder_model = args.bert_model

    samples = load_samples(args.data_path)
    validate_dataset(samples)
    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split == "val"]
    test = [s for s in samples if s.split == "test"]
    if args.limit:
        train, val, test = train[: args.limit], val[: args.limit], test[: args.limit]
        logging.info("Smoke-test limit applied per split: %d", args.limit)

    train_apis = {api for sample in train for api in sample.api_seq}
    descriptions = load_or_create_descriptions(train_apis, args.cache_path, args.description_source)
    all_apis = {api for sample in samples for api in sample.api_seq}
    encoding_descriptions = dict(descriptions)
    unseen_apis = all_apis - set(encoding_descriptions)
    if unseen_apis:
        logging.info(
            "Encoding %d APIs unseen in training with transient rule descriptions.",
            len(unseen_apis),
        )
        encoding_descriptions.update(
            {api: rule_template_description(api) for api in unseen_apis}
        )
    api_embeddings, actual_encoder = encode_descriptions(
        descriptions=encoding_descriptions,
        encoder=args.encoder,
        encoder_model=args.encoder_model,
        batch_size=args.encode_batch_size,
        hash_dim=args.hash_dim,
    )
    logging.info("Encoded %d API descriptions using encoder=%s", len(api_embeddings), actual_encoder)

    if args.prepare_only:
        emb_dim = len(next(iter(api_embeddings.values())))
        logging.info(
            "Prepare-only complete: lazy shapes x_train=(%d, %d, %d) x_val=(%d, %d, %d) x_test=(%d, %d, %d)",
            len(train),
            args.max_len,
            emb_dim,
            len(val),
            args.max_len,
            emb_dim,
            len(test),
            args.max_len,
            emb_dim,
        )
        return
    started = time.time()
    task_results: Dict[str, object] = {}

    detection_mapping = {"benign": 0, "malware": 1}
    y_train = label_array(train, detection_mapping, "label")
    y_val = label_array(val, detection_mapping, "label")
    y_test = label_array(test, detection_mapping, "label")
    val_metrics, test_metrics, epochs_run = train_cnn(
        train, y_train, val, y_val, test, y_test, api_embeddings, 2, args, "detection"
    )
    task_results["detection"] = {
        "epochs_run": epochs_run,
        "validation": val_metrics,
        "test": test_metrics,
    }

    if args.run_family:
        families = sorted({s.family for s in train})
        family_mapping = {family: idx for idx, family in enumerate(families)}
        if len(family_mapping) > 2:
            logging.info("Running family classification with classes=%s", family_mapping)
            fy_train = label_array(train, family_mapping, "family")
            fy_val = label_array(val, family_mapping, "family")
            fy_test = label_array(test, family_mapping, "family")
            fam_val_metrics, fam_test_metrics, fam_epochs = train_cnn(
                train,
                fy_train,
                val,
                fy_val,
                test,
                fy_test,
                api_embeddings,
                len(family_mapping),
                args,
                "family",
            )
            task_results["family"] = {
                "classes": families,
                "epochs_run": fam_epochs,
                "validation": fam_val_metrics,
                "test": fam_test_metrics,
            }
        else:
            logging.info("Family labels unavailable or degenerate; skipping family classifier.")

    elapsed = time.time() - started
    output = {
        "method": "gpt4_api_bert_cnn",
        "status": "ok",
        "dataset": args.dataset_name,
        "data_path": str(args.data_path),
        "split_source": "existing jsonl.gz split field",
        "split_seed": args.seed,
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "description_source": args.description_source,
        "description_cache": str(args.cache_path),
        "training_api_count": len(train_apis),
        "encoder": actual_encoder,
        "encoder_model": args.encoder_model,
        "cnn": {
            "max_len": args.max_len,
            "channels": args.channels,
            "kernels": [int(k) for k in args.kernels.split(",")],
            "dropout": args.dropout,
        },
        "tasks": task_results,
        "runtime_seconds": elapsed,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)

    if args.baseline_csv:
        detection = task_results["detection"]
        append_baseline_result(
            args.baseline_csv,
            {
                "method": "gpt4_api_bert_cnn",
                "status": "ok",
                "epochs_run": detection["epochs_run"],
                "best_val_f1": detection["validation"]["f1"],
                "threshold": 0.5,
                **{f"val_{key}": value for key, value in detection["validation"].items()},
                **{f"test_{key}": value for key, value in detection["test"].items()},
                "runtime_seconds": elapsed,
                "error": "",
            },
        )
    logging.info("RESULT %s", output)


def result_row(
    args: argparse.Namespace,
    actual_encoder: str,
    task: str,
    metrics: Dict[str, float],
    n_train: int,
    n_val: int,
    n_test: int,
) -> Dict[str, object]:
    return {
        "method": "gpt4_api_bert_cnn",
        "task": task,
        "dataset": str(args.data_path),
        "split_seed": args.seed,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "encoder": actual_encoder,
        "description_source": args.description_source,
        "max_len": args.max_len,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
    }


if __name__ == "__main__":
    main()
