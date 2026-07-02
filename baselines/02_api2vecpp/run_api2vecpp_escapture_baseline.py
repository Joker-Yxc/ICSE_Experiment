#!/usr/bin/env python3
"""Run an API2Vec++-style baseline on the local EsCapture/Quo Vadis reports.

The upstream API2Vec++ code expects XLSX traces with pid/category/arg columns.
The local EsCapture data is JSON with entrypoint-level API lists, so this runner
keeps the API2Vec++ representation/training shape while adapting graph paths to
the available sequence structure:

  JSON API sequence -> path corpus -> BPE tokenizer + small RoBERTa MLM
  -> TextCNN classifier over RoBERTa hidden states.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tokenizers import ByteLevelBPETokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaModel, RobertaTokenizer


BASELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "datasets_50k/quo_vadis/data/raw"
BENIGN_FOLDERS = {"report_clean", "report_windows_syswow64"}
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


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


def collect_escapture_binary_reports(
    data_root: Path,
    max_attack_samples: int,
    max_normal_samples: int,
    seed: int,
) -> list[dict]:
    attack_rows: list[dict] = []
    normal_rows: list[dict] = []
    for folder in sorted(data_root.glob("windows_emulation_*set/report_*")):
        if not folder.is_dir():
            continue
        is_benign = folder.name in BENIGN_FOLDERS
        family = "benign" if is_benign else folder.name.removeprefix("report_")
        target = normal_rows if is_benign else attack_rows
        for path in sorted(folder.glob("*.json")):
            target.append(
                {
                    "path": str(path),
                    "sha256": path.stem,
                    "folder": folder.name,
                    "family": family,
                    "detection_label": 0 if is_benign else 1,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(attack_rows)
    rng.shuffle(normal_rows)
    if max_attack_samples > 0:
        attack_rows = attack_rows[:max_attack_samples]
    if max_normal_samples > 0:
        normal_rows = normal_rows[:max_normal_samples]
    rows = attack_rows + normal_rows
    rng.shuffle(rows)
    return rows


def split_rows(rows: list[dict], train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[list[dict], list[dict], list[dict]]:
    split1 = int(train_ratio * len(rows))
    split2 = int((train_ratio + val_ratio) * len(rows))
    return rows[:split1], rows[split1:split2], rows[split2:]


def normalize_api_name(api_name: str) -> str:
    return api_name.rsplit(".", 1)[-1].strip() or "UNK_API"


def load_api_sequence(path: Path) -> list[str]:
    report = json.loads(path.read_text(errors="ignore"))
    if not isinstance(report, list):
        report = [report]
    apis: list[str] = []
    for entry in report:
        if not isinstance(entry, dict):
            continue
        for api in entry.get("apis", []) or []:
            if isinstance(api, dict) and api.get("api_name"):
                apis.append(normalize_api_name(str(api["api_name"])))
    return apis


def load_or_build_sequences(rows: list[dict], cache_path: Path, rebuild: bool) -> list[list[str]]:
    if cache_path.exists() and not rebuild:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    seqs: list[list[str]] = []
    kept_rows: list[dict] = []
    for row in tqdm(rows, desc=f"load {cache_path.stem}"):
        try:
            seq = load_api_sequence(Path(row["path"]))
        except Exception as exc:
            print(f"[skip] {row['path']}: {type(exc).__name__}: {exc}")
            continue
        if not seq:
            continue
        seqs.append(seq)
        kept_rows.append(row)
    rows[:] = kept_rows
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(seqs), encoding="utf-8")
    cache_path.with_suffix(".meta.json").write_text(json.dumps(kept_rows, indent=2), encoding="utf-8")
    return seqs


def generate_path_corpus(
    sequences: list[list[str]],
    rng: random.Random,
    walks_per_sequence: int,
    walk_steps: int,
) -> list[str]:
    corpus: list[str] = []
    for seq in sequences:
        if len(seq) < 2:
            corpus.append(" ".join(seq))
            continue
        corpus.append(" ".join(seq[:walk_steps]))
        for _ in range(walks_per_sequence):
            start = rng.randrange(0, len(seq))
            max_len = min(walk_steps, len(seq) - start)
            if max_len <= 1:
                continue
            length = rng.randint(2, max_len)
            corpus.append(" ".join(seq[start : start + length]))
    rng.shuffle(corpus)
    return corpus


def train_or_load_tokenizer(corpus: list[str], tokenizer_dir: Path, vocab_size: int, rebuild: bool) -> RobertaTokenizer:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    if rebuild or not (vocab_path.exists() and merges_path.exists()):
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = tokenizer_dir / "api2vecpp_paths.txt"
        corpus_path.write_text("\n".join(corpus), encoding="utf-8")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[str(corpus_path)], vocab_size=vocab_size, min_frequency=1, special_tokens=SPECIAL_TOKENS)
        tokenizer.save_model(str(tokenizer_dir))
    return RobertaTokenizer.from_pretrained(str(tokenizer_dir), model_max_length=512)


class EncodedTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer: RobertaTokenizer, max_length: int, mlm: bool = False):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.mlm = mlm
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        if self.mlm:
            labels = item["input_ids"].clone()
            rand = torch.rand(labels.shape)
            special = torch.zeros(labels.shape, dtype=torch.bool)
            for token_id in self.special_ids:
                special |= labels.eq(token_id)
            mask = (rand < 0.15) & ~special
            item["input_ids"][mask] = self.mask_token_id
            labels[~mask] = -100
            item["labels"] = labels
        elif self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class RobertaTextCNN(nn.Module):
    def __init__(self, encoder: RobertaModel, hidden_size: int, dropout: float, num_filters: int, filter_sizes: list[int]):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (size, hidden_size)) for size in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(states).unsqueeze(1)
        xs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        xs = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in xs]
        x = torch.cat(xs, dim=1)
        return self.fc(self.dropout(x)).squeeze(1)


def build_roberta_config(tokenizer: RobertaTokenizer, max_length: int, hidden_size: int, layers: int, heads: int) -> RobertaConfig:
    return RobertaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_length + 2,
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=hidden_size * 4,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def train_mlm(
    model: RobertaForMaskedLM,
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(loader, desc=f"mlm epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[mlm] epoch={epoch} loss={np.mean(losses):.6f}")


def train_classifier(
    model: RobertaTextCNN,
    train_data: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(loader, desc=f"cls epoch {epoch}"):
            labels = batch.pop("labels").float().to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[cls] epoch={epoch} loss={np.mean(losses):.6f}")


@torch.no_grad()
def predict(model: RobertaTextCNN, dataset: Dataset, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size)
    model.to(device)
    model.eval()
    probs: list[np.ndarray] = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    return metrics


def save_predictions(path: Path, rows: list[dict], probs: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sha256", "family", "label", "prob_malware", "pred", "path"])
        writer.writeheader()
        for row, prob in zip(rows, probs):
            writer.writerow(
                {
                    "sha256": row["sha256"],
                    "family": row["family"],
                    "label": row["detection_label"],
                    "prob_malware": f"{prob:.8f}",
                    "pred": int(prob >= 0.5),
                    "path": row["path"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(BASELINE_ROOT / "results" / "main_50k"))
    parser.add_argument("--max-attack-samples", type=int, default=500)
    parser.add_argument("--max-normal-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mlm-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mlm-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mlm-lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--walks-per-sequence", type=int, default=5)
    parser.add_argument("--walk-steps", type=int, default=49)
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"[setup] data_root={data_root}")
    print(f"[setup] out_dir={out_dir}")
    print(f"[setup] device={device}")

    rows = collect_escapture_binary_reports(data_root, args.max_attack_samples, args.max_normal_samples, args.seed)
    train_rows, val_rows, test_rows = split_rows(rows)
    print(f"[data] selected total={len(rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    cache_prefix = f"escapture_binary_a{args.max_attack_samples}_n{args.max_normal_samples}"
    train_seqs = load_or_build_sequences(train_rows, out_dir / f"seq_train_{cache_prefix}.json", args.rebuild_cache)
    val_seqs = load_or_build_sequences(val_rows, out_dir / f"seq_val_{cache_prefix}.json", args.rebuild_cache)
    test_seqs = load_or_build_sequences(test_rows, out_dir / f"seq_test_{cache_prefix}.json", args.rebuild_cache)
    train_texts = [" ".join(seq) for seq in train_seqs]
    val_texts = [" ".join(seq) for seq in val_seqs]
    test_texts = [" ".join(seq) for seq in test_seqs]
    y_train = [int(r["detection_label"]) for r in train_rows]
    y_val = np.array([int(r["detection_label"]) for r in val_rows], dtype=np.int64)
    y_test = np.array([int(r["detection_label"]) for r in test_rows], dtype=np.int64)

    rng = random.Random(args.seed)
    corpus = generate_path_corpus(train_seqs + val_seqs, rng, args.walks_per_sequence, args.walk_steps)
    (out_dir / "path_corpus.txt").write_text("\n".join(corpus), encoding="utf-8")
    print(f"[paths] generated {len(corpus)} training paths")

    tokenizer = train_or_load_tokenizer(corpus, out_dir / "tokenizer", args.vocab_size, args.rebuild_cache)
    config = build_roberta_config(tokenizer, args.max_length, args.hidden_size, args.layers, args.heads)
    mlm_model_path = out_dir / "api2vecpp_mlm"
    if mlm_model_path.exists() and not args.rebuild_cache:
        mlm = RobertaForMaskedLM.from_pretrained(str(mlm_model_path))
    else:
        mlm = RobertaForMaskedLM(config)
        mlm_data = EncodedTextDataset(corpus, None, tokenizer, args.max_length, mlm=True)
        train_mlm(mlm, mlm_data, args.mlm_epochs, args.mlm_batch_size, args.mlm_lr, device)
        mlm.save_pretrained(str(mlm_model_path))

    encoder = RobertaModel(config)
    encoder.load_state_dict(mlm.roberta.state_dict(), strict=False)
    model = RobertaTextCNN(encoder, args.hidden_size, args.dropout, num_filters=128, filter_sizes=[2, 3, 4])

    train_data = EncodedTextDataset(train_texts, y_train, tokenizer, args.max_length)
    val_data = EncodedTextDataset(val_texts, y_val.tolist(), tokenizer, args.max_length)
    test_data = EncodedTextDataset(test_texts, y_test.tolist(), tokenizer, args.max_length)

    t0 = time.time()
    train_classifier(model, train_data, args.epochs, args.batch_size, args.lr, device)
    val_probs = predict(model, val_data, args.batch_size, device)
    test_probs = predict(model, test_data, args.batch_size, device)
    val_metrics = binary_metrics(y_val, val_probs)
    test_metrics = binary_metrics(y_test, test_probs)
    print(f"[result] validation detection: {val_metrics}")
    print(f"[result] test detection: {test_metrics}")

    metrics = {
        "baseline": "API2Vec++",
        "adaptation": "EsCapture JSON API sequence -> API2Vec++-style path corpus + BPE/RoBERTa MLM + TextCNN binary classifier",
        "upstream_repo": "https://github.com/yyyjn/API2VecPlus",
        "dataset": "Quo Vadis Speakeasy local reports via escapture",
        "data_root": str(data_root),
        "settings": vars(args),
        "sample_counts": {
            "selected_total": len(rows),
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
            "path_corpus": len(corpus),
        },
        "label_protocol": "report_clean/report_windows_syswow64=benign; all other report_* folders=malware",
        "validation_detection": val_metrics,
        "test_detection": test_metrics,
        "runtime_seconds": round(time.time() - t0, 3),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_predictions(out_dir / "detection_predictions.csv", test_rows, test_probs)
    torch.save(model.state_dict(), out_dir / "api2vecpp_textcnn_classifier.pt")
    print(f"[done] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
