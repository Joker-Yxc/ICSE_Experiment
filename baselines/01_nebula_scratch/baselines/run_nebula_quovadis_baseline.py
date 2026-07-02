#!/usr/bin/env python3
"""Run Nebula-Scratch on local Quo Vadis Speakeasy reports.

This is the fair Nebula baseline used for the main comparison:
- balanced malware/benign sampling
- same 70/15/15 split shape as the EsCapturer run
- Nebula BPE tokenizer is reused, but Transformer weights are randomly initialized
- no released pretrained Nebula model weights are used for training
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


NEBULA_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = NEBULA_ROOT.parents[1]
if str(NEBULA_ROOT) not in sys.path:
    sys.path.insert(0, str(NEBULA_ROOT))

from nebula import Nebula  # noqa: E402


BENIGN_FOLDERS = {"report_clean", "report_windows_syswow64"}


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
        family = folder.name.removeprefix("report_")
        is_benign = folder.name in BENIGN_FOLDERS
        target = normal_rows if is_benign else attack_rows
        for path in sorted(folder.glob("*.json")):
            target.append(
                {
                    "path": str(path),
                    "sha256": path.stem,
                    "folder": folder.name,
                    "family": "benign" if is_benign else family,
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


def load_or_encode(rows: list[dict], nebula: Nebula, cache_path: Path, rebuild: bool) -> tuple[np.ndarray, list[dict]]:
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and meta_path.exists() and not rebuild:
        return np.load(cache_path), json.loads(meta_path.read_text())

    xs: list[np.ndarray] = []
    kept: list[dict] = []
    skipped = 0
    for idx, row in enumerate(rows, 1):
        try:
            report = json.loads(Path(row["path"]).read_text(errors="ignore"))
            x = nebula.preprocess(report)
        except Exception as exc:
            skipped += 1
            print(f"[skip] {row['path']}: {type(exc).__name__}: {exc}")
            continue
        if x is None:
            skipped += 1
            continue
        xs.append(np.asarray(x, dtype=np.int64))
        kept.append(row)
        if idx % 250 == 0:
            print(f"[encode] {idx}/{len(rows)} reports processed, kept={len(kept)}, skipped={skipped}")

    if not xs:
        raise RuntimeError("No reports were encoded. Check the input dataset path.")

    X = np.vstack(xs).astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, X)
    meta_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    return X, kept


def reset_model_parameters(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, probs))
    return out


def select_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 197):
        preds = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_compact_split(compact_data: Path, data_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    report_index: dict[str, Path] = {}
    for path in data_root.glob("windows_emulation_*set/report_*/*.json"):
        if path.stem in report_index:
            raise RuntimeError(f"Ambiguous report identity: {path.stem}")
        report_index[path.stem] = path

    opener = gzip.open if compact_data.suffix == ".gz" else open
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    with opener(compact_data, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            split = row.get("split")
            if split not in splits:
                raise RuntimeError(f"Invalid split at line {line_number}: {split!r}")
            sample_id = str(row["sample_id"])
            report_path = report_index.get(sample_id)
            if report_path is None:
                raise RuntimeError(f"No raw Speakeasy report for sample_id={sample_id!r}")
            label = str(row["label"]).lower()
            splits[split].append(
                {
                    "path": str(report_path),
                    "sha256": sample_id,
                    "folder": report_path.parent.name,
                    "family": str(row.get("family", "")),
                    "detection_label": 0 if label == "benign" else 1,
                    "split": split,
                }
            )
    return splits["train"], splits["val"], splits["test"]


@torch.no_grad()
def predict_binary(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X).long()), batch_size=batch_size)
    for (xb,) in loader:
        logits = model(xb.to(device)).float().squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_binary(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"[train detection] epoch={epoch} loss={np.mean(losses):.6f}")


def save_detection_predictions(
    path: Path, rows: list[dict], probs: np.ndarray, threshold: float
) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row, prob in zip(rows, probs):
            payload = {
                "sample_id": row["sha256"],
                "family": row["family"],
                "label": row["detection_label"],
                "score": float(prob),
                "prediction": int(prob >= threshold),
                "threshold": threshold,
            }
            f.write(json.dumps(payload) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nebula-Scratch baseline on Quo Vadis Speakeasy reports.")
    parser.add_argument(
        "--data-root",
        default=str(WORKSPACE_ROOT / "datasets_50k/quo_vadis/data/raw"),
    )
    parser.add_argument(
        "--compact-data",
        help="Leakage-safe JSONL/JSONL.GZ with fixed train/val/test assignments",
    )
    parser.add_argument("--out-dir", default=str(NEBULA_ROOT / "results" / "main_50k"))
    parser.add_argument("--max-attack-samples", type=int, default=500)
    parser.add_argument("--max-normal-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=512)
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

    print(f"[setup] baseline=Nebula-Scratch")
    print(f"[setup] data_root={data_root}")
    print(f"[setup] out_dir={out_dir}")
    print(f"[setup] device={device}")

    compact_data = Path(args.compact_data).resolve() if args.compact_data else None
    if compact_data:
        train_rows, val_rows, test_rows = load_compact_split(compact_data, data_root)
        all_rows = train_rows + val_rows + test_rows
    else:
        all_rows = collect_escapture_binary_reports(
            data_root, args.max_attack_samples, args.max_normal_samples, args.seed
        )
        train_rows, val_rows, test_rows = split_rows(all_rows)
    print(
        f"[data] selected total={len(all_rows)} "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )

    neb = Nebula(vocab_size=50000, seq_len=args.seq_len, tokenizer="bpe")
    reset_model_parameters(neb.model)
    print("[setup] model weights reinitialized; pretrained tokenizer is used only for input encoding")
    neb.model.to(device)

    cache_prefix = (
        f"leakage_safe_{sha256_file(compact_data)[:12]}"
        if compact_data
        else f"scratch_a{args.max_attack_samples}_n{args.max_normal_samples}"
    )
    X_train, train_rows = load_or_encode(
        train_rows, neb, out_dir / f"X_train_{cache_prefix}_seq{args.seq_len}.npy", args.rebuild_cache
    )
    X_val, val_rows = load_or_encode(
        val_rows, neb, out_dir / f"X_val_{cache_prefix}_seq{args.seq_len}.npy", args.rebuild_cache
    )
    X_test, test_rows = load_or_encode(
        test_rows, neb, out_dir / f"X_test_{cache_prefix}_seq{args.seq_len}.npy", args.rebuild_cache
    )

    y_train = np.array([r["detection_label"] for r in train_rows], dtype=np.int64)
    y_val = np.array([r["detection_label"] for r in val_rows], dtype=np.int64)
    y_test = np.array([r["detection_label"] for r in test_rows], dtype=np.int64)

    t0 = time.time()
    random_probs = predict_binary(neb.model, X_test, args.batch_size, device)
    random_metrics = binary_metrics(y_test, random_probs, 0.5)
    print(f"[result] random-init before training: {random_metrics}")

    train_binary(neb.model, X_train, y_train, args.epochs, args.batch_size, args.lr, device)
    val_probs = predict_binary(neb.model, X_val, args.batch_size, device)
    test_probs = predict_binary(neb.model, X_test, args.batch_size, device)
    threshold = select_threshold(y_val, val_probs)
    val_metrics = binary_metrics(y_val, val_probs, threshold)
    test_metrics = binary_metrics(y_test, test_probs, threshold)
    print(f"[result] validation detection: {val_metrics}")
    print(f"[result] test detection: {test_metrics}")

    metrics = {
        "baseline": "Nebula-Scratch",
        "dataset": "Quo Vadis Speakeasy local reports",
        "data_root": str(data_root),
        "compact_data": str(compact_data) if compact_data else None,
        "data_sha256": sha256_file(compact_data) if compact_data else None,
        "split_type": (
            "leakage-safe report-group-disjoint exact-sequence-disjoint"
            if compact_data
            else "random preliminary"
        ),
        "seed": args.seed,
        "threshold": threshold,
        "threshold_selection": "validation F1",
        "settings": vars(args),
        "encoded_shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "label_protocol": {
            "detection": "report_clean/report_windows_syswow64=benign; all other report_* folders=malware",
        },
        "random_init_before_training": random_metrics,
        "validation_detection": val_metrics,
        "test_detection": test_metrics,
        "runtime_seconds": round(time.time() - t0, 3),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    save_detection_predictions(
        out_dir / "validation_predictions.jsonl.gz", val_rows, val_probs, threshold
    )
    save_detection_predictions(
        out_dir / "test_predictions.jsonl.gz", test_rows, test_probs, threshold
    )
    torch.save(neb.model.state_dict(), out_dir / "nebula_scratch_detection.pt")
    print(f"[done] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
