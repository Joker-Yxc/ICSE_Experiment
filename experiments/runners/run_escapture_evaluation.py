#!/usr/bin/env python3
"""Run the full dual-view EsCapturer evaluation on compact JSONL.GZ datasets.

This runner is intentionally separate from ``run_compact_baselines.py``.  The
latter reports the validated TF-IDF classifier results already present in the
repository, whereas this file evaluates the neural sequence/graph model that
implements interleaving-prior adaptive fusion.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import platform
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from escapture.escapture_true import (
    EsCapturer,
    IntentionMapper,
    SyscallVocab,
    build_intention_graph,
    set_random_seed,
)
from escapture.llm_behavior_extractor import FrozenTemplateBehaviorExtractor


VARIANTS = {
    "full": {},
    "wo_semantic_elements": {"semantic_mode": "matched_nonsemantic"},
    "wo_llm_elements": {"semantic_mode": "matched_nonsemantic"},
    "wo_sequence_view": {"use_sequence_view": False},
    "wo_graph_view": {"use_graph_view": False},
    "wo_interleaving_prior_in_gating": {"use_gating_prior": False},
    "wo_structural_bias_in_attention": {"use_structural_bias": False},
    "wo_relation_features": {"use_relation_features": False},
    "wo_preference_loss": {"lambda_pref": 0.0},
    "hard_switching": {"hard_switching": True},
    "sigmoid_weighting": {"weighting_mode": "sigmoid"},
    "random_interleaving_matrix": {"prior_mode": "random"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz",
    )
    parser.add_argument("--out-dir", default="datasets_50k/quo_vadis/results/full_model")
    parser.add_argument(
        "--experiment",
        choices=["main", "ablation", "unknown_family", "profile"],
        default="main",
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument(
        "--objective",
        choices=["dsvdd", "bce"],
        default="dsvdd",
        help="Detection objective. BCE changes only the final head/objective and "
        "uses balanced attack/benign training pairs.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variants for an ablation run; default uses all canonical variants.",
    )
    parser.add_argument(
        "--held-out-family",
        action="append",
        default=[],
        help="Family to remove from train/validation. Repeat for multiple runs; use 'all' for every malware family.",
    )
    parser.add_argument("--seeds", default="7", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--malware-class-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda-pref", type=float, default=0.1)
    parser.add_argument("--gating-temperature", type=float, default=1.0)
    parser.add_argument("--max-units", type=int, default=5)
    parser.add_argument(
        "--unit-selection",
        choices=["prefix", "uniform-cover"],
        default="prefix",
        help="How to enforce max-units. 'prefix' reproduces the legacy truncation; "
        "'uniform-cover' retains calls across the complete model input.",
    )
    parser.add_argument(
        "--benign-sampling",
        choices=["paired-prefix", "epoch-resample"],
        default="paired-prefix",
        help="'paired-prefix' reproduces the legacy fixed subset. 'epoch-resample' "
        "visits every benign sample each epoch and resamples attacks for pairing.",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Deterministic smoke-test limit; zero uses every row.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=0,
        help="Optional deterministic cap for expensive neural runs; zero uses all training samples.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--save-view-weights", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument(
        "--cache-init",
        default="",
        help="Optional frozen-template cache copied into each new run directory before extraction.",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help="Optional file checked after each epoch to request a graceful stop.",
    )
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(value) / (1024.0 * 1024.0)
    return float(value) / 1024.0


def load_rows(path: Path, limit_per_split: int = 0) -> tuple[dict[str, list[dict]], float]:
    started = time.perf_counter()
    rows = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = row["split"]
            if limit_per_split and len(rows[split]) >= limit_per_split:
                continue
            rows[split].append(row)
    return dict(rows), time.perf_counter() - started


def malware_families(rows: dict[str, list[dict]]) -> list[str]:
    return sorted(
        {
            str(row.get("family", "unknown"))
            for split_rows in rows.values()
            for row in split_rows
            if row["label"] == "malware"
        }
    )


def apply_unknown_family_protocol(
    rows: dict[str, list[dict]], held_out_family: str
) -> dict[str, list[dict]]:
    filtered = {
        "train": [
            row
            for row in rows["train"]
            if not (row["label"] == "malware" and row.get("family") == held_out_family)
        ],
        "val": [
            row
            for row in rows["val"]
            if not (row["label"] == "malware" and row.get("family") == held_out_family)
        ],
        "test": [
            row
            for row in rows["test"]
            if row["label"] == "benign" or row.get("family") == held_out_family
        ],
    }
    if not any(row["label"] == "malware" for row in filtered["test"]):
        raise ValueError(f"Held-out family {held_out_family!r} has no malware test samples")
    return filtered


def cap_training_rows(rows: list[dict], max_per_class: int, seed: int) -> list[dict]:
    if not max_per_class:
        return list(rows)
    rng = np.random.default_rng(seed)
    output = []
    for label in ("benign", "malware"):
        candidates = [row for row in rows if row["label"] == label]
        selected = rng.permutation(len(candidates))[:max_per_class]
        output.extend(candidates[index] for index in selected)
    rng.shuffle(output)
    return output


def equal_length_groups(seq: list[str], unit_count: int) -> list[dict]:
    """Create non-semantic groups with the same count as semantic units."""
    if not seq:
        return [{"start": 0, "end": 1, "intention": "raw_chunk", "syscalls": ["<PAD>"]}]
    unit_count = max(1, min(unit_count, len(seq)))
    boundaries = np.linspace(0, len(seq), unit_count + 1, dtype=int)
    groups = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end <= start:
            continue
        groups.append(
            {
                "unit_id": f"raw_{index}",
                "start": int(start),
                "end": int(end),
                "intention": "raw_chunk",
                "syscalls": seq[start:end],
            }
        )
    return groups


def prepare_samples(
    rows: dict[str, list[dict]],
    max_units: int,
    max_seq_len: int,
    semantic_mode: str,
    cache_path: Path,
    unit_selection: str = "prefix",
) -> tuple[dict[str, list[dict]], SyscallVocab, dict]:
    timings = {}
    train_sequences = [row["api_seq"][:max_seq_len] for row in rows["train"]]

    started = time.perf_counter()
    vocab = SyscallVocab()
    vocab.build_from_samples(train_sequences)
    mapper = IntentionMapper()
    mapper.build(train_sequences, n_groups=max_units)
    timings["vocabulary_and_mapper_seconds"] = time.perf_counter() - started

    extractor = FrozenTemplateBehaviorExtractor(cache_path)
    extraction_seconds = 0.0
    unit_seconds = 0.0
    truncation_seconds = 0.0
    samples = {}
    ordered_splits = ["train", "val", "test"]
    for split in ordered_splits:
        split_rows = rows[split]
        samples[split] = []
        for row in split_rows:
            started = time.perf_counter()
            seq = row["api_seq"][:max_seq_len]
            truncation_seconds += time.perf_counter() - started
            sample_id = str(row.get("sample_id", f"{split}_{len(samples[split])}"))
            started = time.perf_counter()
            elements = extractor.extract_sequence(
                seq, sample_id=sample_id, max_len=max_seq_len
            )
            extraction_seconds += time.perf_counter() - started

            started = time.perf_counter()
            semantic_units = extractor.build_units_from_elements(
                elements,
                sample_id=sample_id,
                max_units=max_units,
                unit_selection=unit_selection,
            )
            if semantic_mode == "llm_template":
                groups = [unit.to_group() for unit in semantic_units]
            elif semantic_mode == "matched_nonsemantic":
                groups = equal_length_groups(seq, len(semantic_units))
            else:
                raise ValueError(f"Unsupported semantic_mode: {semantic_mode}")
            unit_seconds += time.perf_counter() - started
            samples[split].append(
                {
                    "sample_id": sample_id,
                    "family": row.get("family", "unknown"),
                    "groups": groups,
                    "label": int(row["label"] == "malware"),
                    "input_call_count": len(seq),
                    "retained_call_count": sum(
                        len(group.get("syscalls", [])) for group in groups
                    ),
                }
            )
        if split == "train":
            # Persist only train-derived memoization. The template rules are
            # frozen, but this also prevents held-out-family test identifiers
            # from appearing in the reproducibility artifact.
            extractor.save_cache()
            timings["train_cache_entries"] = len(extractor._cache)
            timings["train_cache_bytes"] = (
                cache_path.stat().st_size if cache_path.exists() else 0
            )
    cache_profile_started = time.perf_counter()
    profiled_samples = 0
    for split in ordered_splits:
        for row in rows[split]:
            extractor.extract_sequence(
                row["api_seq"][:max_seq_len],
                sample_id=str(row.get("sample_id", "sample")),
                max_len=max_seq_len,
            )
            profiled_samples += 1
    cache_profile_seconds = time.perf_counter() - cache_profile_started
    timings["semantic_extraction_seconds"] = extraction_seconds
    timings["sequence_truncation_seconds"] = truncation_seconds
    timings["behavior_unit_construction_seconds"] = unit_seconds
    timings["cache_hit_profile_seconds"] = cache_profile_seconds
    timings["cache_hit_profile_ms_per_sample"] = (
        1000.0 * cache_profile_seconds / max(profiled_samples, 1)
    )
    return samples, vocab, timings


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(float(scores.min()), float(scores.max()), 301):
        predictions = (scores >= threshold).astype(np.int64)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)
    return best_threshold, best_f1


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "threshold": float(threshold),
        "tp": int(np.sum((predictions == 1) & (labels == 1))),
        "fp": int(np.sum((predictions == 1) & (labels == 0))),
        "fn": int(np.sum((predictions == 0) & (labels == 1))),
        "tn": int(np.sum((predictions == 0) & (labels == 0))),
    }


def predict(
    model: EsCapturer,
    samples: list[dict],
    device: torch.device,
    collect_view_weights: bool = False,
    objective: str = "dsvdd",
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, list[dict], float]:
    model.eval()
    labels = []
    scores = []
    predictions = []
    weight_buckets = {"c0": [], "c1": []}
    weight_pairs = []
    started = time.perf_counter()
    with torch.no_grad():
        for sample in samples:
            if objective == "dsvdd":
                _, distance = model.forward_dsvdd(sample, device)
                score = -float(distance.item())
            elif objective == "bce":
                logit = model.forward_classifier(sample, device)
                score = float(torch.sigmoid(logit).item())
            else:
                raise ValueError(f"Unsupported objective: {objective}")
            labels.append(sample["label"])
            scores.append(score)
            predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "family": sample["family"],
                    "label": sample["label"],
                    "score": score,
                    "unit_count": len(sample["groups"]),
                    "input_call_count": sample.get("input_call_count"),
                    "retained_call_count": sample.get("retained_call_count"),
                    "retained_call_coverage": (
                        sample.get("retained_call_count", 0)
                        / max(sample.get("input_call_count", 0), 1)
                    ),
                }
            )
            if collect_view_weights and model.fusion.last_view_weights is not None:
                weights = model.fusion.last_view_weights[0, ..., 1].cpu()
                from escapture.escapture_true import build_relation_tensors

                c_matrix, _ = build_relation_tensors(
                    sample["groups"],
                    torch.device("cpu"),
                    prior_mode=model.prior_mode,
                    random_seed=model.random_seed,
                )
                weight_buckets["c0"].extend(weights[c_matrix == 0].tolist())
                weight_buckets["c1"].extend(weights[c_matrix == 1].tolist())
                for i in range(c_matrix.size(0)):
                    for j in range(c_matrix.size(1)):
                        weight_pairs.append(
                            {
                                "sample_id": sample["sample_id"],
                                "family": sample["family"],
                                "label": sample["label"],
                                "i": i,
                                "j": j,
                                "c_ij": int(c_matrix[i, j].item()),
                                "w_graph": float(weights[i, j].item()),
                            }
                        )
    elapsed = time.perf_counter() - started
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        predictions,
        weight_buckets,
        weight_pairs,
        elapsed,
    )


def initialize_center(model: EsCapturer, train_samples: list[dict], device: torch.device) -> None:
    attacks = [sample for sample in train_samples if sample["label"] == 1]
    seeds = attacks[:40] or train_samples[:40]
    model.eval()
    vectors = []
    with torch.no_grad():
        for sample in seeds:
            vectors.append(model.dsvdd(model.encode_sample(sample, device)).squeeze(0))
    center = torch.stack(vectors).mean(dim=0)
    center[center.abs() < 0.01] = 0.01
    model.dsvdd.center.copy_(center)


def make_training_pairs(
    attack_count: int,
    benign_count: int,
    benign_sampling: str,
) -> list[tuple[int, int]]:
    if attack_count < 1 or benign_count < 1:
        raise ValueError("Training requires both attack and benign samples")
    if benign_sampling == "paired-prefix":
        return [
            (int(index), int(index))
            for index in np.random.permutation(min(attack_count, benign_count))
        ]
    if benign_sampling == "epoch-resample":
        benign_order = np.random.permutation(benign_count)
        attack_order = np.random.permutation(attack_count)
        attack_indices = np.resize(attack_order, len(benign_order))
        return [
            (int(attack_index), int(benign_index))
            for attack_index, benign_index in zip(attack_indices, benign_order)
        ]
    raise ValueError(f"Unsupported benign_sampling: {benign_sampling}")


def train(
    model: EsCapturer,
    train_samples: list[dict],
    val_samples: list[dict],
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    lambda_pref: float,
    benign_sampling: str,
    objective: str,
    malware_class_weight: float,
    focal_gamma: float,
    min_delta: float,
    best_state_path: Path,
    stop_file: Path | None,
) -> tuple[list[dict], dict, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5
    )
    if objective == "dsvdd":
        initialize_center(model, train_samples, device)
    attacks = [sample for sample in train_samples if sample["label"] == 1]
    benign = [sample for sample in train_samples if sample["label"] == 0]
    best_f1 = -1.0
    best_state = None
    stale = 0
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        pairs = make_training_pairs(
            len(attacks), len(benign), benign_sampling
        )
        total_loss = 0.0
        total_pref = 0.0
        for attack_index, benign_index in pairs:
            optimizer.zero_grad()
            if objective == "dsvdd":
                _, attack_distance, attack_pref = model.forward_dsvdd(
                    attacks[attack_index], device, return_pref_loss=True
                )
                _, benign_distance, benign_pref = model.forward_dsvdd(
                    benign[benign_index], device, return_pref_loss=True
                )
                attack_loss = attack_distance.mean()
                margin = max(
                    float(attack_distance.detach().mean().item()) + 2.0, 3.0
                )
                benign_loss = F.relu(margin - benign_distance).mean()
            elif objective == "bce":
                attack_logit, attack_pref = model.forward_classifier(
                    attacks[attack_index], device, return_pref_loss=True
                )
                benign_logit, benign_pref = model.forward_classifier(
                    benign[benign_index], device, return_pref_loss=True
                )
                logits = torch.cat(
                    [attack_logit.reshape(-1), benign_logit.reshape(-1)]
                )
                targets = torch.tensor(
                    [1.0, 0.0], dtype=logits.dtype, device=device
                )
                per_sample_bce = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                    pos_weight=torch.tensor(
                        malware_class_weight,
                        dtype=logits.dtype,
                        device=device,
                    ),
                )
                if focal_gamma > 0:
                    probabilities = torch.sigmoid(logits)
                    target_probabilities = (
                        targets * probabilities
                        + (1.0 - targets) * (1.0 - probabilities)
                    )
                    attack_loss = (
                        (1.0 - target_probabilities).pow(focal_gamma)
                        * per_sample_bce
                    ).mean()
                else:
                    attack_loss = per_sample_bce.mean()
                benign_loss = torch.zeros(
                    (), dtype=logits.dtype, device=device
                )
            else:
                raise ValueError(f"Unsupported objective: {objective}")
            preference_loss = 0.5 * (attack_pref + benign_pref)
            loss = attack_loss + benign_loss + lambda_pref * preference_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_pref += float(preference_loss.item())
        scheduler.step()
        val_labels, val_scores, _, _, _, _ = predict(
            model, val_samples, device, objective=objective
        )
        threshold, val_f1 = choose_threshold(val_labels, val_scores)
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(pairs), 1),
            "preference_loss": total_pref / max(len(pairs), 1),
            "val_f1": val_f1,
            "threshold": threshold,
            "training_pairs": len(pairs),
            "unique_benign_seen": len({benign_index for _, benign_index in pairs}),
            "unique_attacks_seen": len({attack_index for attack_index, _ in pairs}),
        }
        history.append(entry)
        print(
            f"[epoch] {epoch}/{epochs} loss={entry['train_loss']:.6f} "
            f"pref={entry['preference_loss']:.6f} val_f1={val_f1:.6f}",
            flush=True,
        )
        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_state_path)
            stale = 0
        else:
            stale += 1
        if stop_file is not None and stop_file.exists():
            print(
                f"[stop] graceful stop requested by {stop_file} after epoch {epoch}",
                flush=True,
            )
            break
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return history, best_state, time.perf_counter() - training_started


def hardware_metadata(device: torch.device) -> dict:
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
    }
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        metadata["gpu"] = "Apple Metal Performance Shaders"
    return metadata


def run_one(
    args: argparse.Namespace,
    rows: dict[str, list[dict]],
    data_load_seconds: float,
    variant_name: str,
    seed: int,
    held_out_family: str | None = None,
) -> dict:
    config = {
        "semantic_mode": "llm_template",
        "use_sequence_view": True,
        "use_graph_view": True,
        "prior_mode": "real",
        "hard_switching": False,
        "use_gating_prior": True,
        "use_structural_bias": True,
        "use_relation_features": True,
        "weighting_mode": "softmax",
        "lambda_pref": args.lambda_pref,
        "beta": args.beta,
        "gating_temperature": args.gating_temperature,
        "max_units": args.max_units,
        "unit_selection": args.unit_selection,
        "benign_sampling": args.benign_sampling,
        "objective": args.objective,
        "malware_class_weight": args.malware_class_weight,
        "focal_gamma": args.focal_gamma,
        **VARIANTS[variant_name],
    }
    run_rows = (
        apply_unknown_family_protocol(rows, held_out_family)
        if held_out_family
        else {split: list(items) for split, items in rows.items()}
    )
    run_rows["train"] = cap_training_rows(
        run_rows["train"], args.max_train_per_class, seed
    )
    set_random_seed(seed)
    device = get_device(args.device)
    run_name = variant_name
    if held_out_family:
        run_name = f"{variant_name}__heldout_{held_out_family}"
    run_dir = Path(args.out_dir) / run_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "experiment": args.experiment,
                "variant": variant_name,
                "held_out_family": held_out_family,
                "seed": seed,
                "epochs": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "embed_dim": args.embed_dim,
                "learning_rate": args.learning_rate,
                "malware_class_weight": args.malware_class_weight,
                "focal_gamma": args.focal_gamma,
                "max_seq_len": args.max_seq_len,
                "limit_per_split": args.limit_per_split,
                "max_train_per_class": args.max_train_per_class,
                "requested_device": args.device,
                "resolved_device": str(device),
                "cache_init": args.cache_init,
                "stop_file": args.stop_file,
                **config,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    cache_path = run_dir / "behavior_template_cache.json"
    if args.cache_init:
        cache_init = Path(args.cache_init)
        if not cache_init.exists():
            raise FileNotFoundError(f"Cache init file does not exist: {cache_init}")
        if cache_path.resolve() != cache_init.resolve():
            shutil.copyfile(cache_init, cache_path)
    elif cache_path.exists():
        cache_path.unlink()

    samples, vocab, preparation = prepare_samples(
        run_rows,
        args.max_units,
        args.max_seq_len,
        config["semantic_mode"],
        cache_path,
        unit_selection=args.unit_selection,
    )
    max_group_length = max(
        len(group["syscalls"])
        for split_samples in samples.values()
        for sample in split_samples
        for group in sample["groups"]
    )
    model = EsCapturer(
        vocab.vocab_size,
        embed_dim=args.embed_dim,
        output_dim=args.embed_dim,
        max_seq_len=min(max_group_length, args.max_seq_len),
        beta=args.beta,
        use_sequence_view=config["use_sequence_view"],
        use_graph_view=config["use_graph_view"],
        prior_mode=config["prior_mode"],
        hard_switching=config["hard_switching"],
        use_gating_prior=config["use_gating_prior"],
        use_structural_bias=config["use_structural_bias"],
        use_relation_features=config["use_relation_features"],
        weighting_mode=config["weighting_mode"],
        gating_temperature=args.gating_temperature,
        random_seed=seed,
    ).to(device)
    model.vocab = vocab
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    graph_started = time.perf_counter()
    graph_count = 0
    for split_samples in samples.values():
        for sample in split_samples:
            for group in sample["groups"]:
                build_intention_graph(group["syscalls"], vocab)
                graph_count += 1
    graph_construction_seconds = time.perf_counter() - graph_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_state_path = run_dir / "best_state.pt"
    stop_file = Path(args.stop_file) if args.stop_file else None
    history, best_state, training_seconds = train(
        model,
        samples["train"],
        samples["val"],
        device,
        args.epochs,
        args.patience,
        args.learning_rate,
        config["lambda_pref"],
        args.benign_sampling,
        args.objective,
        args.malware_class_weight,
        args.focal_gamma,
        args.min_delta,
        best_state_path,
        stop_file,
    )
    checkpoint_path = run_dir / "best_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "vocabulary": vocab.syscall2idx,
            "config": config,
            "seed": seed,
        },
        checkpoint_path,
    )

    val_labels, val_scores, val_predictions, _, _, val_seconds = predict(
        model, samples["val"], device, objective=args.objective
    )
    threshold, _ = choose_threshold(val_labels, val_scores)
    test_labels, test_scores, test_predictions, weights, weight_pairs, test_seconds = predict(
        model,
        samples["test"],
        device,
        collect_view_weights=args.save_view_weights,
        objective=args.objective,
    )
    val_result = metrics(val_labels, val_scores, threshold)
    test_result = metrics(test_labels, test_scores, threshold)
    for row in val_predictions:
        row["prediction"] = int(row["score"] >= threshold)
    for row in test_predictions:
        row["prediction"] = int(row["score"] >= threshold)
    for filename, records in (
        ("validation_predictions.jsonl.gz", val_predictions),
        ("test_predictions.jsonl.gz", test_predictions),
    ):
        with gzip.open(run_dir / filename, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if args.save_view_weights:
        with gzip.open(
            run_dir / "test_view_weights.jsonl.gz", "wt", encoding="utf-8"
        ) as handle:
            for record in weight_pairs:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    efficiency = {
        "data_loading_seconds": data_load_seconds,
        **preparation,
        "training_seconds": training_seconds,
        "epochs_run": len(history),
        "validation_inference_seconds": val_seconds,
        "test_inference_seconds": test_seconds,
        "test_latency_ms_per_sample": 1000.0 * test_seconds / max(len(test_labels), 1),
        "test_throughput_samples_per_second": len(test_labels) / max(test_seconds, 1e-12),
        "graph_construction_seconds": graph_construction_seconds,
        "graphs_constructed": graph_count,
        "peak_rss_mb": peak_rss_mb(),
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else None
        ),
        "parameter_count": parameter_count,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "cache_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
    }
    result = {
        "method": "escapture_dual_view",
        "objective": args.objective,
        "variant": variant_name,
        "held_out_family": held_out_family,
        "seed": seed,
        "status": "ok",
        "data": str(args.data),
        "split_policy": "existing compact JSONL.GZ split field used verbatim",
        "split_counts": {split: len(items) for split, items in samples.items()},
        "config": config,
        "hardware": hardware_metadata(device),
        "history": history,
        "validation": val_result,
        "test": test_result,
        "efficiency": efficiency,
        "checkpoint": str(checkpoint_path),
    }
    if args.save_view_weights:
        result["view_weight_summary"] = {
            key: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "q1": float(np.quantile(values, 0.25)) if values else None,
                "q3": float(np.quantile(values, 0.75)) if values else None,
                "iqr": (
                    float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                    if values
                    else None
                ),
                "std": float(np.std(values)) if values else None,
            }
            for key, values in weights.items()
        }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def aggregate(results: list[dict], keys: tuple[str, ...]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        group_key = tuple(result.get(key) for key in keys)
        grouped[group_key].append(result)
    output = []
    for group_key, group_results in grouped.items():
        entry = {key: value for key, value in zip(keys, group_key)}
        for metric_name in ("accuracy", "precision", "recall", "f1", "auc"):
            values = [item["test"][metric_name] for item in group_results]
            entry[f"{metric_name}_mean"] = float(np.mean(values))
            entry[f"{metric_name}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        entry["seeds"] = [item["seed"] for item in group_results]
        output.append(entry)
    full_by_family = {
        entry.get("held_out_family"): entry
        for entry in output
        if entry.get("variant") == "full"
    }
    for entry in output:
        full = full_by_family.get(entry.get("held_out_family"))
        if full and entry.get("variant") != "full":
            entry["f1_delta_from_full"] = (
                entry["f1_mean"] - full["f1_mean"]
            )
            entry["auc_delta_from_full"] = (
                entry["auc_mean"] - full["auc_mean"]
            )
    return {"runs": len(results), "groups": output}


def main() -> None:
    args = parse_args()
    rows, data_load_seconds = load_rows(Path(args.data), args.limit_per_split)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    if args.experiment == "ablation":
        variants = (
            [value.strip() for value in args.variants.split(",") if value.strip()]
            if args.variants
            else [name for name in VARIANTS if name != "wo_llm_elements"]
        )
        unknown_variants = sorted(set(variants) - set(VARIANTS))
        if unknown_variants:
            raise ValueError(f"Unknown variants: {unknown_variants}")
        families = [None]
    elif args.experiment == "unknown_family":
        variants = ["full"]
        requested = args.held_out_family or ["all"]
        families = malware_families(rows) if "all" in requested else requested
    else:
        variants = [args.variant]
        families = [None]

    results = []
    for family in families:
        for variant in variants:
            for seed in seeds:
                print(
                    f"[run] variant={variant} seed={seed} held_out_family={family}",
                    flush=True,
                )
                results.append(
                    run_one(
                        args,
                        rows,
                        data_load_seconds,
                        variant,
                        seed,
                        held_out_family=family,
                    )
                )
    summary_keys = (
        ("variant", "held_out_family")
        if args.experiment == "unknown_family"
        else ("variant",)
    )
    summary = aggregate(results, summary_keys)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_summary:
        (out_dir / f"{args.experiment}_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
