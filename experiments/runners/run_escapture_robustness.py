#!/usr/bin/env python3
"""Evaluate a clean-trained EsCapturer checkpoint on perturbed test traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch

from escapture.escapture_true import EsCapturer, SyscallVocab, set_random_seed
from escapture.llm_behavior_extractor import FrozenTemplateBehaviorExtractor
from experiments.runners.run_escapture_evaluation import get_device, metrics, predict


PERTURBATIONS = ("insertion", "deletion", "local_reordering")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--full-artifact", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--intensities", default="0.1,0.2,0.3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, list[dict]]:
    rows = {"train": [], "val": [], "test": []}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["split"]].append(row)
    return rows


def stable_rng(seed: int, sample_id: str, perturbation: str, intensity: float) -> np.random.Generator:
    payload = f"{seed}|{sample_id}|{perturbation}|{intensity:.6f}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


def perturb_sequence(
    sequence: list[str],
    perturbation: str,
    intensity: float,
    rng: np.random.Generator,
    benign_calls: list[str],
    benign_probabilities: np.ndarray,
) -> list[str]:
    seq = list(sequence)
    if not seq:
        return seq
    if perturbation == "insertion":
        count = max(1, int(round(len(seq) * intensity)))
        inserted = rng.choice(benign_calls, size=count, p=benign_probabilities).tolist()
        positions = sorted(rng.integers(0, len(seq) + 1, size=count).tolist())
        output = []
        previous = 0
        for position, call in zip(positions, inserted):
            output.extend(seq[previous:position])
            output.append(call)
            previous = position
        output.extend(seq[previous:])
        return output
    if perturbation == "deletion":
        count = min(len(seq) - 1, max(1, int(round(len(seq) * intensity))))
        removed = set(rng.choice(len(seq), size=count, replace=False).tolist())
        return [call for index, call in enumerate(seq) if index not in removed]
    if perturbation == "local_reordering":
        chunks = [seq[start : start + 5] for start in range(0, len(seq), 5)]
        eligible = [index for index, chunk in enumerate(chunks) if len(chunk) > 1]
        count = min(len(eligible), max(1, int(round(len(eligible) * intensity))))
        selected = rng.choice(eligible, size=count, replace=False).tolist()
        for index in selected:
            original = chunks[index]
            reordered = list(rng.permutation(original))
            if reordered == original:
                reordered = original[1:] + original[:1]
            chunks[index] = reordered
        return [call for chunk in chunks for call in chunk]
    raise ValueError(f"Unsupported perturbation: {perturbation}")


def make_vocab(mapping: dict[str, int]) -> SyscallVocab:
    vocab = SyscallVocab()
    vocab.syscall2idx = dict(mapping)
    vocab.idx2syscall = {index: call for call, index in mapping.items()}
    vocab.vocab_size = len(mapping)
    return vocab


def make_model(checkpoint: dict, device: torch.device) -> EsCapturer:
    config = checkpoint["config"]
    state = checkpoint["state_dict"]
    embed_dim = int(state["seq_encoder.embedding.weight"].shape[1])
    sequence_length = int(state["seq_encoder.pos_encoding"].shape[1])
    model = EsCapturer(
        len(checkpoint["vocabulary"]),
        embed_dim=embed_dim,
        output_dim=embed_dim,
        max_seq_len=sequence_length,
        beta=config["beta"],
        use_sequence_view=config["use_sequence_view"],
        use_graph_view=config["use_graph_view"],
        prior_mode=config["prior_mode"],
        hard_switching=config["hard_switching"],
        use_gating_prior=config["use_gating_prior"],
        use_structural_bias=config["use_structural_bias"],
        use_relation_features=config["use_relation_features"],
        weighting_mode=config["weighting_mode"],
        gating_temperature=config["gating_temperature"],
        random_seed=checkpoint["seed"],
    ).to(device)
    model.vocab = make_vocab(checkpoint["vocabulary"])
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_test_samples(
    rows: list[dict],
    extractor: FrozenTemplateBehaviorExtractor,
    max_seq_len: int,
    max_group_length: int,
    max_units: int,
    unit_selection: str,
) -> list[dict]:
    samples = []
    model_call_capacity = max_group_length * max_units
    for row in rows:
        sequence = row["api_seq"][:max_seq_len]
        if len(sequence) > model_call_capacity:
            retained = np.linspace(0, len(sequence) - 1, model_call_capacity, dtype=int)
            sequence = [sequence[index] for index in retained]
        sample_id = str(row["sample_id"])
        elements = extractor.extract_sequence(sequence, sample_id=sample_id, max_len=max_seq_len)
        units = extractor.build_units_from_elements(
            elements,
            sample_id=sample_id,
            max_unit_len=max_group_length,
            max_units=max_units,
            unit_selection=unit_selection,
        )
        groups = [unit.to_group() for unit in units]
        samples.append(
            {
                "sample_id": sample_id,
                "family": row.get("family", "unknown"),
                "groups": groups,
                "label": int(row["label"] == "malware"),
                "input_call_count": len(sequence),
                "retained_call_count": sum(len(group["syscalls"]) for group in groups),
            }
        )
    return samples


def write_jsonl_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    data_path = Path(args.data)
    full_dir = Path(args.full_artifact)
    output_root = Path(args.out_dir)
    rows = load_rows(data_path)
    expected_counts = {"train": 35000, "val": 7500, "test": 7500}
    actual_counts = {split: len(values) for split, values in rows.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected split counts: {actual_counts}")

    full_metrics = json.loads((full_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(full_metrics["validation"]["threshold"])
    checkpoint = torch.load(full_dir / "best_model.pt", map_location="cpu", weights_only=False)
    device = get_device(args.device)
    model = make_model(checkpoint, device)
    config = checkpoint["config"]
    max_group_length = int(checkpoint["state_dict"]["seq_encoder.pos_encoding"].shape[1])
    max_seq_len = int(json.loads((full_dir / "config.json").read_text())["max_seq_len"])
    cache_path = full_dir / "behavior_template_cache.json"
    extractor = FrozenTemplateBehaviorExtractor(cache_path)

    benign_counter = Counter(
        call
        for row in rows["train"]
        if row["label"] == "benign"
        for call in row["api_seq"][:max_seq_len]
    )
    benign_calls = sorted(benign_counter)
    benign_probabilities = np.asarray([benign_counter[call] for call in benign_calls], dtype=float)
    benign_probabilities /= benign_probabilities.sum()
    intensities = [float(value) for value in args.intensities.split(",") if value.strip()]
    baselines = {
        "quo_vadis": {"f1": 0.9626, "auc": 0.9935},
        "zenodo_11079764": {"f1": 0.9409, "auc": 0.9791},
    }
    baseline = baselines[args.dataset]
    summary = []

    for perturbation in PERTURBATIONS:
        for intensity in intensities:
            label = f"intensity_{int(round(intensity * 100)):02d}"
            run_dir = output_root / perturbation / label
            run_dir.mkdir(parents=True, exist_ok=True)
            perturbed_rows = []
            for row in rows["test"]:
                rng = stable_rng(args.seed, str(row["sample_id"]), perturbation, intensity)
                perturbed = dict(row)
                perturbed["api_seq"] = perturb_sequence(
                    row["api_seq"],
                    perturbation,
                    intensity,
                    rng,
                    benign_calls,
                    benign_probabilities,
                )
                perturbed["perturbation"] = perturbation
                perturbed["intensity"] = intensity
                perturbed_rows.append(perturbed)
            write_jsonl_gz(run_dir / "perturbed_test.jsonl.gz", perturbed_rows)

            started = time.perf_counter()
            samples = prepare_test_samples(
                perturbed_rows,
                extractor,
                max_seq_len,
                max_group_length,
                config["max_units"],
                config["unit_selection"],
            )
            labels, scores, predictions, _, _, inference_seconds = predict(
                model, samples, device, objective="bce"
            )
            result = metrics(labels, scores, threshold)
            result["delta_f1"] = result["f1"] - baseline["f1"]
            result["delta_auc"] = result["auc"] - baseline["auc"]
            for prediction in predictions:
                prediction["prediction"] = int(prediction["score"] >= threshold)
                prediction["perturbation"] = perturbation
                prediction["intensity"] = intensity
            write_jsonl_gz(run_dir / "test_predictions.jsonl.gz", predictions)

            run_config = {
                "dataset": args.dataset,
                "data": str(data_path),
                "full_artifact": str(full_dir),
                "checkpoint": str(full_dir / "best_model.pt"),
                "cache": str(cache_path),
                "seed": args.seed,
                "split_counts": actual_counts,
                "perturbation": perturbation,
                "intensity": intensity,
                "threshold": threshold,
                "threshold_source": "clean validation",
                "checkpoint_source": "clean validation",
                "objective": "bce",
                "device": str(device),
                "training_performed": False,
                "local_reordering_window": 5 if perturbation == "local_reordering" else None,
                "model_call_capacity": max_group_length * config["max_units"],
                "capacity_selection": "uniform-cover",
            }
            (run_dir / "config.json").write_text(
                json.dumps(run_config, indent=2), encoding="utf-8"
            )
            payload = {
                **run_config,
                "prediction_count": len(predictions),
                "metrics": result,
                "preparation_and_inference_seconds": time.perf_counter() - started,
                "inference_seconds": inference_seconds,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            entry = {
                "Dataset": args.dataset,
                "Perturbation": perturbation,
                "Intensity": intensity,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "AUC": result["auc"],
                "Delta_F1": result["delta_f1"],
                "Delta_AUC": result["delta_auc"],
                "Prediction_count": len(predictions),
                "Artifact_path": str(run_dir),
            }
            summary.append(entry)
            print(json.dumps(entry), flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary_seed7.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_root / "summary_seed7.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
