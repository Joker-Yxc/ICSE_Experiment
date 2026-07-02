# DEEPCAPA-adapted

This baseline adapts **DEEPCAPA: Identifying Malicious Capabilities in Windows
Malware**, published at ACSAC 2024 (pages 826-842, DOI
`10.1109/ACSAC63791.2024.00072`). The official repository is
<https://github.com/ucsb-seclab/DeepCapa>; its code is preserved under
`official_code/` at commit `f14f03ce3710c279d8c603d1451559e202fdc3ca`.

## Original task and adaptation

DEEPCAPA is a post-detection malicious capability identification system. It
extracts API sequences from process-memory control-flow graphs, pretrains an API
language model, and fine-tunes a separate binary classifier for each MITRE
ATT&CK technique. It is not originally a benign/malware detector.

The official fine-tuning loader requires per-technique MITRE labels, an
official dataset layout, and a compatible pretraining checkpoint. The compact
datasets used here contain Windows API traces, binary labels, and malware
family labels, but no MITRE capability labels. The official code therefore
cannot be run directly for this task.

This directory provides an explicitly named **DEEPCAPA-adapted fallback
reimplementation**. It treats `api_seq` as the Windows API-call sequence,
augments each API with a deterministic capability-oriented category, and
retains the official model's main structure: API embeddings, positional
encoding, Transformer encoding, sequence attention, CNN feature extraction,
and classification. The original capability head is replaced by a
benign/malware head plus a malware-only family head. Heuristic capability
categories are input features and are not claimed to be MITRE ground truth.

## Data

- Quo Vadis:
  `datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz`
- Zenodo:
  `datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz`

Each dataset contains 50,000 samples: 25,000 benign and 25,000 malware. The
existing `split` field is used verbatim: 35,000 train, 7,500 validation, and
7,500 test samples with seed 7. No sample is repartitioned. Vocabulary,
capability mappings, and family mappings are learned from train only.

## Training

The runner supports a preferred maximum sequence length of 1024 and a 512
fallback. On the available 16 GB Apple M4 system, the full Quo Vadis run at
1024 was terminated by the operating system during the first epoch because of
unified-memory pressure. Final runs therefore use the documented low-memory
configuration `max_length=512` and `batch_size=16`. Training uses at most 30
epochs, early stopping patience 5, and validation binary F1. The validation
split selects the binary threshold, which is then applied unchanged to test.

```bash
PYTHON=python3

"$PYTHON" baselines/05_deepcapa_adapted/train_deepcapa_adapted.py \
  --dataset quo_vadis --max-length 512 --fallback-max-length 512 --batch-size 16
"$PYTHON" baselines/05_deepcapa_adapted/train_deepcapa_adapted.py \
  --dataset zenodo_11079764 --max-length 512 --fallback-max-length 512 --batch-size 16
```

Run both datasets with:

```bash
bash baselines/05_deepcapa_adapted/run_all.sh
```

## Outputs

- `datasets_50k/quo_vadis/results/deepcapa_adapted_metrics.json`
- `datasets_50k/zenodo_11079764/results/deepcapa_adapted_metrics.json`
- One deduplicated `deepcapa_adapted` row in each `baseline_results.csv`
- Logs under `baselines/05_deepcapa_adapted/logs/`
- Encoded arrays and best checkpoints under
  `datasets_50k/<dataset>/artifacts/deepcapa_adapted/`

## Results

Both full runs used the existing splits, seed 7, `max_length=512`,
`batch_size=16`, at most 30 epochs, patience 5, and validation F1 early
stopping. Thresholds were selected on validation and applied unchanged to test.

| Dataset | Epochs | Best epoch | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quo Vadis | 12 | 7 | 0.9649 | 0.9693 | 0.9603 | 0.9648 | 0.9953 |
| Zenodo 11079764 | 27 | 22 | 0.9447 | 0.9364 | 0.9541 | 0.9452 | 0.9863 |

Family classification was evaluated only on the 3,750 malware samples in each
test split.

| Dataset | Families | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro AUC OVR |
|---|---:|---:|---:|---:|---:|---:|
| Quo Vadis | 7 | 0.8773 | 0.8423 | 0.8567 | 0.8440 | 0.9882 |
| Zenodo 11079764 | 124 train families | 0.7173 | 0.3671 | 0.3068 | 0.3113 | 0.9682 |

Zenodo macro AUC includes the 117 family classes having both positive and
negative test examples. Full precision values, training histories, runtime,
configuration, and cache manifests are stored in the metrics JSON files.
