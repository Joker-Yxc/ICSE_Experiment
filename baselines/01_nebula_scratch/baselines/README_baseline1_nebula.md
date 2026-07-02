# Baseline 1: Nebula-Scratch

Nebula is used as the first Transformer/self-attention dynamic malware behavior baseline.
For the main comparison, only the from-scratch version is retained.

## Dataset

Local dataset:

```text
../../datasets_50k/quo_vadis/data/raw
```

The existing `escapture` Quo Vadis data can be used directly. Its JSON files are Speakeasy dynamic behavior reports and match Nebula's expected input format.

Detection labels:

- `report_clean`, `report_windows_syswow64`: benign
- all other `report_*` folders: malware

## Protocol

This run matches the EsCapturer experiment scale:

- 500 malware reports + 500 benign reports
- split: 70/15/15
- train: 700
- validation: 150
- test: 150
- epochs: 30
- batch size: 32
- input length: 512

Nebula's BPE tokenizer is reused for report encoding, but the Transformer model weights are randomly initialized. Released pretrained Nebula model weights are not used.

## Reproduction

Run from the workspace root:

```bash
escapture/.venv/bin/python baselines/01_nebula_scratch/baselines/run_nebula_quovadis_baseline.py \
  --max-attack-samples 25000 \
  --max-normal-samples 25000 \
  --epochs 30 \
  --batch-size 32 \
  --out-dir baselines/01_nebula_scratch/results/main_50k
```

## Result

Test set:

| Method | Init | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Nebula-Scratch | random | 0.8267 | 0.7805 | 0.8889 | 0.8312 | 0.9103 |

Validation set:

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Nebula-Scratch | 0.8400 | 0.7907 | 0.9189 | 0.8500 | 0.9324 |

Outputs retained:

```text
baselines/01_nebula_scratch/results/main_50k/metrics.json
baselines/01_nebula_scratch/results/main_50k/detection_predictions.csv
baselines/01_nebula_scratch/results/main_50k/nebula_scratch_detection.pt
```
