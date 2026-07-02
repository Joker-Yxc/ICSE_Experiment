# MME Baseline

## Paper and official repository

- Paper: *Mitigating the Impact of Malware Evolution on API Sequence-Based Windows Malware Detectors*
- Authors: Xingyuan Wei, Ce Li, Qiujian Lv, Ning Li, Degang Sun, and Yan Wang
- Venue: IEEE Transactions on Information Forensics and Security, 2026, volume 21, pages 45-60
- DOI: `10.1109/TIFS.2025.3637727`
- Official repository: <https://github.com/XingYuanWei/MME>
- Local official checkout: `official_code/` at commit `8a055f0`

MME is a dynamic Windows API sequence malware detector. It is not a GPT/LLM
method. Its main ingredients are API knowledge enhancement, contrastive
learning, and a downstream sequence detector such as TextCNN or LSTM.

## Official-code availability and fallback status

As checked for this reproduction, the official repository contains only a
README. It documents precomputed `.npz` embeddings and gives Baidu Netdisk
links for the dataset and MME Windows API knowledge graph, but it does not
publish model training source. The official knowledge graph and system resource
encoding are not present in the cloned repository.

This directory therefore uses an explicitly marked **fallback reproduction**:

1. Build an API transition/co-occurrence graph from the existing **training
   split only**.
2. Build resource features from API names and heuristic Windows resource
   categories (file, registry, process, memory, network, and related groups).
3. Smooth resource features over graph neighbors to initialize API embeddings.
4. Train an MME-TextCNN-style detector with binary classification, supervised
   contrastive loss, and an auxiliary malware-family classification head.

This is a paper-structure-oriented reimplementation, not unpublished official
MME source code. Generated fallback files are stored under `data/<dataset>/`.

## Data and adaptation

- Quo Vadis:
  `datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz`
- Zenodo:
  `datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz`

The runner reads `api_seq` and strictly honors each sample's existing `split`
field. It never repartitions the data. The conversion stage writes compressed
`train.npz`, `val.npz`, and `test.npz` files with the official README's core
field convention: `x`, `y`, and `y_family`. It also retains `sample_id`.
Vocabulary, graph, and family mappings are learned from train only.

## Run

The tested environment is the existing project environment:

```bash
escapture/.venv/bin/python baselines/07_mme/run_mme.py --dataset quo_vadis
escapture/.venv/bin/python baselines/07_mme/run_mme.py --dataset zenodo_11079764
```

Useful smoke test:

```bash
escapture/.venv/bin/python baselines/07_mme/run_mme.py \
  --dataset quo_vadis --limit-per-split 256 --epochs 1 --rebuild-cache --skip-csv
```

## Outputs

- Quo Vadis metrics:
  `datasets_50k/quo_vadis/results/mme_metrics.json`
- Zenodo metrics:
  `datasets_50k/zenodo_11079764/results/mme_metrics.json`
- Aggregate CSV files:
  `datasets_50k/<dataset>/results/baseline_results.csv`
- Logs:
  `baselines/07_mme/logs/`
- Converted arrays, fallback graph, resource embeddings, and checkpoints:
  `datasets_50k/<dataset>/artifacts/mme/`

## Results

All binary thresholds were selected on validation F1 and then applied unchanged
to the test split.

| Dataset | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Quo Vadis | 0.9380 | 0.9252 | 0.9531 | 0.9389 | 0.9890 |
| Zenodo 11079764 | 0.9279 | 0.9184 | 0.9392 | 0.9287 | 0.9789 |

Family classification is evaluated on malware samples only. Detailed family
accuracy, macro precision/recall/F1, and one-vs-rest macro AUC are included in
each `mme_metrics.json`. For macro AUC, only classes having both positive and
negative examples in that evaluation split are included.
