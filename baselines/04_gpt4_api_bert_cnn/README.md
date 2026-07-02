# GPT-4 API Description + BERT + CNN

This directory reproduces the pipeline from **Prompt Engineering-assisted
Malware Dynamic Analysis Using GPT-4**:

- Pei Yan, Shunquan Tan, Miaohui Wang, and Jiwu Huang
- arXiv:2312.08317 (2023)
- Paper: <https://arxiv.org/abs/2312.08317>
- Official repository: <https://github.com/yan-scnu/Prompted_Dynamic_Detection>

The official repository was shallow-cloned into `official_code/`. At the
checked commit (`adb7483`), it contains only `README.md` and `Theme.jpg`; the
authors state that model weights, GPT-generated content, and full training code
are not publicly included. This directory therefore provides an adaptation of
the pipeline described in the paper.

## Data

- Quo Vadis:
  `datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz`
- Zenodo:
  `datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz`

Both files contain 50,000 samples: 25,000 benign and 25,000 malware. The code
strictly reads the existing `split` field and does not resplit the data:

- train: 35,000
- validation: 7,500
- test: 7,500
- recorded split seed: 7

## Adaptation

1. Read each sample's Windows `api_seq`.
2. Collect API names occurring in each training split.
3. Generate label-free API descriptions with the configured DeepSeek API.
4. Cache the union of training API descriptions in
   `datasets_50k/artifacts/gpt4_api_bert_cnn/api_descriptions.json`.
5. Mean-pool `prajjwal1/bert-tiny` token representations into one 128-dimensional
   vector per API description.
6. Convert every API sequence into an embedding sequence of length 512.
7. Train a 1D CNN with kernels 3, 5, and 7 for benign/malware classification.
8. Train an additional family classifier because both datasets provide family
   labels.

The same fixed training configuration is used for both datasets: seed 7, 64
CNN channels, batch size 256, at most 15 epochs, and early-stopping patience 3.
No dataset-specific result tuning was performed. Metrics are Accuracy,
Precision, Recall, F1, and AUC. Family metrics use macro averaging and
one-vs-rest macro AUC.

## DeepSeek Configuration

Create `.env` from `.env.example` and set:

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

`.env` is ignored by Git. Never commit the API key.

## Commands

Quo Vadis:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 escapture/.venv/bin/python \
  baselines/04_gpt4_api_bert_cnn/train_gpt4_api_bert_cnn.py \
  --data-path datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz \
  --dataset-name quo_vadis \
  --metrics-path datasets_50k/quo_vadis/results/gpt4_api_bert_cnn_metrics.json \
  --baseline-csv datasets_50k/quo_vadis/results/baseline_results.csv \
  --log-path baselines/04_gpt4_api_bert_cnn/logs/quo_vadis_train.log \
  --description-source deepseek --encoder bert \
  --encoder-model prajjwal1/bert-tiny --max-len 512 --channels 64 \
  --batch-size 256 --epochs 15 --patience 3 --run-family
```

Zenodo uses the same arguments, replacing the data, dataset name, metrics,
baseline CSV, and log paths with the corresponding
`datasets_50k/zenodo_11079764/` paths.

## Results

Test-set metrics from the completed run:

| Dataset | Task | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| Quo Vadis | detection | 0.9552 | 0.9723 | 0.9371 | 0.9544 | 0.9926 |
| Quo Vadis | family (macro) | 0.8825 | 0.8244 | 0.7811 | 0.7982 | 0.9849 |
| Zenodo | detection | 0.9220 | 0.9135 | 0.9323 | 0.9228 | 0.9737 |
| Zenodo | family (macro) | 0.7355 | 0.1817 | 0.1163 | 0.1182 | 0.9182 |

Full results:

- `datasets_50k/quo_vadis/results/gpt4_api_bert_cnn_metrics.json`
- `datasets_50k/zenodo_11079764/results/gpt4_api_bert_cnn_metrics.json`
- `datasets_50k/quo_vadis/results/baseline_results.csv`
- `datasets_50k/zenodo_11079764/results/baseline_results.csv`

Run logs are under `baselines/04_gpt4_api_bert_cnn/logs/`. Best checkpoints
are under `datasets_50k/<dataset>/artifacts/gpt4_api_bert_cnn/`.
