# Baseline 2: API2Vec++ on EsCapture-Aligned Reports

API2Vec++ is used as the second API-sequence representation baseline.

Upstream repository:

```text
https://github.com/yyyjn/API2VecPlus
```

## Dataset And Protocol

Local dataset:

```text
../../datasets_50k/quo_vadis/data/raw
```

The experiment follows the same binary detection protocol used for baseline 1:

- 500 malware reports + 500 benign reports
- split: 70/15/15
- train: 700
- validation: 150
- test: 150
- epochs: 30
- batch size: 32
- seed: 7

Labels:

- `report_clean`, `report_windows_syswow64`: benign
- all other `report_*` folders: malware

## Adaptation Note

The upstream API2Vec++ code expects XLSX traces with `pid`, `category`, `args`, and `return` columns. The local EsCapture data is JSON with entrypoint-level `apis` lists. The runner therefore keeps the API2Vec++ training shape while adapting the graph/path stage to the available fields:

```text
EsCapture JSON API sequence
-> API2Vec++-style path corpus
-> BPE tokenizer + small RoBERTa MLM
-> TextCNN binary classifier
```

The upstream source is kept under:

```text
baselines/02_api2vecpp/API2VecPlus
```

The local reproducible runner is:

```text
baselines/02_api2vecpp/run_api2vecpp_escapture_baseline.py
```

## Reproduction Command

Run from the workspace root:

```bash
escapture/.venv/bin/python baselines/02_api2vecpp/run_api2vecpp_escapture_baseline.py \
  --max-attack-samples 25000 \
  --max-normal-samples 25000 \
  --epochs 30 \
  --mlm-epochs 2 \
  --batch-size 32 \
  --mlm-batch-size 32 \
  --lr 3e-4 \
  --out-dir baselines/02_api2vecpp/results/main_50k \
  --rebuild-cache \
  --device auto
```

## Final Result

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9195 | 0.8954 | 0.9499 | 0.9218 | 0.9806 |
| Test | 0.9265 | 0.9083 | 0.9488 | 0.9281 | 0.9824 |

Runtime: 37.119 seconds in the compact 50k runner.

Outputs:

```text
baselines/02_api2vecpp/results/main_50k/metrics.json
baselines/02_api2vecpp/results/main_50k/detection_predictions.csv
baselines/02_api2vecpp/results/main_50k/api2vecpp_textcnn_classifier.pt
baselines/02_api2vecpp/results/main_50k/api2vecpp_mlm/
baselines/02_api2vecpp/results/main_50k/tokenizer/
```
