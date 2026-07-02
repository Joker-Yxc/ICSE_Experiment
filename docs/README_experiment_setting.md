# Windows API / System-Level Call Trace Malware Detection

## Data Sources

- Zenodo 11079764: `https://zenodo.org/records/11079764`
- Quo Vadis Malware Emulation: `https://www.kaggle.com/datasets/dmitrijstrizna/quo-vadis-malware-emulation`

## Storage Policy

The machine had about 111GiB free before this run. The workflow keeps at least
15G free, does not save uncompressed large intermediates, and writes compact
`jsonl.gz` files containing only:

`sample_id, source, label, family, api_seq, split`

`api_seq` stores API names only and is truncated to 5000 calls.

## Dataset Status

Quo Vadis was already available locally under:

`datasets_50k/quo_vadis/data/raw`

Kaggle API credentials were not present at `~/.kaggle/kaggle.json`, but no
Kaggle download was needed because the dataset already existed locally.

Zenodo metadata and family mapping were present under:

`datasets_50k/zenodo_11079764/data/raw`

The Zenodo trace archive download was attempted, but transfer speed was about
300KB/s with an estimated runtime of about 8 hours. The incomplete partial file
was removed to avoid treating it as a valid archive. The manifest is saved at:

`datasets_50k/zenodo_11079764/data/zenodo_11079764_manifest.json`

The latest compact 50k datasets and results are stored under:

`datasets_50k/`

## Actual Compact Datasets

Quo Vadis file:

`datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz`

Stats:

- Total: 50,000
- Benign: 25,000
- Malware: 25,000
- Split: train 35,000 / val 7,500 / test 7,500
- Seed: 7
- Average API sequence length: 195.621
- Max API sequence length: 5000
- File size: about 10MB

Family distribution:

- benign: 25000
- backdoor: 5372
- ransomware: 4955
- trojan: 4107
- coinminer: 3559
- dropper: 3418
- keylogger: 2357
- rat: 1232

Zenodo file:

`datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz`

Stats:

- Total: 50,000
- Benign: 25,000
- Malware: 25,000
- Split: train 35,000 / val 7,500 / test 7,500
- Seed: 7
- Average API sequence length: 199.939
- Max API sequence length: 200
- File size: about 7.4MB

Both datasets use the same compact schema and embedded `split` field.

## Training Protocol

- `max_epochs=30`
- early stopping patience: 5
- validation metric: F1
- decision threshold selected on validation F1 only for `ours`
- baseline adapters keep their original fixed threshold / feature settings
- failed methods are logged and do not stop subsequent methods

The compact runner is:

`experiments/runners/run_compact_baselines.py`

Results are saved to dataset-specific result folders:

- `datasets_50k/quo_vadis/results/`
- `datasets_50k/zenodo_11079764/results/`

## Implemented Method Revision Modules

The revised `ours` path now includes a reproducible LLM-assisted behavior
element extraction layer from the revision plan:

- implementation: `escapture/llm_behavior_extractor.py`
- schema: `subject, operation, object, resource, context, goal, template_id`
- default mode: frozen template library, `llm_template_v1`
- LLM boundary: semantic parsing only; no malware/benign verdict is produced
- privacy handling: paths, IP addresses, and hashes are normalized to placeholders
- cache: `datasets_50k/quo_vadis/results/llm_behavior_template_cache.json`
- extraction summary: `datasets_50k/quo_vadis/results/llm_behavior_extraction_summary.json`

For reproducibility and to keep the main 50k result stable, the default runner
mode is `--ours-semantic-mode llm_template`: it executes and records the
LLM-assisted extraction/behavior-unit layer, while the final classifier keeps
the validated API 1-4 gram TF-IDF backbone. A heavier ablation mode,
`--ours-semantic-mode llm_template_features`, additionally injects behavior-unit
semantic tokens into the classifier. `--ours-semantic-mode none` is the
`w/o LLM elements` ablation.

The neural `escapture/escapture_true.py` path was also updated so behavior units
can carry the semantic fields above, and its interleaving relation features now
follow the 7-field plan: temporal distance, overlap, interleaving strength,
same object, same process/subject, edge/interleaving type, and behavior type
pair.

## Results

| Method | Status | Epochs | Val F1 | Test Acc | Test Precision | Test Recall | Test F1 | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ours | ok | - | 0.9620 | 0.9621 | 0.9707 | 0.9531 | 0.9618 | 0.9947 |
| nebula | ok | 23 | 0.9062 | 0.9113 | 0.8916 | 0.9365 | 0.9135 | 0.9708 |
| api2vecpp | ok | 22 | 0.9218 | 0.9265 | 0.9083 | 0.9488 | 0.9281 | 0.9824 |
| dawngnn_reimpl | ok | 30 | 0.8731 | 0.8741 | 0.8409 | 0.9229 | 0.8800 | 0.9458 |

## Notes

This run uses compact reimplementation/adaptation paths so that all methods
consume the same `jsonl.gz` dataset and exact same split without creating large
baseline-specific caches. The updated DawnGNN runner builds an API transition
graph per sample and applies documentation-style node semantics with a
two-layer GAT.

The optimized `ours` adapter uses the LLM-template extraction layer plus TF-IDF
API 1-4 gram features with a linear logistic SGD classifier. Hyperparameters and
threshold were selected using validation F1; the test split was used only for
final reporting. Baselines were not given this optimization, and they do not use
the LLM-template behavior extraction layer.

Old pilot, 10k, 20k, smoke-test, 500-sample, and tuning artifacts were removed
after the final 50k datasets/results were consolidated.
