# EsCapturer Experimental Artifact

This repository contains the code, compact datasets, baselines, evaluation
scripts, results, and paper materials for the Windows API / system-level call
trace malware-detection experiments.

## Repository layout

| Path | Purpose |
| --- | --- |
| `escapture/` | EsCapturer implementation, examples, and unit tests |
| `experiments/data_preparation/` | Dataset download and protocol preparation |
| `experiments/runners/` | Main, baseline, ablation, and robustness runners |
| `experiments/tuning/` | Hyperparameter-search scripts |
| `experiments/reporting/` | Evidence collection, analysis, and plotting |
| `baselines/` | Upstream or adapted baseline implementations |
| `datasets_50k/` | Compact datasets, manifests, metrics, and experiment artifacts |
| `tests/` | Repository-level regression tests |
| `docs/` | Paper sources, experiment notes, and figures |
| `output/` | Final paper PDF and packaged Overleaf export |

Generated caches, local virtual environments, credentials, logs, smoke runs,
and downloaded raw source reports are intentionally excluded from Git.

## Environment

Python 3.12 is the currently tested environment for the main implementation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r escapture/requirements.txt
```

Individual baselines have their own `requirements.txt` files because several
upstream implementations require different dependency versions.

## Data

Original datasets:

- [Zenodo 11079764](https://zenodo.org/records/11079764)
- [Quo Vadis Malware Emulation](https://www.kaggle.com/datasets/dmitrijstrizna/quo-vadis-malware-emulation)

The repository keeps the compact, split-aware datasets used by the experiment:

- `datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz`
- `datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz`

Downloaded raw reports are excluded because they are large, externally
available source data. Preparation scripts are in
`experiments/data_preparation/`; dataset schema and statistics are documented
in `datasets_50k/README.md` and `docs/README_experiment_setting.md`.

## Reproducing the evaluation

Protocol-specific commands are documented in `experiments/README.md`. The main
leakage-safe evaluation entry point is:

```bash
python experiments/runners/run_escapture_evaluation.py \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/full_model \
  --experiment main --variant full --seeds 7,17,27 --save-view-weights
```

Run the automated checks from the repository root:

```bash
python -m unittest discover -s tests
python -m unittest discover -s escapture/tests
```

## Results and paper

Consolidated result tables are under `datasets_50k/`. Confirmatory per-run
artifacts remain under each dataset's `results/` directory. Paper sources and
figures are under `docs/`; the final rendered PDF is under `output/pdf/`.

## Baseline provenance

Folders named `official_code/` contain imported upstream implementations.
Their included README and license files remain authoritative. Local adaptation
scripts and experiment notes are stored alongside them.
