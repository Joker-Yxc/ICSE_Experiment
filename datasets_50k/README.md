# Main 50k Experiment Data and Results

This folder keeps only the latest 50,000-sample compact datasets and their
corresponding experiment results.

## Layout

- `quo_vadis/data/`
  - `quo_vadis_main_50k.jsonl.gz`
  - `quo_vadis_main_50k.jsonl.gz.stats.json`
  - `raw/`: locally retained Quo Vadis source reports
- `quo_vadis/results/`
  - `baseline_results.csv`
  - per-method metrics JSON files
  - LLM-assisted behavior extraction summary/cache for `ours`
- `quo_vadis/artifacts/`
  - baseline-specific encoded caches and checkpoints
- `zenodo_11079764/data/`
  - `zenodo_main_50k.jsonl.gz`
  - `zenodo_main_50k.jsonl.gz.stats.json`
  - `zenodo_11079764_manifest.json`
  - `raw/`: Zenodo record metadata and family mapping
- `zenodo_11079764/results/`
  - `baseline_results.csv`
  - per-method metrics JSON files
- `zenodo_11079764/artifacts/`
  - baseline-specific encoded caches and checkpoints
- `main_50k_all_methods_two_datasets.md`
  - complete cross-dataset result summary for all eight methods
- `all_method_results.csv`
  - normalized binary-detection results for both datasets
- `family_classification_results.csv`
  - family-classification results reported by the four additional methods
- `artifacts/`
  - cross-dataset shared caches such as generated API descriptions

Both compact datasets contain:

`sample_id, source, label, family, api_seq, split`

Both use the same split policy:

- 25,000 benign + 25,000 malware
- train/val/test = 35,000 / 7,500 / 7,500
- seed = 7

Shared experiment datasets and source data must stay under this directory.
Method folders such as `escapture/` contain code only.
