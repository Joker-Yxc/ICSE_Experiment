# Experiment Scripts

This folder contains experiment orchestration scripts. The method
implementation itself is under `escapture/`, baseline implementations are under
`baselines/`, and compact 50k datasets/results are under `datasets_50k/`.

## Layout

- `data_preparation/`
  - source download, sequence conversion, compact dataset construction scripts
- `runners/`
  - shared experiment runners for our method and baselines
- `tuning/`
  - tuning/ablation scripts for our method

Run scripts from the workspace root, for example:

```bash
python experiments/runners/run_compact_baselines.py
```

## ICSE confirmatory protocol

Files directly under `datasets_50k/<dataset>/results/` were produced from the
preliminary embedded binary-stratified split. They are diagnostic artifacts and
must not be copied into confirmatory RQ tables.

Generate the leakage-safe datasets:

```bash
python3 experiments/data_preparation/prepare_leakage_safe_splits.py \
  --data datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/leakage_safe --dataset quo_vadis

python3 experiments/data_preparation/prepare_leakage_safe_splits.py \
  --data datasets_50k/zenodo_11079764/data/zenodo_main_50k.jsonl.gz \
  --out-dir datasets_50k/zenodo_11079764/leakage_safe \
  --dataset zenodo_11079764
```

Run the standard simple and sequence baselines once per seed:

```bash
for seed in 7 17 27; do
  escapture/.venv/bin/python experiments/runners/run_sequence_baselines.py \
    --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
    --out-dir datasets_50k/quo_vadis/results/standard_baselines \
    --models all --seed "$seed"
done
```

Use the analogous Zenodo path for the second dataset. XGBoost variants require
the `xgboost` package; the runner fails explicitly if it is unavailable.

Run the graph controls once per seed:

```bash
for seed in 7 17 27; do
  escapture/.venv/bin/python experiments/runners/run_graph_baselines.py \
    --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
    --out-dir datasets_50k/quo_vadis/results/graph_baselines \
    --models gat,rgcn --seed "$seed"
done
```

Run EsCapturer-full once per dataset:

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/full_model \
  --experiment main --variant full --seeds 7,17,27 --save-view-weights
```

Run the graph controls once per seed:

```bash
for seed in 7 17 27; do
  escapture/.venv/bin/python experiments/runners/run_graph_baselines.py \
    --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
    --out-dir datasets_50k/quo_vadis/results/graph_baselines \
    --models gat,rgcn --seed "$seed"
done
```

Run all ten ablations:

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/ablation \
  --experiment ablation --seeds 7,17,27 --save-view-weights
```

Generate fixed family-disjoint folds:

```bash
python3 experiments/data_preparation/prepare_family_disjoint_protocol.py \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/family_disjoint \
  --dataset quo_vadis --min-groups 1

python3 experiments/data_preparation/prepare_family_disjoint_protocol.py \
  --data datasets_50k/zenodo_11079764/leakage_safe/zenodo_11079764_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/zenodo_11079764/family_disjoint \
  --dataset zenodo_11079764 --min-groups 20 --top-n 10
```

The generated Zenodo manifest currently selects nine eligible families because
only nine meet the fixed 20-SHA criterion.

## Full dual-view evaluation

`run_compact_baselines.py` retains the existing TF-IDF classifier result for
historical comparison. The complete sequence/graph model and its ICSE
evaluation protocol use:

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --data datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/full_model \
  --experiment main --variant full --seeds 7,17,27
```

Run all required ablations:

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --data datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/ablation \
  --experiment ablation --seeds 7,17,27 --save-view-weights
```

Run the seven Quo Vadis leave-one-family-out experiments:

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --data datasets_50k/quo_vadis/data/quo_vadis_main_50k.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/unknown_family \
  --experiment unknown_family --held-out-family all --seeds 7,17,27
```

Generate identical leave-one-family-out input files for external baseline
runners:

```bash
python3 experiments/data_preparation/prepare_unknown_family_protocol.py
```

The generated manifest records the exact train, validation, benign-test, and
held-out-malware counts for each family. Pass the corresponding `.jsonl.gz`
file to the full-model runner, `run_compact_baselines.py`, DawnGNN's
`--compact_data`, or DeepCapa-adapted's `--data-path`. Use family-specific
output/cache directories so checkpoints from different held-out families
cannot be reused accidentally.

The runner strictly uses the embedded train/validation/test split, selects the
decision threshold on validation only, saves the best checkpoint and
per-sample scores, and records stage-level timing and memory. Use
`--limit-per-split` and `--max-train-per-class` only for smoke tests; results
from capped runs must not be placed in paper tables.
