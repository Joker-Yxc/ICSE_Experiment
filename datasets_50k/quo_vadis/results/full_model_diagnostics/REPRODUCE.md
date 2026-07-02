# Quo Vadis seed-7 full-model repair commands

Run from the workspace root with `escapture/.venv/bin/python`.

## Pre-fix diagnosis

```bash
escapture/.venv/bin/python experiments/reporting/diagnose_full_model.py \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --full-predictions datasets_50k/quo_vadis/results/full_model/full/seed_7/test_predictions.jsonl.gz \
  --svm-predictions datasets_50k/quo_vadis/results/standard_baselines/api_ngram_svm/seed_7/test_predictions.jsonl.gz \
  --full-metrics datasets_50k/quo_vadis/results/full_model/full/seed_7/metrics.json \
  --svm-metrics datasets_50k/quo_vadis/results/standard_baselines/api_ngram_svm/seed_7/metrics.json \
  --max-seq-len 512 --max-units 5 --unit-selection prefix \
  --out-dir datasets_50k/quo_vadis/results/full_model_diagnostics/p2_before_seed_7
```

## P2: complete unit coverage and benign epoch resampling

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --experiment main --variant full \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/full_model_repairs/p2_unit16_uniform_epoch_resample \
  --seeds 7 --epochs 30 --patience 5 --embed-dim 32 \
  --learning-rate 5e-4 --beta 1.0 --lambda-pref 0.1 \
  --max-units 16 --unit-selection uniform-cover \
  --benign-sampling epoch-resample --max-seq-len 512 --device auto
```

## Post-fix diagnosis and comparison

```bash
escapture/.venv/bin/python experiments/reporting/diagnose_full_model.py \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --full-predictions datasets_50k/quo_vadis/results/full_model_repairs/p2_unit16_uniform_epoch_resample/full/seed_7/test_predictions.jsonl.gz \
  --svm-predictions datasets_50k/quo_vadis/results/standard_baselines/api_ngram_svm/seed_7/test_predictions.jsonl.gz \
  --full-metrics datasets_50k/quo_vadis/results/full_model_repairs/p2_unit16_uniform_epoch_resample/full/seed_7/metrics.json \
  --svm-metrics datasets_50k/quo_vadis/results/standard_baselines/api_ngram_svm/seed_7/metrics.json \
  --max-seq-len 512 --max-units 16 --unit-selection uniform-cover \
  --reference-diagnosis datasets_50k/quo_vadis/results/full_model_diagnostics/p2_before_seed_7/diagnosis.json \
  --out-dir datasets_50k/quo_vadis/results/full_model_diagnostics/p2_after_seed_7
```

## P3a: gating regularization

This keeps the P2 data representation and sampling fixes, then changes only
the gating/preference regularization setting requested for P3.

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --experiment main --variant full \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/full_model_repairs/p3a_gating_regularization \
  --seeds 7 --epochs 30 --patience 5 --embed-dim 32 \
  --learning-rate 5e-4 --beta 0.1 --lambda-pref 0.001 \
  --gating-temperature 2.0 \
  --max-units 16 --unit-selection uniform-cover \
  --benign-sampling epoch-resample --max-seq-len 512 \
  --save-view-weights --device auto
```

After P3a completes:

```bash
escapture/.venv/bin/python experiments/reporting/analyze_view_weights.py \
  --inputs datasets_50k/quo_vadis/results/full_model_repairs/p3a_gating_regularization/full/seed_7/test_view_weights.jsonl.gz \
  --dataset quo_vadis \
  --output datasets_50k/quo_vadis/results/full_model_repairs/p3a_gating_regularization/view_weight_analysis_seed_7.json \
  --seed 7
```

## P3b: supervised BCE head

This restores the P2 gating settings and changes only the detection
objective/head. Training uses balanced attack/benign pairs, validation selects
the threshold, and test labels are not used for selection.

```bash
escapture/.venv/bin/python experiments/runners/run_escapture_evaluation.py \
  --experiment main --variant full --objective bce \
  --data datasets_50k/quo_vadis/leakage_safe/quo_vadis_leakage_safe.jsonl.gz \
  --out-dir datasets_50k/quo_vadis/results/full_model_repairs/p3b_supervised_bce \
  --seeds 7 --epochs 30 --patience 5 --embed-dim 32 \
  --learning-rate 5e-4 --beta 1.0 --lambda-pref 0.1 \
  --gating-temperature 1.0 \
  --max-units 16 --unit-selection uniform-cover \
  --benign-sampling epoch-resample --max-seq-len 512 --device auto
```

## P2/P3a/P3b/SVM comparison

```bash
escapture/.venv/bin/python experiments/reporting/compare_full_model_repairs.py \
  --run p2_dsvdd=datasets_50k/quo_vadis/results/full_model_diagnostics/p2_after_seed_7/diagnosis.json \
  --run p3a_dsvdd=datasets_50k/quo_vadis/results/full_model_diagnostics/p3a_seed_7/diagnosis.json \
  --run p3b_bce=datasets_50k/quo_vadis/results/full_model_diagnostics/p3b_seed_7/diagnosis.json \
  --output-json datasets_50k/quo_vadis/results/full_model_repairs/p3_comparison_seed_7.json \
  --output-csv datasets_50k/quo_vadis/results/full_model_repairs/p3_comparison_seed_7.csv
```
