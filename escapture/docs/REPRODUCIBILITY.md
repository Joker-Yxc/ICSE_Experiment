# Reproducibility Guide

## Environment

- Python: 3.10--3.12
- Core dependencies: NumPy, PyTorch, and scikit-learn
- Deterministic seed used by the example and default paper protocol: `7`

Create the environment with:

```bash
./setup_env.sh
source .venv/bin/activate
python -m pip freeze > environment.lock.txt
```

Keep `environment.lock.txt` with a tagged artifact release when exact package
versions have been validated on the target server.

## Input formats

The standalone command accepts either:

1. `--attack` and `--normal` text files, with one whitespace-separated call
   sequence per line; or
2. one or more `--dataset_root` paths whose directory/file names identify
   benign/normal and malware/attack samples.

Example:

```bash
python escapture.py \
  --attack /path/to/attack_seq.txt \
  --normal /path/to/normal_seq.txt \
  --epochs 30 \
  --embed_dim 32 \
  --lr 5e-4 \
  --n_groups 5 \
  --beta 1.0 \
  --lambda_pref 0.1 \
  --semantic_extractor template \
  --seed 7 \
  --output results/seed_7.json
```

## Minimum reporting record

For each table or figure, archive:

- the Git commit and release tag;
- source dataset version, provider URL, and SHA-256 checksum;
- preprocessing command and split manifest;
- every random seed and full command line;
- CPU/GPU model, memory, operating system, and runtime;
- Python, CUDA, PyTorch, NumPy, and scikit-learn versions;
- validation-only threshold-selection procedure;
- per-sample test scores, aggregate metrics, and confidence intervals; and
- any deviation from the paper configuration.

Use at least three predetermined seeds for stochastic experiments. Do not tune
hyperparameters or select a decision threshold on the test split.

## Artifact boundary

The current directory contains the standalone method implementation. Before an
archival release, copy the final data-preparation, leakage-safe split,
evaluation, ablation, robustness, and reporting scripts into versioned
`scripts/` or `experiments/` directories in this repository. Commands in a
paper must resolve using only the tagged repository plus documented external
downloads.

## Release checklist

- [ ] Remove machine-specific paths, credentials, caches, and generated files.
- [ ] Add the final paper title, authors, affiliation, and contact information.
- [ ] Add an author-approved software license.
- [ ] Add `CITATION.cff` and the final BibTeX entry.
- [ ] Add dataset licenses, checksums, statistics, and split manifests.
- [ ] Add the exact experiment runners used for every reported result.
- [ ] Add a small non-sensitive smoke-test dataset.
- [ ] Run tests and a clean-environment reproduction.
- [ ] Tag the artifact version and archive it with a DOI.

