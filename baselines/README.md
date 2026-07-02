# Baselines

This folder contains the baseline implementations used for the Windows API /
system-level call trace malware detection experiments.

## Core Paper Baselines

- `01_nebula_scratch/`
  - Nebula-Scratch baseline and adapted Quo Vadis runner.
- `02_api2vecpp/`
  - API2Vec++ upstream code plus local compact-data runner.
- `03_dawngnn_reimpl/`
  - Updated DawnGNN transition-graph and two-layer GAT reimplementation.

## Additional Baselines

- `04_gpt4_api_bert_cnn/`
- `05_deepcapa_adapted/`
- `06_apili_adapted/`
- `07_mme/`
  - MME-TextCNN fallback reproduction with a training-only API graph.

The latest consolidated 50k datasets and result CSV files are stored under:

`datasets_50k/`
