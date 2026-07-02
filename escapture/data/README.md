# Data

This directory contains documentation only. Raw traces, compact datasets,
malware samples, and generated splits must not be committed to Git.

## Sources

The experiments currently use:

- [Quo Vadis Malware Emulation](https://www.kaggle.com/datasets/dmitrijstrizna/quo-vadis-malware-emulation)
- [API Call Traces for Malware Detection (Zenodo 11079764)](https://zenodo.org/records/11079764)

Download data from the original providers and review their current licenses and
terms before use or redistribution.

## Compact schema

The internal compact format is gzip-compressed JSON Lines (`.jsonl.gz`), with
one object per sample:

```json
{
  "sample_id": "dataset-specific-id",
  "source": "dataset-name",
  "label": "benign",
  "family": "benign",
  "api_seq": ["OpenFile", "ReadFile", "CloseHandle"],
  "split": "train"
}
```

Required fields are `sample_id`, `source`, `label`, `api_seq`, and `split`.
Allowed split values are `train`, `val`, and `test`. Store no host paths,
credentials, personal information, or executable malware content in compact
files.

## Recommended local layout

```text
data/
├── raw/                  # Provider downloads; never committed
├── processed/            # Compact JSONL.GZ files; never committed
├── manifests/            # Split IDs and checksums; suitable for Git
└── README.md
```

## Release policy

Do not place a large dataset in normal Git history, and do not assume Git LFS
makes redistribution legally permissible.

For an artifact release:

1. Keep download and preprocessing scripts in the code repository.
2. Commit deterministic split manifests, dataset statistics, and SHA-256
   checksums.
3. If redistribution is permitted, publish compact derived data through an
   archival service such as Zenodo and link its versioned DOI from the README.
4. If redistribution is not permitted, publish only scripts and manifests so
   users can reconstruct the exact split from provider-authorized downloads.
5. Include only a tiny, non-sensitive synthetic sample under `examples/` for
   smoke testing.

