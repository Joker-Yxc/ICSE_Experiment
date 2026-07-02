# EsCapturer

Official research implementation of **EsCapturer**, a dual-view anomaly
detection method for system/API call traces. EsCapturer combines sequential and
graph representations of behavior units with an interleaving-prior adaptive
fusion module.

> **Release status:** the code is being prepared for an accompanying research
> artifact. Paper metadata, the archival DOI, and the final citation will be
> added when they become available.

## Overview

The current implementation provides:

- dynamic vocabulary construction from real call traces;
- deterministic behavior-element extraction;
- sequential and graph encoders;
- interleaving-prior adaptive fusion;
- Deep SVDD-based anomaly detection; and
- JSON result, vocabulary, intention-group, and extraction-cache artifacts.

The behavior extractor is a frozen, deterministic semantic-template library and
does not call an online LLM at runtime. Claims about an "LLM-assisted" stage
should be made only together with documentation of the source model, prompts,
distillation procedure, and human review.

## Repository layout

```text
.
├── escapture.py                 # Stable command-line entrypoint
├── escapture_true.py            # Core model, training, and evaluation
├── llm_behavior_extractor.py    # Deterministic behavior extraction
├── requirements.txt
├── setup_env.sh
├── data/
│   └── README.md                # Dataset sources, schema, and release policy
├── docs/
│   └── REPRODUCIBILITY.md       # Full reproducibility protocol
├── examples/
│   ├── attack_seq.txt
│   └── normal_seq.txt
└── tests/
    └── test_behavior_extractor.py
```

Generated datasets, checkpoints, caches, and result files are intentionally
excluded from Git.

## Installation

Python 3.10--3.12 is recommended.

```bash
git clone https://github.com/Joker-Yxc/EsCapturer.git
cd EsCapturer
./setup_env.sh
source .venv/bin/activate
```

For an existing environment:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Input files contain one sample per line and whitespace-separated API/system
calls within each sample:

```text
OpenFile ReadFile CloseHandle
CreateFile WriteFile FlushFileBuffers CloseHandle
```

Run a small end-to-end example:

```bash
python escapture.py \
  --attack examples/attack_seq.txt \
  --normal examples/normal_seq.txt \
  --epochs 1 \
  --seed 7 \
  --output results/example.json
```

For a real experiment, replace the example files with full datasets and use
the paper configuration described in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The example is a functional
smoke test only and must not be used to report model quality.

## Data

The study uses the Quo Vadis Malware Emulation dataset and Zenodo record
11079764. Do not commit raw datasets or malware artifacts to this repository.
Dataset links, the compact schema, licensing cautions, and the recommended
release strategy are documented in [data/README.md](data/README.md).

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q escapture.py escapture_true.py llm_behavior_extractor.py
```

## Reproducibility

Report the dataset version/checksum, split manifest, random seeds, complete
command, hardware, Python version, and dependency versions for every result.
The recommended protocol and artifact checklist are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation

Citation metadata will be added after the paper title, author order, venue, and
archival identifier are finalized.

## License

No software license has been granted yet. Before making the repository public,
the authors should add the license approved by all contributors and separately
verify the redistribution terms of every dataset.

