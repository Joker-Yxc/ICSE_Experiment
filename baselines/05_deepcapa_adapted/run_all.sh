#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON" "$ROOT/train_deepcapa_adapted.py" \
  --dataset quo_vadis --max-length 512 --fallback-max-length 512 --batch-size 16
"$PYTHON" "$ROOT/train_deepcapa_adapted.py" \
  --dataset zenodo_11079764 --max-length 512 --fallback-max-length 512 --batch-size 16
