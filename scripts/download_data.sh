#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Load credentials from .env if present (KAGGLE_USERNAME / KAGGLE_KEY).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p data/raw

echo "Downloading OpenSanctions targets..."
curl -L "https://data.opensanctions.org/datasets/latest/default/targets.simple.csv" \
  -o data/raw/opensanctions_targets.csv

echo "Downloading IBM AML transaction dataset (HI-Small_Trans.csv only)..."
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]] || [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  uv run kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
    -f HI-Small_Trans.csv -p data/raw --unzip
else
  echo "Skipping Kaggle download — set KAGGLE_USERNAME/KAGGLE_KEY in .env or add ~/.kaggle/kaggle.json"
fi

echo "Done. Raw data in data/raw/"
