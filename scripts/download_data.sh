#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/raw

echo "Downloading OpenSanctions targets..."
curl -L "https://data.opensanctions.org/datasets/latest/default/targets.simple.csv" \
  -o data/raw/opensanctions_targets.csv

echo "Downloading IBM AML transaction dataset (requires ~/.kaggle/kaggle.json)..."
if command -v kaggle &>/dev/null && [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
    -p data/raw --unzip
else
  echo "Skipping Kaggle download — install kaggle CLI and add credentials to ~/.kaggle/kaggle.json"
fi

echo "Done. Raw data in data/raw/"
