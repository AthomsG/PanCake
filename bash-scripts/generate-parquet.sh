#!/bin/bash
set -euo pipefail

GENE_DIR="${1:-}"
if [[ -z "${GENE_DIR}" ]]; then
  echo "Usage: $0 /path/to/gene_dir [rank] [output_dir]"
  exit 1
fi

RANK="${2:-s__}"
OUTPUT_DIR="${3:-outputs/generate-parquet}"
SANDPIPER_DIR="${OUTPUT_DIR}/sandpiper"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
REQ_FILE="${SCRIPT_DIR}/requirements/generate-parquet.txt"

mkdir -p "${SANDPIPER_DIR}"
mkdir -p "${OUTPUT_DIR}/parquet"

echo "=== [1/5] Creating temporary venv ==="
python3 -m venv "${SCRIPT_DIR}/.venv"

echo "=== [2/5] Activating venv & installing requirements ==="
source "${SCRIPT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip -q

python -m pip install -v -r "${REQ_FILE}" -q || { echo "Failed to install requirements"; exit 1; }

echo "=== [3/5] Querying Sandpiper API ==="
python "${SRC_DIR}/query-sandpiper.py" \
  "${GENE_DIR}" \
  "${SANDPIPER_DIR}" \
  --conc 20 --timeout 60 || { echo "Sandpiper query failed but continuing"; }

echo "=== [4/5] Generating parquet files ==="
python "${SRC_DIR}/gather-data.py" \
  --genes \
  --gene_reads \
  --taxon "${RANK}" \
  --genes_dir "${GENE_DIR}" \
  --species_dir "${SANDPIPER_DIR}" \
  --data_dir "${OUTPUT_DIR}" \
  --verbose || { echo "Parquet generation failed"; exit 1; }

echo "=== [5/5] Cleaning up venv ==="
deactivate
rm -rf "${SCRIPT_DIR}/.venv"

echo
echo "✅ Done."
echo "   Sandpiper TSVs in: ${SANDPIPER_DIR}"
echo "   Parquet outputs in: ${OUTPUT_DIR}/parquet"
