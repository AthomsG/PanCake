#!/bin/bash
# Script to process data

set -euo pipefail

# First parameter is base output dir, second is data directory for reads
BASE_DIR="${1:-}"
DATA_DIR="${2:-}"
if [[ -z "${BASE_DIR}" || -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 /path/to/output_base_dir /path/to/data_dir"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
REQ_FILE="${SCRIPT_DIR}/requirements/process-data.txt"

PARQUET_DIR="${BASE_DIR}/generate-parquet/parquet"
READS_DIR="${DATA_DIR}"  # Use the data directory passed from the slurm script
OUT_DIR="${BASE_DIR}/process-data"

mkdir -p "${OUT_DIR}"

echo "=== [1/4] Creating temporary venv ==="
python3 -m venv "${SCRIPT_DIR}/.venv"

echo "=== [2/4] Activating venv & installing requirements ==="
source "${SCRIPT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip -q
python -m pip install -r "${REQ_FILE}" -q || { echo "Failed to install requirements"; exit 1; }

echo "=== [3/4] Processing data ==="
python "${SRC_DIR}/process-data.py" \
  --parquet_dir "${PARQUET_DIR}" \
  --reads_dir "${READS_DIR}" \
  --rank "s__" \
  --out_dir "${OUT_DIR}"

echo "=== [4/4] Cleaning up venv ==="
deactivate
rm -rf "${SCRIPT_DIR}/.venv"

echo
echo "✅ Done."
echo "   Processed outputs in: ${OUT_DIR}"
