#!/bin/bash
set -euo pipefail

# Usage:
# ./train-models.sh [rank] [base_input_dir] [base_output_dir] [n_null] [logreg_tol] [logreg_max_iter] [cv_folds] [val_frac] [seed]
RANK="${1:-s__}"
BASE_IN="$2"
BASE_OUT="$3"
N_NULL="${4:-10}"
LOGREG_TOL="${5:-1e-4}"
LOGREG_MAX_IT="${6:-100}"
CV_FOLDS="${7:-5}"
VAL_FRAC="${8:-0.15}"
SEED="${9:-10}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
REQ_FILE="${SCRIPT_DIR}/requirements/train-models.txt"

PARQUET_IN="${BASE_IN}"
OUT_DIR="${BASE_OUT}"

mkdir -p "${OUT_DIR}"

echo "=== [1/4] Creating temporary venv ==="
python3 -m venv "${SCRIPT_DIR}/.venv"

echo "=== [2/4] Activating venv & installing requirements ==="
source "${SCRIPT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip -q
python -m pip install -r "${REQ_FILE}" -q || { echo "Failed to install requirements"; exit 1; }

echo "=== [3/4] Training models (LogReg LBFGS + MLP; ${N_NULL} null perms; ${CV_FOLDS}-fold CV) ==="
echo "Using rank: ${RANK}"
echo "Input directory: ${PARQUET_IN}"
echo "Output directory: ${OUT_DIR}"

python "${SRC_DIR}/train-models.py" \
  --input_dir "${PARQUET_IN}" \
  --out_dir "${OUT_DIR}" \
  --rank "${RANK}" \
  --n_null "${N_NULL}" \
  --logreg_tol "${LOGREG_TOL}" \
  --logreg_max_iter "${LOGREG_MAX_IT}" \
  --cv_folds "${CV_FOLDS}" \
  --val_frac "${VAL_FRAC}" \
  --seed "${SEED}"

echo "=== [4/4] Cleaning up venv ==="
deactivate
rm -rf "${SCRIPT_DIR}/.venv"

echo
echo "✅ Done. Results in: ${OUT_DIR}"
