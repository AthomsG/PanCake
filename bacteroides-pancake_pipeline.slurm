#!/usr/bin/env bash
#SBATCH --job-name=bacteroides-pancake_pipeline
#SBATCH --output=logs/bacteroides-pancake_pipeline_%j.out
#SBATCH --error=logs/bacteroides-pancake_pipeline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.gaehtgens@univie.ac.at

set -euo pipefail

# change data dir [HERE] for different focal taxon
DATA_DIR="../data/pancake/bacteroides-genes" # <-- HERE
SCRIPTS_DIR="./bash-scripts"
DATA_NAME=$(basename "$DATA_DIR")

cleanup() {
    if [ $? -ne 0 ]; then
        echo "Job failed, cleaning up .venv directory"
        rm -rf .venv
    fi
}
trap cleanup EXIT

mkdir -p logs
mkdir -p outputs/${DATA_NAME}

echo "=== Starting pancake pipeline on $(hostname) with GPU: ${CUDA_VISIBLE_DEVICES:-none} ==="

# ------------------------------------ # 
# ---------- Data Gathering ---------- #
# ------------------------------------ # 

echo "[Stage 1] Generating parquet data"
$SCRIPTS_DIR/generate-parquet.sh $DATA_DIR "s__" outputs/${DATA_NAME}/generate-parquet

# ------------------------------------- # 
# ---------- Data Processing ---------- #
# ------------------------------------- # 

echo "[Stage 2] Processing data"
$SCRIPTS_DIR/process-data.sh outputs/${DATA_NAME} $DATA_DIR

# ------------------------------------ # 
# ---------- Model Training ---------- #
# ------------------------------------ # 

echo "[Stage 3] Training models"
# 1: RANK
# 2: BASE_IN
# 3: BASE_OUT
# 4: N_NULL
# 5: LOGREG_TOL
# 6: LOGREG_MAX_IT
# 7: CV_FOLDS
# 8: VAL_FRAC
# 9: SEED
$SCRIPTS_DIR/train-models.sh \
  "s__" \
  "outputs/${DATA_NAME}/process-data" \
  "outputs/${DATA_NAME}/models" \
  10 \
  1e-4 \
  400 \
  10 \
  0.15 \
  10

echo "=== Pancake pipeline finished successfully! ==="
