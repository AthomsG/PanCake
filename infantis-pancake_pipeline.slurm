#!/usr/bin/env bash
#SBATCH --job-name=infantis-pancake_pipeline
#SBATCH --output=logs/infantis-pancake_pipeline_%j.out
#SBATCH --error=logs/infantis-pancake_pipeline_%j.err
#SBATCH --partition=zen2_0512_l40sx2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.gaehtgens@univie.ac.at

set -euo pipefail

# change data dir for different focal taxon
DATA_DIR="../data/pancake/infantis-genes"
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

echo "[Stage 1] Generating parquet data"
# $SCRIPTS_DIR/generate-parquet.sh $DATA_DIR "s__" outputs/${DATA_NAME}/generate-parquet

echo "[Stage 2] Processing data"
# $SCRIPTS_DIR/process-data.sh outputs/${DATA_NAME} $DATA_DIR

echo "[Stage 3] Training models"
$SCRIPTS_DIR/train-models.sh "s__" outputs/${DATA_NAME}/process-data outputs/${DATA_NAME}/models

echo "=== Pancake pipeline finished successfully! ==="
