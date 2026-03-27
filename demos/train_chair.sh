#!/bin/bash
#
# Train a BlendedPC model on the ShapeTalk chair category.
#
# Before running, set these environment variables (or edit the defaults below):
#   SHAPETALK_DIR  — path to the ShapeTalk dataset root
#   OUTPUT_DIR     — where to save checkpoints, loss plots, and sample grids
#
# To change the category, copy this script and adjust --object and the
# script name (e.g. train_table.sh with --object table).
#
# Usage:
#   bash demos/train_chair.sh

set -e

# ── User configuration ────────────────────────────────────────────────────
SHAPETALK_DIR="${SHAPETALK_DIR:?Please set SHAPETALK_DIR to the ShapeTalk dataset root}"
OUTPUT_DIR="${OUTPUT_DIR:?Please set OUTPUT_DIR to the desired output directory}"

# ── Environment setup ─────────────────────────────────────────────────────
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Run training ──────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

python finetune.py \
    --object chair \
    --shapetalk_dir "$SHAPETALK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 500 \
    --batch_size 10 \
    --accumulate_grad_batches 6 \
    --val_freq 5 \
    --lr 1e-4 \
    --copy_prob 0.1 \
    --cond_drop_prob 0.5 \
    --num_val_samples 10 \
    --num_test_samples 10
