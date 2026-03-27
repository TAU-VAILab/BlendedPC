#!/bin/bash
#
# Overfit test: train on a small fixed subset to verify the model can memorize it.
# Validates every few epochs so you can visually track whether the outputs
# converge to the training data.
#
# Before running, set these environment variables (or edit the defaults below):
#   SHAPETALK_DIR  — path to the ShapeTalk dataset root
#   OUTPUT_DIR     — where to save checkpoints, loss plots, and sample grids
#
# Look at:
#   - samples/val/reference.png  → what the ground-truth shapes look like
#   - samples/val/epoch_*.png    → model outputs should converge toward reference
#   - train_loss.png             → loss should drop steadily
#
# Usage:
#   bash demos/test_overfit.sh

set -e

# ── User configuration ────────────────────────────────────────────────────
SHAPETALK_DIR="${SHAPETALK_DIR:?Please set SHAPETALK_DIR to the ShapeTalk dataset root}"
OUTPUT_DIR="${OUTPUT_DIR:?Please set OUTPUT_DIR to the desired output directory}"

# ── Environment setup ─────────────────────────────────────────────────────
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# To clear all caches (e.g. after changing mask or encoding logic), run:
#   rm -rf /tmp/blendedpc_cache

cd "$(dirname "$0")/.."

python finetune.py \
    --object chair \
    --shapetalk_dir "$SHAPETALK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --subset_size 64 \
    --epochs 500 \
    --batch_size 10 \
    --accumulate_grad_batches 6 \
    --val_freq 5 \
    --lr 1e-4 \
    --copy_prob 0.1 \
    --cond_drop_prob 0.5 \
    --num_val_samples 5 \
    --num_test_samples 5
