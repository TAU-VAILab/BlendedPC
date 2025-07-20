#!/bin/bash

python run_inference.py \
    --prompt "backrest has vertical slats" \
    --shape_category "chair" \
    --input_path "inputs/demo_chair.npz" \
    --part "back" 