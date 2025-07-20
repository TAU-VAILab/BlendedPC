#!/bin/bash

python run_inference.py \
    --prompt "curved legs" \
    --shape_category "table" \
    --input_path "inputs/demo_table.npz" \
    --part "leg" 