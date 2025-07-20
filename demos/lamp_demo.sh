#!/bin/bash

python run_inference.py \
    --prompt "dome shaped shade" \
    --shape_category "lamp" \
    --input_path "inputs/demo_lamp.npz" \
    --part "shade" 