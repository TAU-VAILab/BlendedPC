# Blended Point Cloud Diffusion for Localized Text-Guided Shape Editing

**Etai Sella<sup>1</sup>, Noam Atia<sup>1</sup>, Ron Mokady<sup>2</sup>, Hadar Averbuch-Elor<sup>3</sup>**

<sup>1</sup> Tel Aviv University  <sup>2</sup> BRIA AI <sup>3</sup> Cornell University  

This is the official PyTorch implementation of **BlendedPC**.

[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)](https://arxiv.org/abs/)  
[[Project Website](https://tau-vailab.github.io/BlendedPC/)]

## Abstract

Natural language offers a highly intuitive interface for enabling localized, fine-grained edits of 3D shapes. However, prior works face challenges in preserving global coherence while locally modifying the input 3D shape.

We introduce an **inpainting-based framework** for editing shapes represented as point clouds. Our approach leverages foundation 3D diffusion models for localized shape edits, adding structural guidance through partial conditional shapes to preserve global identity. To enhance identity preservation within edited regions, we propose an **inference-time coordinate blending algorithm**. This algorithm balances reconstruction of the full shape with inpainting over progressive noise levels, enabling seamless blending of original and edited shapes without requiring costly and inaccurate inversion.

Extensive experiments demonstrate that our method outperforms existing techniques across multiple metrics, measuring both fidelity to the original shape and adherence to textual prompts.

<p align="center">
<img src="webpage_assets/images/teaser.png">
</p>

---

## Getting Started

### Cloning the repository

```bash
git clone git@github.com:TAU-VAILab/BlendedPC.git
cd BlendedPC
```

### Setting up the environment

```bash
conda create --name blended-pc -y python=3.11
conda activate blended-pc
pip install -e .
```

---

## Running the Demo

Run one of the following scripts to test our "chair", "lamp" or "table" models:

```bash
bash demos/chair_demo.sh 
```
```bash
bash demos/lamp_demo.sh 
```
```bash
bash demos/table_demo.sh 
```

Model checkpoints are automatically downloaded from the Hugging Face Hub by default.

**Expected Outputs:**

- `input.png`: The original input shape
- `reconstruction.png`: Output of the model using the "copy" prompt
- `masked.png`: Input shape with masked regions
- `output.png`: Final output after editing

---

## Using other shapes from ShapeTalk

Download the ShapeTalk dataset from [here](https://changeit3d.github.io/#dataset).  
Then run the script with your desired parameters:

```bash
python run_inference.py --prompt <YOUR-PROMPT> --shape_category <SHAPE-CATEGORY> --input_path <INPUT-PATH> --part <SHAPE-PART>
```

Please refer to the previously mentioned demo scripts for examples on how to set these arguments.

---

## Training a Model

Coming soon...

---

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{

}
```

---

## Acknowledgements

We thank the authors of [Point-E](https://github.com/openai/point-e) for their outstanding codebase, which served as a foundation for this project.
