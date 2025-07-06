import os
import torch
import shutil
import random
import argparse
import json
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from point_e.util.point_cloud import PointCloud
from point_e.util.rendering import render_point_cloud
from spice import SPICE, NUM_POINTS_LOW, NUM_POINTS_HIGH, NUM_INFERENCE_STEPS

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="backrest has vertical slats")
    parser.add_argument("--output_dir", type=str, default="outputs")
    #parser.add_argument("--blending_cache_dir", type=str, default="blending_cache")
    parser.add_argument("--shapetalk_dir", type=str, default="shapetalk")
    parser.add_argument("--transition_timestep", type=int, default=20, choices=list(range(NUM_INFERENCE_STEPS)))
    parser.add_argument("--shape_category", type=str, default="chair", choices=["chair", "table", "lamp"])
    parser.add_argument("--input_uid", type=str, default="chair/ShapeNet/4b3ddc244c521f5c6a9ab6fc87e1604e")
    parser.add_argument("--seed", type=int, default=0, help="Set seed for reproducibility.")
    parser.add_argument("--copy_prompt", type=str, default="COPY", help="Use COPY when using our pretrained models!")
    parser.add_argument("--part", type=str, default="back", choices=["leg", "arm", "seat", "back", "top", "support", "base", "shade", "bulb", "tube"])
    parser.add_argument("--save_pcs", action="store_true", help="save final and intermediate point clouds to disk")
    parser.add_argument("--keep_latents", type=bool, default=False)
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    subfolders = [
        f for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f)) and \
            f.startswith(args.shape_category)
    ]
    if subfolders:
        subfolders.sort()
        last_subfolder = subfolders[-1]
        last_number = int(last_subfolder.split("_")[-1], 16)
        new_number = last_number + 1
        new_subfolder = f"{args.shape_category}_{new_number:05x}"
    else:
        new_subfolder = f"{args.shape_category}_00000"
    save_path = os.path.join(args.output_dir, new_subfolder)
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    print(f"Downloading and loading BlendedPC model...")
    checkpoint_path = hf_hub_download(repo_id="noamatia/BPCDiff", filename=f"{args.shape_category}.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SPICE.load_from_checkpoint(dev=device, checkpoint_path=checkpoint_path)
    model.eval()

    # save metadata json file to out folder
    metadata = {
        "prompt": args.prompt,
        "output_dir": args.output_dir,
        "shapetalk_dir": args.shapetalk_dir,
        "transition_timestep": args.transition_timestep,
        "shape_category": args.shape_category,
        "input_uid": args.input_uid,
        "copy_prompt": args.copy_prompt,
        "part": args.part,
        "seed": args.seed,
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Input point cloud
    print(f"Loading input point cloud...")
    input_pc = PointCloud.load_shapetalk(args.input_uid, args.shapetalk_dir).farthest_point_sample(NUM_POINTS_HIGH)
    if args.save_pcs:
        input_pc.save(os.path.join(save_path, 'input_pc.npz'))
    print(f"Rendering input point cloud, saving to {os.path.join(save_path, 'input.png')}...")
    Image.fromarray(render_point_cloud(input_pc)).save(os.path.join(save_path, 'input.png'))

    # Copy point cloud
    print(f"Copying point cloud...")
    blending_dir = os.path.join(save_path, 'reconstruction_latents')
    os.makedirs(blending_dir, exist_ok=True)
    input_pc = input_pc.farthest_point_sample(1024)
    input_latents = input_pc.encode().unsqueeze(0).to(device)
    samples = model.sampler.sample_batch(
                batch_size=1,
                blending_cache_dir=blending_dir,
                guidances=[input_latents, None],
                model_kwargs={"texts": [args.copy_prompt]},
            )
    copy_pc = model.sampler.output_to_point_clouds(samples)[0]
    if args.save_pcs:
        copy_pc.save(os.path.join(save_path, 'reconstructed_pc.npz'))
    print(f"Rendering reconstructed point cloud, saving to {os.path.join(save_path, 'reconstruction.png')}...")
    Image.fromarray(render_point_cloud(copy_pc)).save(os.path.join(save_path, 'reconstruction.png'))
    
    # Masked point cloud
    print(f"Masking point cloud...")
    input_pc.set_shape_category(args.shape_category)
    masked_pc = input_pc.segment_pointcloud()
    masked_indices = masked_pc.indices(args.part)
    indices = np.delete(np.arange(NUM_POINTS_LOW), masked_indices)
    masked_pc.coords[indices] = 0.0
    for c in "RGB":
        masked_pc.channels[c][indices] = 1.0
    if args.save_pcs:
        masked_pc.save(os.path.join(save_path, 'masked_pc.npz'))
    print(f"Rendering masked point cloud, saving to {os.path.join(save_path, 'masked.png')}...")
    Image.fromarray(render_point_cloud(masked_pc)).save(os.path.join(save_path, 'masked.png'))
    
    # Output point cloud
    print(f"Generating output point cloud...")
    masked_latents = masked_pc.encode().unsqueeze(0).to(device)
    samples = model.sampler.sample_batch(
                batch_size=1,
                blending_cache_dir=blending_dir,
                model_kwargs={"texts": [args.prompt]},
                guidances=[masked_latents, None],
                blending_indices_list=[masked_indices],
                transition_timestep=NUM_INFERENCE_STEPS - args.transition_timestep,
            )
    output_pc = model.sampler.output_to_point_clouds(samples)[0]
    if args.save_pcs:
        output_pc.save(os.path.join(save_path, 'output_pc.npz'))
    print(f"Rendering output point cloud, saving to {os.path.join(save_path, 'output.png')}...")
    Image.fromarray(render_point_cloud(output_pc)).save(os.path.join(save_path, 'output.png'))

    # delete blending directory if keep_latents is False
    if not args.keep_latents:
        shutil.rmtree(blending_dir)

if __name__ == "__main__":
    main()
    