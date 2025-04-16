import os
import torch
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from point_e.util.point_cloud import PointCloud
from point_e.util.rendering import render_point_cloud
from spice import SPICE, NUM_POINTS_LOW, NUM_POINTS_HIGH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="thinner legs")
    parser.add_argument("--blending_dir", type=str, default="blending_dir")
    parser.add_argument("--shapetalk_dir", type=str, default="shapetalk")
    parser.add_argument("--transition_timestep", type=int, default=45, choices=list(range(65)))
    parser.add_argument("--shape_category", type=str, default="chair", choices=["chair", "table", "lamp"])
    parser.add_argument("--input_uid", type=str, default="chair/ShapeNet/4c97f421c4ea4396d8ac5d7ad0953104")
    parser.add_argument("--copy_prompt", type=str, default="COPY", help="Use COPY when using our pretrained models!")
    parser.add_argument("--part", type=str, default="leg", choices=["leg", "arm", "seat", "back", "top", "support", "base", "shade", "bulb", "tube"])
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Load the model
    print(f"Downloading and loading BlendedPC model...")
    checkpoint_path = hf_hub_download(repo_id="noamatia/BPCDiff", filename=f"{args.shape_category}.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SPICE.load_from_checkpoint(dev=device, checkpoint_path=checkpoint_path)
    model.eval()
    
    # Input point cloud
    print(f"Loading input point cloud...")
    input_pc = PointCloud.load_shapetalk(args.input_uid, args.shapetalk_dir).farthest_point_sample(NUM_POINTS_HIGH)
    print(f"Rendering input point cloud to outputs/input.png...")
    Image.fromarray(render_point_cloud(input_pc)).save("outputs/input.png")

    # Copy point cloud
    print(f"Copying point cloud...")
    input_pc = input_pc.farthest_point_sample(1024)
    input_latents = input_pc.encode().unsqueeze(0).to(device)
    samples = model.sampler.sample_batch(
                batch_size=1,
                blending_dir=args.blending_dir,
                guidances=[input_latents, None],
                model_kwargs={"texts": [args.copy_prompt]},
            )
    copy_pc = model.sampler.output_to_point_clouds(samples)[0]
    print(f"Rendering copy point cloud to outputs/copy.png...")
    Image.fromarray(render_point_cloud(copy_pc)).save("outputs/copy.png")
    
    # Masked point cloud
    print(f"Masking point cloud...")
    input_pc.set_shape_category(args.shape_category)
    masked_pc = input_pc.segment_pointcloud()
    masked_indices = masked_pc.indices(args.part)
    indices = np.delete(np.arange(NUM_POINTS_LOW), masked_indices)
    masked_pc.coords[indices] = 0.0
    for c in "RGB":
        masked_pc.channels[c][indices] = 1.0
    print(f"Rendering masked point cloud to outputs/masked.png...")
    Image.fromarray(render_point_cloud(masked_pc)).save("outputs/masked.png")
    
    # Output point cloud
    print(f"Generating output point cloud...")
    masked_latents = masked_pc.encode().unsqueeze(0).to(device)
    samples = model.sampler.sample_batch(
                batch_size=1,
                blending_dir=args.blending_dir,
                model_kwargs={"texts": [args.prompt]},
                guidances=[masked_latents, None],
                blending_indices_list=[masked_indices],
                transition_timestep=args.transition_timestep,
            )
    output_pc = model.sampler.output_to_point_clouds(samples)[0]
    print(f"Rendering output point cloud to outputs/output.png...")
    Image.fromarray(render_point_cloud(output_pc)).save("outputs/output.png")
    
if __name__ == "__main__":
    main()
    