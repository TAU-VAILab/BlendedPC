import os
import torch
import shutil
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from point_e.util.point_cloud import PointCloud
from point_e.util.rendering import render_point_cloud
from spice import SPICE, NUM_POINTS_LOW, NUM_POINTS_HIGH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="spindle backrest")
    parser.add_argument("--blending_cache_dir", type=str, default="blending_cache")
    parser.add_argument("--shapetalk_dir", type=str, default="shapetalk")
    parser.add_argument("--transition_timestep", type=int, default=45, choices=list(range(65)))
    parser.add_argument("--shape_category", type=str, default="chair", choices=["chair", "table", "lamp"])
    parser.add_argument("--input_uid", type=str, default="chair/ShapeNet/4b3ddc244c521f5c6a9ab6fc87e1604e")
    parser.add_argument("--copy_prompt", type=str, default="COPY", help="Use COPY when using our pretrained models!")
    parser.add_argument("--part", type=str, default="back", choices=["leg", "arm", "seat", "back", "top", "support", "base", "shade", "bulb", "tube"])
    parser.add_argument("--D", action="store_true", help="Use blending cache directory to get deterministic prev results.")
    args = parser.parse_args()

    if args.blending_cache_dir is None:
        raise ValueError("blending_cache_dir must be specified.")

    if not args.D and os.path.exists(args.blending_cache_dir):
        print(f"Removing existing blending cache directory: {args.blending_cache_dir}")
        shutil.rmtree(args.blending_cache_dir)

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
    coy_pc_path = os.path.join(args.blending_cache_dir, "copy_pc.npz")
    if args.D:
        assert os.path.exists(coy_pc_path), f"Copy point cloud not found at {coy_pc_path}. Please run without --D to generate it."
        print(f"Loading copy point cloud from {coy_pc_path}...")
        copy_pc = PointCloud.load(coy_pc_path)
    else:
        input_pc = input_pc.farthest_point_sample(1024)
        input_latents = input_pc.encode().unsqueeze(0).to(device)
        samples = model.sampler.sample_batch(
                    batch_size=1,
                    blending_cache_dir=args.blending_cache_dir,
                    guidances=[input_latents, None],
                    model_kwargs={"texts": [args.copy_prompt]},
                )
        copy_pc = model.sampler.output_to_point_clouds(samples)[0]
        copy_pc.save(coy_pc_path)
    print(f"Rendering copy point cloud to outputs/copy.png...")
    Image.fromarray(render_point_cloud(copy_pc)).save("outputs/copy.png")
    
    # Masked point cloud
    print(f"Masking point cloud...")
    masked_pc_path = os.path.join(args.blending_cache_dir, "masked_pc.npz")
    masked_indices_path = os.path.join(args.blending_cache_dir, "masked_indices.npy")
    if args.D:
        assert os.path.exists(masked_pc_path), f"Masked point cloud not found at {masked_pc_path}. Please run without --D to generate it."
        assert os.path.exists(masked_indices_path), f"Masked indices not found at {masked_indices_path}. Please run with --D=False to generate it."
        print(f"Loading masked point cloud from {masked_pc_path}...")
        masked_pc = PointCloud.load(masked_pc_path)
        print(f"Loading masked indices from {masked_indices_path}...")
        masked_indices = np.load(masked_indices_path)
    else:
        input_pc.set_shape_category(args.shape_category)
        masked_pc = input_pc.segment_pointcloud()
        masked_indices = masked_pc.indices(args.part)
        indices = np.delete(np.arange(NUM_POINTS_LOW), masked_indices)
        masked_pc.coords[indices] = 0.0
        for c in "RGB":
            masked_pc.channels[c][indices] = 1.0
        masked_pc.save(masked_pc_path)
        np.save(masked_indices_path, masked_indices)
    print(f"Rendering masked point cloud to outputs/masked.png...")
    Image.fromarray(render_point_cloud(masked_pc)).save("outputs/masked.png")
    
    # Output point cloud
    print(f"Generating output point cloud...")
    masked_latents = masked_pc.encode().unsqueeze(0).to(device)
    samples = model.sampler.sample_batch(
                batch_size=1,
                blending_cache_dir=args.blending_cache_dir,
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
    