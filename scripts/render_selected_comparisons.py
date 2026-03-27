#!/usr/bin/env python3
"""
Render BlendedPC and ChangeIt3d point clouds for the selected comparison folders.

Standalone script that avoids importing the full point_e package (which requires ailia).
Reimplements the minimal PointCloud loading and Mitsuba rendering inline.

Usage:
    python render_selected_comparisons.py                          # render all
    python render_selected_comparisons.py --sweep                  # orientation sweep on first sample
    python render_selected_comparisons.py --orientation 6          # render all with orientation 6
    python render_selected_comparisons.py --method BlendedPC       # only render BlendedPC
"""

import os
import argparse
import numpy as np
import mitsuba as mi
mi.set_variant("scalar_rgb")
from PIL import Image, ImageDraw, ImageFont

SELECTED_DIR = "/nfs/usr/esella/mys3gallery/prox_e_new_comparisons/000_selected"
BLENDED_DIR = os.path.join(SELECTED_DIR, "BlendedPC")
CHANGEIT_DIR = os.path.join(SELECTED_DIR, "ChangeIt3d")

ALL_METHODS = {
    "BlendedPC": BLENDED_DIR,
    "ChangeIt3d": CHANGEIT_DIR,
}

# ── Orientation transforms (3x3 rotation matrices) ──

ORIENTATION_TRANSFORMS = {
    0:  ("none",              np.eye(3)),
    1:  ("flip_z",            np.diag([1, 1, -1]).astype(float)),
    2:  ("flip_y",            np.diag([1, -1, 1]).astype(float)),
    3:  ("flip_x",            np.diag([-1, 1, 1]).astype(float)),
    4:  ("flip_y_and_z",      np.diag([1, -1, -1]).astype(float)),
    5:  ("swap_y_z",          np.array([[1,0,0],[0,0,1],[0,1,0]], dtype=float)),
    6:  ("swap_y_z_flip_z",   np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)),
    7:  ("swap_y_z_flip_y",   np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)),
    8:  ("swap_y_z_flip_both",np.array([[1,0,0],[0,0,-1],[0,-1,0]], dtype=float)),
    9:  ("rot_180_x",         np.diag([1, -1, -1]).astype(float)),
    10: ("rot_180_y",         np.diag([-1, 1, -1]).astype(float)),
    11: ("rot_180_z",         np.diag([-1, -1, 1]).astype(float)),
    12: ("rot_90_x",          np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)),
    13: ("rot_-90_x",         np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)),
    14: ("rot_90_y",          np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=float)),
    15: ("rot_-90_y",         np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=float)),
}

# Compound rotations: rot_90_y then another 90-degree rotation
_R90y = np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=float)
_R90x = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
_Rn90x = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
_R90z = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
_Rn90z = np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=float)
_R180x = np.diag([1,-1,-1]).astype(float)
_R180z = np.diag([-1,-1,1]).astype(float)

_Rn90y = np.array([[0,0,-1],[0,1,0],[1,0,0]], dtype=float)

ORIENTATION_TRANSFORMS[28] = ("rot_90_z",  _R90z)
ORIENTATION_TRANSFORMS[29] = ("rot_-90_z", _Rn90z)

COMPOUND_TRANSFORMS = {
    16: ("90y_then_90x",   _R90x  @ _R90y),
    17: ("90y_then_-90x",  _Rn90x @ _R90y),
    18: ("90y_then_90z",   _R90z  @ _R90y),
    19: ("90y_then_-90z",  _Rn90z @ _R90y),
    20: ("90y_then_180x",  _R180x @ _R90y),
    21: ("90y_then_180z",  _R180z @ _R90y),
    22: ("-90y_then_90x",  _R90x  @ _Rn90y),
    23: ("-90y_then_-90x", _Rn90x @ _Rn90y),
    24: ("-90y_then_90z",  _R90z  @ _Rn90y),
    25: ("-90y_then_-90z", _Rn90z @ _Rn90y),
    26: ("-90y_then_180x", _R180x @ _Rn90y),
    27: ("-90y_then_180z", _R180z @ _Rn90y),
}
ORIENTATION_TRANSFORMS.update(COMPOUND_TRANSFORMS)

# ── Mitsuba XML templates (from point_e/util/rendering.py) ──

# 1. Improved Integrator and Camera
# --- XML_HEAD CHANGES ---
# Switch gaussian to box and increase samples slightly to compensate for sharpness
XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="8"/>
    </integrator>
    
    <emitter type="constant">
        <rgb name="radiance" value="1, 1, 1"/>
    </emitter>

    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="0.6428,1.7660,0.6840" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="45"/>
        <sampler type="independent">
            <integer name="sampleCount" value="1536"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="712"/>
            <integer name="height" value="712"/>
            <rfilter type="box"/> 
        </film>
    </sensor>
"""

# --- XML_BALL CHANGES ---
# Shrink the radius slightly (0.012 instead of 0.015) 
# This creates tiny gaps between points, which defines the shape better.
XML_BALL = """
    <shape type="sphere">
        <float name="radius" value="0.012"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
# 3. Soft shadows and Floor only
XML_TAIL = """
    <shape type="rectangle">
        <bsdf type="diffuse">
            <rgb name="reflectance" value="1, 1, 1"/>
        </bsdf>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.25"/>
        </transform>
    </shape>

    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="5" y="5" z="1"/>
            <lookat origin="-2, 2, 10" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="1.5, 1.5, 1.5"/>
        </emitter>
    </shape>
</scene>
"""

def load_raw_coords(npz_path: str) -> np.ndarray:
    """Load raw point cloud coords from .npz (no axis transform applied)."""
    with open(npz_path, "rb") as f:
        coords = np.load(f)["pointcloud"].astype(np.float32)
    return coords.squeeze()  # (1, N, 3) -> (N, 3)


def apply_shapetalk_transform(coords: np.ndarray) -> np.ndarray:
    """Apply the standard ShapeTalk axis reordering: swap axes and flip x."""
    out = coords.copy()
    out[:, [0, 1, 2]] = out[:, [2, 0, 1]]
    out[:, 0] = -1.0 * out[:, 0]
    return out


def apply_orientation(coords: np.ndarray, orientation: int) -> np.ndarray:
    """Apply a 3x3 orientation transform to point cloud coordinates."""
    if orientation is None or orientation == 0:
        return coords
    _, mat = ORIENTATION_TRANSFORMS[orientation]
    return coords @ mat.T


def render_point_cloud(coords: np.ndarray, output_path: str = None) -> np.ndarray:
    """Render a point cloud using Mitsuba."""
    xml_segments = [XML_HEAD]
    for point in coords:
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        rgb = np.array([x + 0.5, y + 0.5, z + 0.5 - 0.0125])
        color = np.clip(rgb, 0.001, 1.0)
        color /= np.linalg.norm(color)
        xml_segments.append(XML_BALL.format(
            x, y, z, float(color[0]), float(color[1]), float(color[2])
        ))
    xml_segments.append(XML_TAIL)
    xml_content = "".join(xml_segments)

    scene = mi.load_string(xml_content)
    img = mi.render(scene)
    if output_path is not None:
        mi.util.write_bitmap(output_path, img)
    img = np.array(img)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def orientation_sweep(methods: dict, sweep_range: tuple = None, sample: str = None):
    """Render a sample in orientation range and produce a grid."""
    for method_name, method_dir in methods.items():
        if not os.path.isdir(method_dir):
            continue
        npz_files = sorted(f for f in os.listdir(method_dir) if f.endswith(".npz"))
        if not npz_files:
            continue

        if sample:
            npz_file = f"{sample}.npz"
            if npz_file not in npz_files:
                print(f"  Sample {sample} not found in {method_name}")
                continue
        else:
            npz_file = npz_files[0]
        npz_path = os.path.join(method_dir, npz_file)
        assignment_id = npz_file[:-4]
        print(f"\n=== Orientation sweep for {method_name} (sample: {assignment_id}) ===")

        raw_coords = load_raw_coords(npz_path)
        out_dir = os.path.join(SELECTED_DIR, f"_sweep_{method_name}")
        os.makedirs(out_dir, exist_ok=True)

        image_paths = []
        labels = []

        indices = sorted(ORIENTATION_TRANSFORMS.keys())
        if sweep_range is not None:
            lo, hi = sweep_range
            indices = [i for i in indices if lo <= i <= hi]

        for idx in indices:
            name, _ = ORIENTATION_TRANSFORMS[idx]
            png_path = os.path.join(out_dir, f"orient_{idx:02d}_{name}.png")
            print(f"  [{idx:2d}] {name} ... ", end="", flush=True)
            try:
                coords = apply_shapetalk_transform(raw_coords)
                coords = apply_orientation(coords, idx)
                img = render_point_cloud(coords)
                Image.fromarray(img).save(png_path)
                image_paths.append(png_path)
                labels.append(f"{idx}: {name}")
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")

        if not image_paths:
            print("  No orientations rendered successfully.")
            continue

        cols = 4
        rows = (len(image_paths) + cols - 1) // cols
        label_h = 30
        sample_img = Image.open(image_paths[0])
        cell_w, cell_h = sample_img.size
        grid_w = cols * cell_w
        grid_h = rows * (cell_h + label_h)
        grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            r, c = divmod(i, cols)
            x = c * cell_w
            y = r * (cell_h + label_h)
            tile = Image.open(img_path)
            grid.paste(tile, (x, y + label_h))
            draw.text((x + 5, y + 5), label, fill=(0, 0, 0), font=font)

        grid_path = os.path.join(out_dir, f"{method_name}_orientation_sweep.png")
        grid.save(grid_path)
        print(f"  Sweep grid saved: {grid_path}")


def render_all(methods: dict, orientation: int = None, force: bool = False):
    """Render all point clouds with the given orientation."""
    for method_name, method_dir in methods.items():
        if not os.path.isdir(method_dir):
            print(f"Skipping {method_name}: {method_dir} not found")
            continue

        npz_files = sorted(f for f in os.listdir(method_dir) if f.endswith(".npz"))
        print(f"\n=== {method_name}: {len(npz_files)} files (orientation={orientation}) ===")

        for npz_file in npz_files:
            assignment_id = npz_file[:-4]
            subfolder = os.path.join(SELECTED_DIR, assignment_id)
            if not os.path.isdir(subfolder):
                print(f"  Skipping {assignment_id}: subfolder not found")
                continue

            output_png = os.path.join(subfolder, f"{method_name}.png")
            if not force and os.path.exists(output_png):
                print(f"  Already exists: {output_png}")
                continue

            npz_path = os.path.join(method_dir, npz_file)
            print(f"  Rendering {assignment_id} -> {method_name}.png ... ", end="", flush=True)
            try:
                raw_coords = load_raw_coords(npz_path)
                coords = apply_shapetalk_transform(raw_coords)
                coords = apply_orientation(coords, orientation)
                img = render_point_cloud(coords)
                Image.fromarray(img).save(output_png)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="Run orientation sweep on first sample per method")
    parser.add_argument("--orientation", type=int, default=None,
                        help="Orientation transform index to apply (0-15)")
    parser.add_argument("--method", type=str, default=None,
                        choices=list(ALL_METHODS.keys()),
                        help="Only process this method")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing renders")
    parser.add_argument("--sweep-range", type=str, default=None,
                        help="Only sweep orientations in range, e.g. '16-27'")
    parser.add_argument("--sample", type=str, default=None,
                        help="Specific assignment ID to use for sweep")
    args = parser.parse_args()

    methods = {args.method: ALL_METHODS[args.method]} if args.method else ALL_METHODS

    sweep_range = None
    if args.sweep_range:
        lo, hi = args.sweep_range.split("-")
        sweep_range = (int(lo), int(hi))

    if args.sweep:
        orientation_sweep(methods, sweep_range=sweep_range, sample=args.sample)
    else:
        render_all(methods, orientation=args.orientation, force=args.force)


if __name__ == "__main__":
    main()
