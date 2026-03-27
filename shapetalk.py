"""
ShapeTalk dataset for training BlendedPC point cloud editing models.

Loads source/target point cloud pairs from ShapeTalk CSV splits. Each sample
provides an edit prompt, a masked source point cloud (inpainting condition),
and a target point cloud (ground truth). Latents are precomputed at init time
for fast training iteration.

Expected directory layout:
    <shapetalk_dir>/
        point_clouds/
            scaled_to_align_rendering/
                <category>/ShapeNet/<uid>.npz

CSV columns required: source_uid, source_object_class, utterance_spelled,
    and at least one of: part_keywords, part_llama3 (for inpainting).
"""

import gc
import os
import copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud

NUM_POINTS = 1024
MASK_COLOR = 1.0
MASK_COORD = 0.0

PROMPTS = "prompts"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"


def get_part(row):
    """Extract part label, preferring keyword-based over LLM-based."""
    for col in ("part_keywords", "part_llama3"):
        val = str(row.get(col, "unknown"))
        if val not in ("unknown", "nan", ""):
            return val
    raise ValueError(f"No part information for row index {row.name}")


class ShapeTalkDataset(Dataset):
    """
    Point cloud editing pairs from ShapeTalk.

    Precomputes encoded latent tensors (6 x NUM_POINTS) at init.
    Lengths are padded to multiples of batch_size for even batching;
    extra indices wrap around to the beginning of the dataset.

    Args:
        csv_path:      Path to a train/test split CSV file.
        shapetalk_dir: Root of the ShapeTalk dataset.
        device:        Torch device to place latent tensors on.
        batch_size:    Used for padding length to full batches.
        inpainting:    If True, mask the source by its labelled part.
        cache_dir:     Directory for caching segmented/masked point clouds.
        subset_size:   Cap the number of samples loaded (useful for debugging).
    """

    def __init__(self, csv_path, shapetalk_dir, device, batch_size,
                 inpainting=True, cache_dir=None, subset_size=None):
        super().__init__()
        self.device = device
        self.pc_dir = os.path.join(
            shapetalk_dir, "point_clouds", "scaled_to_align_rendering")

        df = pd.read_csv(csv_path)
        if subset_size and subset_size < len(df):
            df = df.head(subset_size)

        # Latent cache stores fully-processed (prompt, source, target) tuples
        # on local disk so subsequent runs skip S3 reads, FPS, and segmentation.
        latent_cache_dir = os.path.join(cache_dir, "latents") if cache_dir else None
        if latent_cache_dir:
            os.makedirs(latent_cache_dir, exist_ok=True)

        self.prompts, self.source_latents, self.target_latents = [], [], []
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(csv_path)}")):
            try:
                self._load_sample(row, inpainting, cache_dir, latent_cache_dir)
            except Exception as e:
                print(f"  Skipping {row.get('source_uid', '?')}: {e}")
            if i % 200 == 0:
                gc.collect()

        self._set_length(batch_size)

    def _load_pc(self, uid):
        """Load a ShapeTalk point cloud by its UID string (e.g. 'chair/ShapeNet/<hash>')."""
        return PointCloud.load_shapetalk(os.path.join(self.pc_dir, f"{uid}.npz"))

    def _load_sample(self, row, inpainting, cache_dir, latent_cache_dir):
        uid = row["source_uid"]
        prompt = row["utterance_spelled"]

        # Try loading from the latent cache first (skips S3 reads, FPS, segmentation)
        if latent_cache_dir:
            part = get_part(row) if inpainting else "noinpaint"
            cache_key = f"{uid.replace('/', '_')}_{part}.pt"
            cache_path = os.path.join(latent_cache_dir, cache_key)
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                self.prompts.append(prompt)
                self.source_latents.append(cached["src"])
                self.target_latents.append(cached["tgt"])
                return

        target_pc = self._load_pc(uid).farthest_point_sample(NUM_POINTS)

        if inpainting:
            source_pc = self._masked_pc(target_pc, row, cache_dir)
        else:
            source_pc = self._load_pc(uid).farthest_point_sample(NUM_POINTS)

        src_latent = source_pc.encode()
        tgt_latent = target_pc.encode()

        if latent_cache_dir:
            torch.save({"src": src_latent, "tgt": tgt_latent}, cache_path)

        self.prompts.append(prompt)
        self.source_latents.append(src_latent)
        self.target_latents.append(tgt_latent)

    def _masked_pc(self, target_pc, row, cache_dir):
        """Create a part-masked version of the target for inpainting conditioning."""
        part = get_part(row)
        uid = row["source_uid"]

        if cache_dir:
            path = os.path.join(cache_dir, f"{uid.replace('/', '_')}_{part}.npz")
            if os.path.exists(path):
                return PointCloud.load(path)

        pc = copy.deepcopy(target_pc)
        pc.set_shape_category(row["source_object_class"])
        pc = pc.segment_pointcloud()
        # indices() returns points NOT in the part; invert to get the part itself
        non_part_idx = pc.indices(part)
        part_idx = np.delete(np.arange(len(pc.coords)), non_part_idx)
        pc.coords[part_idx] = MASK_COORD
        for c in "RGB":
            pc.channels[c][part_idx] = MASK_COLOR

        if cache_dir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pc.save(path)

        return pc

    def _set_length(self, batch_size, length=None):
        n = length if length is not None else len(self.prompts)
        self.length = min(n, len(self.prompts))
        r = self.length % batch_size
        self.logical_length = self.length + (batch_size - r if r else 0)

    def set_length(self, batch_size, length=None):
        """Resize the effective dataset length (with batch-size padding)."""
        self._set_length(batch_size, length)

    def __len__(self):
        return self.logical_length

    def __getitem__(self, idx):
        i = idx % self.length
        return {
            PROMPTS: self.prompts[i],
            SOURCE_LATENTS: self.source_latents[i].to(self.device),
            TARGET_LATENTS: self.target_latents[i].to(self.device),
        }
