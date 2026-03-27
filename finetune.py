"""
Training script for BlendedPC point cloud editing diffusion model.

Finetunes a pretrained Point-E model on ShapeTalk editing pairs using
classifier-free guidance with a copy-reconstruction auxiliary task.

Outputs (saved to --output_dir):
    checkpoints/     Model weights every --val_freq epochs
    train_loss.png   Training loss plot (updated each validation)
    test_loss.png    Test loss plot (computed on full test set)
    loss.csv         Per-epoch train loss, per-validation test loss
    samples/val/     Rendered grids on training subset (source->edit, target->copy)
    samples/test/    Rendered grids on held-out test samples

Usage:
    python finetune.py --object chair --shapetalk_dir /path/to/shapetalk

See demos/train_chair.sh for a ready-to-run example.
"""

import os
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def find_free_gpu():
    """Select the GPU with the most free memory via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        free = [int(x) for x in out.split("\n")]
        return str(free.index(max(free)))
    except Exception:
        return "0"


# Must set CUDA_VISIBLE_DEVICES before torch is imported
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = find_free_gpu()

import io
import json
import copy
import random
import tempfile
import torch
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from spice import SPICE
from shapetalk import ShapeTalkDataset, PROMPTS, SOURCE_LATENTS, TARGET_LATENTS
from point_e.util.plotting import plot_point_cloud

torch.set_float32_matmul_precision("high")

TEXTS = "texts"
LABEL_H = 36


# ---------------------------------------------------------------------------
# Loss tracking callback: CSV log + matplotlib plot, no TensorBoard
# ---------------------------------------------------------------------------

class LossPlotCallback(Callback):
    """Tracks train + test loss, prints both at each validation, and saves
    a CSV log plus separate PNG plots for train and test loss.

    All data is accumulated in memory and written as complete files (no append),
    which is required for S3-mounted filesystems.
    """

    def __init__(self, output_dir):
        self.csv_path = os.path.join(output_dir, "loss.csv")
        self.train_plot_path = os.path.join(output_dir, "train_loss.png")
        self.test_plot_path = os.path.join(output_dir, "test_loss.png")
        self.train_losses = []     # (epoch, loss) — every epoch
        self.test_losses = []      # (epoch, loss) — at validation epochs only
        self._cur_epoch_batch = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._cur_epoch_batch.append(outputs["loss"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        if self._cur_epoch_batch:
            self.train_losses.append(
                (trainer.current_epoch, np.mean(self._cur_epoch_batch))
            )
            self._cur_epoch_batch = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = self.train_losses[-1][1] if self.train_losses else None

        test_loss_t = trainer.callback_metrics.get("test/loss")
        if test_loss_t is not None:
            self.test_losses.append((epoch, test_loss_t.item()))

        parts = []
        if train_loss is not None:
            parts.append(f"train={train_loss:.6f}")
        if test_loss_t is not None:
            parts.append(f"test={test_loss_t.item():.6f}")
        if parts:
            print(f"  [Epoch {epoch}] {', '.join(parts)}", flush=True)

        if not self.train_losses and not self.test_losses:
            return

        # Write CSV
        train_dict = dict(self.train_losses)
        test_dict = dict(self.test_losses)
        all_epochs = sorted(
            set(e for e, _ in self.train_losses)
            | set(e for e, _ in self.test_losses)
        )
        lines = ["epoch,train_loss,test_loss\n"]
        for ep in all_epochs:
            tr = train_dict.get(ep)
            te = test_dict.get(ep)
            tr_s = f"{tr:.6f}" if tr is not None else ""
            te_s = f"{te:.6f}" if te is not None else ""
            lines.append(f"{ep},{tr_s},{te_s}\n")
        with open(self.csv_path, "w") as f:
            f.writelines(lines)

        # Separate plots so each curve has its own y-axis scale
        if self.train_losses:
            self._save_plot(self.train_losses, "Train Loss", "tab:blue",
                            self.train_plot_path, epoch)
        if self.test_losses:
            self._save_plot(self.test_losses, "Test Loss", "tab:red",
                            self.test_plot_path, epoch)

    @staticmethod
    def _save_plot(data, title, color, path, epoch):
        epochs, losses = zip(*data)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, losses, linewidth=1.0, marker=".", markersize=3, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{title} (epoch {epoch})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Training model: extends SPICE with diffusion training + debug rendering
# ---------------------------------------------------------------------------

class SPICETrainer(SPICE):
    """
    SPICE model with training capabilities.

    Adds a diffusion training loss with classifier-free guidance dropout
    and a copy-reconstruction auxiliary objective. Periodically renders
    point cloud grids to disk for visual debugging.
    """

    def __init__(self, lr, copy_prob, copy_prompt, output_dir,
                 dev, cond_drop_prob=0.5, guidance_scale=3, batch_size=6):
        super().__init__(dev=dev, cond_drop_prob=cond_drop_prob,
                         guidance_scale=guidance_scale)
        self.lr = lr
        self.copy_prob = copy_prob
        self.copy_prompt = copy_prompt
        self.output_dir = output_dir
        self._batch_size = batch_size

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        prompts = batch[PROMPTS]
        source_latents = batch[SOURCE_LATENTS]
        target_latents = batch[TARGET_LATENTS]

        # Copy-reconstruction: with some probability, condition on the target
        # itself with the copy prompt (teaches identity preservation)
        if random.random() < self.copy_prob:
            texts = [self.copy_prompt] * len(prompts)
            guidance = target_latents
        else:
            texts = prompts
            guidance = source_latents

        loss = self.diffusion.training_losses(
            x_start=target_latents,
            model=self.model,
            t=self._sample_timesteps(),
            model_kwargs={TEXTS: texts, "guidance": guidance},
        )["loss"].mean()

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            self._save_debug_grid(batch, "val")
        elif dataloader_idx == 1:
            loss = self._eval_loss(batch)
            self.log("test/loss", loss, add_dataloader_idx=False)
        else:
            self._save_debug_grid(batch, "test")

    @torch.no_grad()
    def _eval_loss(self, batch):
        """Compute diffusion loss (edit objective) for evaluation."""
        prompts = batch[PROMPTS]
        source_latents = batch[SOURCE_LATENTS]
        target_latents = batch[TARGET_LATENTS]
        n = len(prompts)
        t = torch.randint(0, len(self.diffusion.betas), (n,), device=self.dev)
        return self.diffusion.training_losses(
            x_start=target_latents,
            model=self.model,
            t=t,
            model_kwargs={TEXTS: prompts, "guidance": source_latents},
        )["loss"].mean()

    # -- internals ----------------------------------------------------------

    def _sample_timesteps(self):
        """Sample random diffusion timesteps for the current batch."""
        return torch.tensor(
            random.sample(range(len(self.diffusion.betas)), self._batch_size)
        ).to(self.dev).detach()

    @torch.no_grad()
    def _save_debug_grid(self, batch, split):
        """
        Run inference on a batch and save a labelled image grid.

        Grid layout per sample row (4 columns):
            [source input] [edit output] [target input] [copy output]

        Input renders are cached on the instance so the expensive upsampler
        diffusion only runs once (on the first validation call).
        """
        save_dir = os.path.join(self.output_dir, "samples", split)
        os.makedirs(save_dir, exist_ok=True)

        prompts = batch[PROMPTS]
        source_latents = batch[SOURCE_LATENTS]
        target_latents = batch[TARGET_LATENTS]

        if not hasattr(self, "_ref_images"):
            self._ref_images = {}

        font = self._get_font()
        ref_path = os.path.join(save_dir, "reference.png")
        need_ref = not os.path.exists(ref_path)

        n = len(prompts)
        rows, ref_rows = [], []
        for i, (prompt, src, tgt) in enumerate(
            zip(prompts, source_latents, target_latents)
        ):
            print(f"  [{split}] Rendering sample {i+1}/{n}...", flush=True)

            # Cache input renders (only computed on first validation)
            key = f"{split}_{i}"
            if key not in self._ref_images:
                self._ref_images[key] = (
                    self._render_upsampled(src),
                    self._render_upsampled(tgt),
                )
            src_img, tgt_img = self._ref_images[key]

            edit_img = self._render_guided(prompt, src)
            copy_img = self._render_guided(self.copy_prompt, tgt)

            rows.append(self._hstack([
                self._add_label(src_img,  f"[{i}] Source (input)", font),
                self._add_label(edit_img, f"[{i}] Edit: {prompt[:45]}", font),
                self._add_label(tgt_img,  f"[{i}] Target (input)", font),
                self._add_label(copy_img, f"[{i}] Copy output", font),
            ]))

            if need_ref:
                ref_rows.append(self._hstack([
                    self._add_label(src_img, f"[{i}] Source (masked)", font),
                    self._add_label(tgt_img, f"[{i}] Target", font),
                ]))

        if need_ref and ref_rows:
            Image.fromarray(np.concatenate(ref_rows, axis=0)).save(ref_path)
            print(f"  Saved {split} reference -> {ref_path}")

        epoch = self.current_epoch
        grid_path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
        Image.fromarray(np.concatenate(rows, axis=0)).save(grid_path)
        print(f"  Saved {split} epoch {epoch} -> {grid_path}")

    def _render_upsampled(self, latent):
        """Upsample a low-res latent through the second diffusion stage and render."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = self.sampler.sample_batch(
                batch_size=1, model_kwargs={},
                prev_samples=latent.unsqueeze(0),
                blending_cache_dir=tmpdir,
            )
        return self._pc_to_image(samples)

    def _render_guided(self, prompt, guidance_latent):
        """Run full guided inference (base + upsampler) and render."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = self.sampler.sample_batch(
                batch_size=1,
                guidances=[guidance_latent.unsqueeze(0), None],
                model_kwargs={TEXTS: [prompt]},
                blending_cache_dir=tmpdir,
            )
        return self._pc_to_image(samples)

    def _pc_to_image(self, samples):
        """Convert sampler output to a numpy image via matplotlib (fast)."""
        pc = self.sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf))[..., :3]

    @staticmethod
    def _get_font():
        """Load a readable sized font with graceful fallbacks."""
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(path, 18)
            except OSError:
                continue
        try:
            return ImageFont.load_default(size=18)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _add_label(image, text, font):
        """Return image with a dark header bar containing the label text."""
        h, w = image.shape[:2]
        bar = Image.new("RGB", (w, LABEL_H), (40, 40, 40))
        ImageDraw.Draw(bar).text((8, 8), text, fill=(255, 255, 255), font=font)
        return np.concatenate([np.array(bar), image], axis=0)

    @staticmethod
    def _hstack(images):
        """Horizontally stack images, padding shorter ones to the tallest."""
        max_h = max(img.shape[0] for img in images)
        padded = []
        for img in images:
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.concatenate([img, pad], axis=0)
            padded.append(img)
        return np.concatenate(padded, axis=1)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--object", type=str, default="chair",
                   help="ShapeTalk object category (e.g. chair, table, lamp)")
    g.add_argument("--dataset_version", type=str, default="v1",
                   help="Dataset split subfolder under datasets/")
    g.add_argument("--shapetalk_dir", type=str, required=True,
                   help="Root path to the ShapeTalk dataset")
    g.add_argument("--subset_size", type=int, default=None,
                   help="Limit training set size (for quick debugging)")

    g = p.add_argument_group("training")
    g.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    g.add_argument("--epochs", type=int, default=500,
                   help="Number of training epochs")
    g.add_argument("--batch_size", type=int, default=10,
                   help="Samples per batch")
    g.add_argument("--accumulate_grad_batches", type=int, default=6,
                   help="Gradient accumulation steps (effective batch = batch_size x this)")
    g.add_argument("--copy_prob", type=float, default=0.1,
                   help="Fraction of steps using the copy-reconstruction objective")
    g.add_argument("--copy_prompt", type=str, default="COPY",
                   help="Prompt string used for the copy-reconstruction objective")
    g.add_argument("--cond_drop_prob", type=float, default=0.5,
                   help="Classifier-free guidance conditioning dropout rate")
    g.add_argument("--inpainting", action="store_true", default=True,
                   help="Mask the source point cloud by its labelled part")

    g = p.add_argument_group("evaluation")
    g.add_argument("--val_freq", type=int, default=2,
                   help="Run validation & save checkpoint every N epochs")
    g.add_argument("--num_val_samples", type=int, default=5,
                   help="Number of training-set samples to render at validation")
    g.add_argument("--num_test_samples", type=int, default=5,
                   help="Number of test-set samples to render at validation")
    g.add_argument("--test_subset_size", type=int, default=None,
                   help="Limit test set for loss computation (default: entire test set)")

    g = p.add_argument_group("output")
    g.add_argument("--output_dir", type=str, required=True,
                   help="Root output directory (checkpoints, logs, samples)")

    return p.parse_args()


def build_run_name(args):
    ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name = f"{ts}_{args.object}"
    name += f"_lr{args.lr}_copy{args.copy_prob}_cond{args.cond_drop_prob}"
    if args.subset_size is not None:
        name += f"_subset{args.subset_size}"
    return name


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

    run_name = build_run_name(args)
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Resolve dataset CSV paths (shipped with the repo)
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(repo_dir, "datasets", args.dataset_version, args.object)
    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    for path in (train_csv, test_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset CSV not found: {path}\n"
                f"Make sure datasets/{args.dataset_version}/{args.object}/ exists in the repo."
            )

    # Cache segmented point clouds in /tmp (fast local I/O, avoids NFS churn)
    cache_dir = os.path.join(tempfile.gettempdir(), "blendedpc_cache")

    # -- datasets -----------------------------------------------------------
    print(f"\nLoading training set from {train_csv} ...")
    train_dataset = ShapeTalkDataset(
        csv_path=train_csv,
        shapetalk_dir=args.shapetalk_dir,
        device=device,
        batch_size=args.batch_size,
        inpainting=args.inpainting,
        cache_dir=cache_dir,
        subset_size=args.subset_size,
    )
    print(f"  Training samples: {train_dataset.length}")

    val_render_dataset = copy.deepcopy(train_dataset)
    val_render_dataset.set_length(args.num_val_samples, length=args.num_val_samples)

    print(f"Loading test set from {test_csv} ...")
    test_loss_dataset = ShapeTalkDataset(
        csv_path=test_csv,
        shapetalk_dir=args.shapetalk_dir,
        device=device,
        batch_size=args.batch_size,
        inpainting=args.inpainting,
        cache_dir=cache_dir,
        subset_size=args.test_subset_size,
    )
    print(f"  Test samples (loss): {test_loss_dataset.length}")

    test_render_dataset = ShapeTalkDataset(
        csv_path=test_csv,
        shapetalk_dir=args.shapetalk_dir,
        device=device,
        batch_size=args.num_test_samples,
        inpainting=args.inpainting,
        cache_dir=cache_dir,
        subset_size=args.num_test_samples,
    )
    print(f"  Test samples (render): {test_render_dataset.length}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_render_loader = DataLoader(val_render_dataset, batch_size=args.num_val_samples)
    test_loss_loader = DataLoader(test_loss_dataset, batch_size=args.batch_size)
    test_render_loader = DataLoader(test_render_dataset, batch_size=args.num_test_samples)

    # -- model --------------------------------------------------------------
    model = SPICETrainer(
        lr=args.lr,
        dev=device,
        copy_prob=args.copy_prob,
        copy_prompt=args.copy_prompt,
        output_dir=output_dir,
        batch_size=args.batch_size,
        cond_drop_prob=args.cond_drop_prob,
    )

    # -- training -----------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        save_top_k=-1,
        save_weights_only=True,
        every_n_epochs=args.val_freq,
        dirpath=os.path.join(output_dir, "checkpoints"),
    )
    loss_cb = LossPlotCallback(output_dir)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, loss_cb],
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_freq,
        num_sanity_val_steps=0,
        logger=False,
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Validation every {args.val_freq} epochs")
    print(f"  Batch size {args.batch_size} x {args.accumulate_grad_batches} accumulation")
    print(f"  Copy prob {args.copy_prob}, cond drop {args.cond_drop_prob}")

    val_loaders = [val_render_loader, test_loss_loader, test_render_loader]

    # Initial validation to record epoch-0 performance before any training
    trainer.validate(model, dataloaders=val_loaders)

    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loaders)

    print(f"\nTraining complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
