import torch
import pytorch_lightning as pl
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config

TEXTS = "texts"
UPSAMPLE = "upsample"
NUM_POINTS_LOW = 1024
NUM_POINTS_HIGH = 4096
MODEL_NAME = "base40M-textvec"


class SPICE(pl.LightningModule):
    def __init__(
        self,
        dev: torch.device,
        guidance_scale: int = 3,
        cond_drop_prob: float = 0.5,
    ):
        super().__init__()
        self.dev = dev
        self._init_model(cond_drop_prob, guidance_scale)

    def _init_model(self, cond_drop_prob, guidance_scale):
        self.diffusion = diffusion_from_config(DIFFUSION_CONFIGS[MODEL_NAME])
        upsampler_diffusion = diffusion_from_config(
            DIFFUSION_CONFIGS[UPSAMPLE])
        config = MODEL_CONFIGS[MODEL_NAME]
        config["cond_drop_prob"] = cond_drop_prob
        self.model = model_from_config(config, self.dev)
        self.model.load_state_dict(load_checkpoint(MODEL_NAME, self.dev))
        self.model.create_control_layers()
        upsampler_model = model_from_config(MODEL_CONFIGS[UPSAMPLE], self.dev)
        upsampler_model.eval()
        upsampler_model.load_state_dict(load_checkpoint(UPSAMPLE, self.dev))
        self.sampler = PointCloudSampler(
            device=self.dev,
            guidance_scale=[guidance_scale, 0.0],
            aux_channels=["R", "G", "B"],
            model_kwargs_key_filter=(TEXTS, ""),
            models=[self.model, upsampler_model],
            num_points=[NUM_POINTS_LOW, NUM_POINTS_HIGH - NUM_POINTS_LOW],
            diffusions=[self.diffusion, upsampler_diffusion],
        )
