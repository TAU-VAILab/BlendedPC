"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""

from abc import abstractmethod
import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .checkpoint import checkpoint
from .pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType
from .util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class QKVMultiheadAttention(nn.Module):
    def __init__(self,
                 *,
                 device: torch.device,
                 dtype: torch.dtype,
                 heads: int,
                 n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        # batch size, num of tokens, embedding size of each token
        bs, n_ctx, width = qkv.shape

        # splitted the embedding size into heads, 3 for q, k, v
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))

        # qkv.shape = (bs, n_ctx, heads, 3 * attn_ch)
        qkv = qkv.view(bs, n_ctx, self.heads, -1)

        # q.shape = k.shape = v.shape = (bs, n_ctx, heads, attn_ch)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        # weight_{i1,i2,i3,i4} = sum_{​t=1->attn_ch} q_{i1,i3,i2,t} * k_{i1,i4,i2,t}
        # weight.shape = (bs, heads, n_ctx, n_ctx)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)

        # tmp_{i1,i2,i3,t} = sum_{​t=1->n_ctx} weight_{i1,i3,i2,t} * v_{i1,t,i3,i4}
        # tmp.shape = (bs, n_ctx, heads, attn_ch)
        # output = tmp.reshape(bs, n_ctx, -1)
        # output.shape = (bs, n_ctx, heads * attn_ch)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class MultiheadAttentionBase(nn.Module):
    def __init__(self,
                 *,
                 device: torch.device,
                 dtype: torch.dtype,
                 n_ctx: int, width: int,
                 heads: int,
                 init_scale: float,
                 c_proj: nn.Linear = None,
                 attention: QKVMultiheadAttention = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.init_scale = init_scale
        self.c_proj = self.build_linear(width) if c_proj is None else c_proj
        self.attention = QKVMultiheadAttention(
            device=device, dtype=dtype, heads=heads, n_ctx=n_ctx) if attention is None else attention

    def forward(self, x, x_guidance):
        qkv = self.build_qkv(x, x_guidance)
        x = checkpoint(self.attention, (qkv,), (), True)
        x = self.c_proj(x)
        return x

    @abstractmethod
    def build_qkv(self, x, x_guidance):
        raise NotImplementedError

    def build_linear(self, out_features):
        linear = nn.Linear(self.width, out_features,
                           device=self.device, dtype=self.dtype)
        init_linear(linear, self.init_scale)
        return linear


class MultiheadAttention(MultiheadAttentionBase):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.c_qkv = self.build_linear(self.width * 3)

    def build_qkv(self, x, x_guidance):
        assert x_guidance is None, "x_guidance is not None"
        return self.c_qkv(x)


class MultiheadCrossEntityAttention(MultiheadAttentionBase):
    def __init__(self,
                 c_qkv: nn.Linear,
                 **kwargs):
        super().__init__(**kwargs)
        self.c_kv = self.build_linear(self.width * 2)
        self.c_q1 = self.build_linear(self.width)
        self.c_q2 = self.build_linear(self.width)
        self.zero_conv = self.build_zero_conv()
        self.init_weights(c_qkv)

    @classmethod
    def from_multihead_attention(cls, attn: MultiheadAttentionBase):
        attn_copy = copy.deepcopy(attn)
        kwargs = dict(
            device=attn_copy.device,
            dtype=attn_copy.dtype,
            n_ctx=attn_copy.n_ctx,
            width=attn_copy.width,
            heads=attn_copy.heads,
            init_scale=attn_copy.init_scale,
            c_proj=attn_copy.c_proj,
            attention=attn_copy.attention,
        )
        return cls(attn_copy.c_qkv, **kwargs)

    def build_qkv(self, x, x_guidance):
        assert x_guidance is not None, "x_guidance is None"
        kv = self.c_kv(x)
        q1 = self.c_q1(x)
        q2 = self.c_q2(x_guidance)
        q = q1 + self.zero_conv(q2)
        return torch.cat([q, kv], dim=-1)

    def build_zero_conv(self):
        return nn.Conv1d(in_channels=self.n_ctx,
                         out_channels=self.n_ctx,
                         kernel_size=1,
                         device=self.device,
                         dtype=self.dtype)

    def init_weights(self, c_qkv):
        with torch.no_grad():
            width = self.width
            self.c_kv.weight.copy_(c_qkv.weight[width:, :])
            self.c_kv.bias.copy_(c_qkv.bias[width:])
            self.c_q1.weight.copy_(c_qkv.weight[:width, :])
            self.c_q1.bias.copy_(c_qkv.bias[:width])
            self.c_q2.weight.copy_(c_qkv.weight[:width, :])
            self.c_q2.bias.copy_(c_qkv.bias[:width])
            self.zero_conv.weight.zero_()
            self.zero_conv.bias.zero_()


class MLP(nn.Module):
    def __init__(self,
                 *,
                 device: torch.device,
                 dtype: torch.dtype,
                 width: int,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlockBase(nn.Module):
    def __init__(self,
                 *,
                 device: torch.device,
                 dtype: torch.dtype,
                 n_ctx: int,
                 width: int,
                 heads: int,
                 init_scale: float = 1.0,
                 attn: MultiheadAttentionBase = None,
                 ln_1: nn.LayerNorm = None,
                 mlp: MLP = None,
                 ln_2: nn.LayerNorm = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.init_scale = init_scale
        self.attn = MultiheadAttention(
            device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads, init_scale=init_scale) if attn is None else attn
        self.ln_1 = nn.LayerNorm(width, device=device,
                                 dtype=dtype) if ln_1 is None else ln_1
        self.mlp = MLP(device=device, dtype=dtype,
                       width=width, init_scale=init_scale) if mlp is None else mlp
        self.ln_2 = nn.LayerNorm(width, device=device,
                                 dtype=dtype) if ln_2 is None else ln_2

    def forward(self, x, x_guidance):
        x = x + self.apply_attention(x, x_guidance)
        x = x + self.mlp(self.ln_2(x))
        return x

    @abstractmethod
    def apply_attention(self, x, x_guidance):
        raise NotImplementedError


class ResidualAttentionBlock(ResidualAttentionBlockBase):
    def apply_attention(self, x, x_guidance):
        assert x_guidance is None, "x_guidance is not None"
        return self.attn(self.ln_1(x), x_guidance)


class ResidualCrossEntityAttentionBlock(ResidualAttentionBlockBase):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiheadCrossEntityAttention.from_multihead_attention(
            self.attn)
        self.ln1_guidance = copy.deepcopy(self.ln_1)

    @classmethod
    def from_resblock(cls, resblock: ResidualAttentionBlockBase):
        resblock_copy = copy.deepcopy(resblock)
        kwargs = dict(
            device=resblock_copy.device,
            dtype=resblock_copy.dtype,
            n_ctx=resblock_copy.n_ctx,
            width=resblock_copy.width,
            heads=resblock_copy.heads,
            init_scale=resblock_copy.init_scale,
            attn=resblock_copy.attn,
            ln_1=resblock_copy.ln_1,
            mlp=resblock_copy.mlp,
            ln_2=resblock_copy.ln_2,
        )
        return cls(**kwargs)

    def apply_attention(self, x, x_guidance):
        assert x_guidance is not None, "x_guidance is None"
        return self.attn(self.ln_1(x), self.ln1_guidance(x_guidance))


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.first_ctrl_idx = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, **kwargs):
        for i, resblock in enumerate(self.resblocks):
            if i < self.first_ctrl_idx:
                x = resblock(x, None)
            else:
                ctrl_resblock = self.ctrl_resblocks[i - self.first_ctrl_idx]
                x = ctrl_resblock(x, kwargs['guidance'])
        return x

    def create_control_layers(self, ctrl_layers=None):
        self.ctrl_layers = self.layers if ctrl_layers is None else ctrl_layers
        assert self.ctrl_layers <= self.layers, f"ctrl_layers ({self.ctrl_layers}) must be <= layers ({self.layers})"
        self.first_ctrl_idx = self.layers - self.ctrl_layers
        self.ctrl_resblocks = nn.ModuleList(
            [
                ResidualCrossEntityAttentionBlock.from_resblock(
                    self.resblocks[i + self.first_ctrl_idx]
                )
                for i in range(self.ctrl_layers)
            ]
        )


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.guided = False
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale *
            math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(
            input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(
            width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]], **kwargs
    ) -> torch.Tensor:
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if self.guided:
            kwargs['guidance'] = self.preprocess(
                kwargs['guidance'].reshape(x.shape), cond_as_token, extra_tokens)
        h = self.preprocess(x, cond_as_token, extra_tokens)
        h = self.backbone(h, **kwargs)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens):]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)

    def preprocess(self, x, cond_as_token, extra_tokens):
        h = self.input_proj(x.permute(0, 2, 1))
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)
        h = self.ln_pre(h)
        return h

    def create_control_layers(self, ctrl_layers=None):
        self.guided = True
        self.backbone.create_control_layers(ctrl_layers)
        for name, param in self.named_parameters():
            if name.startswith('backbone'):
                param.requires_grad = True
            elif name.startswith('ln_post'):
                param.requires_grad = True
            elif name.startswith('output_proj'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    @staticmethod
    def is_trainable_param(name):
        return (
            name.startswith("backbone")
            or name.startswith("ln_post")
            or name.startswith("output_proj")
        )

    def freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def print_parameters_status(self):
        for name, param in self.named_parameters():
            print(
                f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}")


class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype,
                         n_ctx=n_ctx + int(token_cond), **kwargs)
        self.n_ctx = n_ctx
        self.token_cond = token_cond
        self.clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device, cache_dir=cache_dir)
        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, texts=model_kwargs['texts']))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
        injection_step: bool = False,
        **kwargs,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param texts: a batch of texts to condition on.
        :param embeddings: a batch of CLIP embeddings to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        clip_out = self.clip(batch_size=len(x), images=images,
                             texts=texts, embeddings=embeddings)
        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        # Rescale the features to have unit variance
        clip_out = math.sqrt(clip_out.shape[1]) * clip_out

        clip_embed = self.clip_embed(clip_out)
        if injection_step:
            assert clip_embed.shape[0] == 4
            clip_embed[1] = clip_embed[0]
            clip_embed[3] = clip_embed[2]

        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]
        return self._forward_with_cond(x, cond, **kwargs)


class CLIPImageGridPointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(device=device, dtype=dtype,
                         n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx
        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,
                                  ), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim,
                      self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        _ = batch_size
        with torch.no_grad():
            return dict(embeddings=self.clip.embed_images_grid(model_kwargs["images"]))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert images is not None or embeddings is not None, "must specify images or embeddings"
        assert images is None or embeddings is None, "cannot specify both images and embeddings"
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        else:
            clip_out = embeddings

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond), (clip_embed, True)]
        return self._forward_with_cond(x, cond)


class UpsamplePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond_input_channels: Optional[int] = None,
        cond_ctx: int = 1024,
        n_ctx: int = 4096 - 1024,
        channel_scales: Optional[Sequence[float]] = None,
        channel_biases: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + cond_ctx, **kwargs)
        self.n_ctx = n_ctx
        self.cond_input_channels = cond_input_channels or self.input_channels
        self.cond_point_proj = nn.Linear(
            self.cond_input_channels, self.backbone.width, device=device, dtype=dtype
        )

        self.register_buffer(
            "channel_scales",
            torch.tensor(channel_scales, dtype=dtype, device=device)
            if channel_scales is not None
            else None,
        )
        self.register_buffer(
            "channel_biases",
            torch.tensor(channel_biases, dtype=dtype, device=device)
            if channel_biases is not None
            else None,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, low_res: torch.Tensor):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        cond = [(t_embed, self.time_token_cond), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)

    def _embed_low_res(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_scales is not None:
            x = x * self.channel_scales[None, :, None]
        if self.channel_biases is not None:
            x = x + self.channel_biases[None, :, None]
        return self.cond_point_proj(x.permute(0, 2, 1))


class CLIPImageGridUpsamplePointDiffusionTransformer(UpsamplePointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096 - 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(device=device, dtype=dtype,
                         n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx

        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,
                                  ), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim,
                      self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "images" not in model_kwargs:
            zero_emb = torch.zeros(
                [batch_size, self.clip.grid_feature_dim, self.clip.grid_size**2],
                device=next(self.parameters()).device,
            )
            return dict(embeddings=zero_emb, low_res=model_kwargs["low_res"])
        with torch.no_grad():
            return dict(
                embeddings=self.clip.embed_images_grid(model_kwargs["images"]),
                low_res=model_kwargs["low_res"],
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        low_res: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
        guidance: Optional[torch.Tensor] = None,
        injection_step: bool = False,
    ):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        elif embeddings is not None:
            clip_out = embeddings
        else:
            # Support unconditional generation.
            clip_out = torch.zeros(
                [len(x), self.clip.grid_feature_dim, self.clip.grid_size**2],
                dtype=x.dtype,
                device=x.device,
            )

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond),
                (clip_embed, True), (low_res_embed, True)]
        kwargs = dict(guidance=guidance) if guidance is not None else {}
        return self._forward_with_cond(x, cond, **kwargs)
