"""Perceptual similarity metrics used by evaluate.py: FlexiCT-3D (whole-volume
CT embedding) and MedSigLIP (per-slice 2D embedding).

FlexiCT-3D: this file inlines the minimal subset of ricklisz/FlexiCT's
`Flexi_CT_3D` backbone (the DINOv3-style 3D ViT, patch_size=8, embed_dim=864,
depth=16) needed to embed a full CT/CBCT volume and compare it against
another via cosine similarity. No VLM/text-tower or nnU-Net downstream code
is included, so the only extra dependencies are torch + numpy (already
required elsewhere in this evaluation method).
Source: https://github.com/ricklisz/FlexiCT (pretraining branch)
License: MIT (FlexiCT code, Copyright (c) 2026 FlexiCT authors). The backbone
layers (attention, MLP, layer scale, RoPE, patch embeddings, transformer
block and backbone, i.e. the sections between the "DINOv3-derived code"
markers below) are derived from Meta's DINOv3 and remain subject to the
DINOv3 License; see third_party_licenses/DINOv3_LICENSE.txt and NOTICE.

MedSigLIP: loaded via `transformers.AutoModel` (see load_similarity_models
and MEDSIGLIP_MODEL_ID below) — a gated Hugging Face model.

The combined `SimilarityModels` class is what evaluate.py actually calls;
the two backbones are otherwise independent of each other.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn

logger = logging.getLogger("similarity_models")


# --------------------------------------------------------------------------
# Checkpoint resolution
# --------------------------------------------------------------------------

FLEXICT_3D_DEFAULT_CHECKPOINT: str | None = None


def _resolve_flexict_3d_checkpoint(explicit_checkpoint: str | None = None) -> str:
    """Resolve the FlexiCT-3D checkpoint path.

    Precedence: explicit argument, then FLEXICT_CHECKPOINT /
    FLEXICT_3D_CHECKPOINT env vars.
    """
    if explicit_checkpoint:
        return str(explicit_checkpoint)
    for env_var in ("FLEXICT_CHECKPOINT", "FLEXICT_3D_CHECKPOINT"):
        value = os.environ.get(env_var)
        if value:
            return value
    if FLEXICT_3D_DEFAULT_CHECKPOINT:
        return FLEXICT_3D_DEFAULT_CHECKPOINT
    raise ValueError(
        "FlexiCT-3D checkpoint path is required. Pass flexict_checkpoint_path, "
        "or set FLEXICT_CHECKPOINT or FLEXICT_3D_CHECKPOINT."
    )


# ==========================================================================
# BEGIN DINOv3-derived code
#
# The code from here to the matching "END DINOv3-derived code" marker is
# derived from Meta's DINOv3 (https://github.com/facebookresearch/dinov3),
# Copyright (c) Meta Platforms, Inc. and affiliates, via the FlexiCT
# implementation (https://github.com/ricklisz/FlexiCT). It is used and
# distributed under the terms of the DINOv3 License. A copy of the DINOv3
# License is provided in third_party_licenses/DINOv3_LICENSE.txt. This part of
# the file is NOT covered by the license of this repository.
# ==========================================================================

# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------

def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


# --------------------------------------------------------------------------
# FFN
# --------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --------------------------------------------------------------------------
# LayerScale / RoPE position embeddings
# --------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


class RopePositionEmbedding3D(nn.Module):
    """
    RoPE positional embedding for 3D grids with no mixing across axes (axial),
    and no learnable weights.

    Returns (sin, cos) each shaped [D*H*W, D_head], suitable for applying to q/k.
    """
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        # For 3 axes, per-axis rotary block is D_head//6
        assert embed_dim % (6 * num_heads) == 0, \
            "For 3D RoPE, embed_dim must be divisible by 6 * num_heads."
        both_periods = (min_period is not None) and (max_period is not None)
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Keep a persistent buffer so teacher.load_state_dict(student.state_dict()) works.
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 6, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        per_axis_block = self.D_head // 6  # 3 axes × per_axis_block = D_head//2

        if self.base is not None:
            # Classic exponential spectrum
            periods = self.base ** (
                2 * torch.arange(per_axis_block, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [per_axis_block]
        else:
            # Linearly spaced in log-period between min_period and max_period
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, per_axis_block, device=device, dtype=dtype)
            periods = base**exponents               # [1, base]
            periods = periods / base                # [1/base, 1]
            periods = periods * self.max_period     # [min_period, max_period]
        self.periods.data = periods

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        """
        Args:
            D, H, W: depth, height, width (integers)

        Returns:
            (sin, cos): two tensors of shape [D*H*W, D_head]
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # --- Build normalized coords in [-1, +1] for each axis ---
        if self.normalize_coords == "max":
            m = max(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / m  # [D]
            coords_h = torch.arange(0.5, H, **dd) / m  # [H]
            coords_w = torch.arange(0.5, W, **dd) / m  # [W]
        elif self.normalize_coords == "min":
            m = min(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / m
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        elif self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        # Meshgrid in (D, H, W) order; last dim stacks (d, h, w)
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"),
            dim=-1
        )  # [D, H, W, 3]
        coords = coords.flatten(0, 2)               # [DHW, 3]
        coords = 2.0 * coords - 1.0                 # shift [0,1] -> [-1,1]

        # --- Optional data-aug on coords (train-time only) ---
        if self.training and self.shift_coords is not None:
            shift_dhw = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords = coords + shift_dhw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_dhw = torch.empty(3, **dd).uniform_(jitter_min, jitter_max).exp()
            coords = coords * jitter_dhw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords = coords * rescale

        # --- Build rotary angles (axial, no mixing) ---
        # periods: [P] where P = D_head//6
        # coords:  [N, 3], broadcast => [N, 3, P]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [N, 3, P]
        angles = angles.flatten(1, 2)  # [N, 3*P] == [N, D_head//2]
        angles = angles.tile(2)        # [N, D_head]

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return (sin, cos)


# --------------------------------------------------------------------------
# Transformer block
# --------------------------------------------------------------------------

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=SelfAttention,
        ffn_layer=Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        x_out = []
        for x, rope in zip(x_list, rope_list):
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
            x_out.append(x_ffn)
        return x_out

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


# --------------------------------------------------------------------------
# Patch embeddings
# --------------------------------------------------------------------------

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D).

    Only used for patch_embed_2D, which the actual (always-3D-volume) forward
    pass never reaches — it's built solely because the checkpoint carries
    weights for it (joint 2D+3D pretraining)."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class PatchEmbed3D(nn.Module):
    """3D volume to patch embedding: (B,C,D,H,W) -> (B,N,D)."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, D, H, W = x.shape
        x = self.proj(x)  # B C D H W
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)  # B HWD C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, D, H, W, self.embed_dim)  # B D H W C
        return x

    def reset_parameters(self):
        # weight: [out_c, in_c/groups, kd, kh, kw]
        nn.init.kaiming_uniform_(self.proj.weight, a=0.0, mode="fan_in", nonlinearity="linear")
        if self.proj.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.proj.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.proj.bias, -bound, bound)


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Flexi_CT_Core(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed_2D = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.patch_embed_3D = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        assert embed_dim % (6 * num_heads) == 0, \
            f"embed_dim ({embed_dim}) must be divisible by 6*num_heads ({6*num_heads}) for 3D RoPE"

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        self.rope_embed_2D = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        self.rope_embed_3D = RopePositionEmbedding3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None

        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        if x.dim() == 5:
            x = self.patch_embed_3D(x)
            B, D, H, W, _ = x.shape
            x = x.flatten(1, 3)
        else:
            x = self.patch_embed_2D(x)
            B, H, W, _ = x.shape
            D = None
            x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device,
            )

        x = torch.cat([cls_token.expand(B, -1, -1), storage_tokens.expand(B, -1, -1), x], dim=1)

        return x, ((D, H, W) if D is not None else (H, W))

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)

        if len(rope[0]) == 3:
            rope_sincos = [self.rope_embed_3D(D=D, H=H, W=W) for D, H, W in rope]
        else:
            rope_sincos = [self.rope_embed_2D(H=H, W=W) for H, W in rope]

        for _, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos)

        n_storage_tokens = self.n_storage_tokens
        norm = self.norm
        cls_norm = self.cls_norm
        local_cls_norm = self.local_cls_norm
        untie_cls_and_patch_norms = self.untie_cls_and_patch_norms
        untie_global_and_local_cls_norm = self.untie_global_and_local_cls_norm
        training = self.training

        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if untie_cls_and_patch_norms or untie_global_and_local_cls_norm:
                if untie_global_and_local_cls_norm and training and idx == 1:
                    x_norm_cls_reg = local_cls_norm(x[:, : n_storage_tokens + 1])
                elif untie_cls_and_patch_norms:
                    x_norm_cls_reg = cls_norm(x[:, : n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = norm(x[:, : n_storage_tokens + 1])
                x_norm_patch = norm(x[:, n_storage_tokens + 1 :])
            else:
                x_norm = norm(x)
                x_norm_cls_reg = x_norm[:, : n_storage_tokens + 1]
                x_norm_patch = x_norm[:, n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def flexi_ct_backbone_base(patch_size=8, **kwargs):
    return Flexi_CT_Core(
        patch_size=patch_size,
        embed_dim=864,
        depth=16,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )


# ==========================================================================
# END DINOv3-derived code
# ==========================================================================

# --------------------------------------------------------------------------
# Flexi_CT_3D wrapper (checkpoint loading + public forward API)
# --------------------------------------------------------------------------

_BACKBONE_KWARGS = dict(
    patch_size=8,
    in_chans=1,
    n_storage_tokens=4,
    qkv_bias=False,
    mask_k_bias=True,
    drop_path_rate=0.2,
    layerscale_init=1e-5,
)


def _load_teacher_into_backbone(backbone: nn.Module, ckpt_path: str) -> None:
    """Checkpoint has keys like `backbone.patch_embed_2D.*` plus DINO/iBOT head keys.
    Strip the prefix and drop the head keys before loading."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["teacher"]
    sd = {
        k[len("backbone."):]: v
        for k, v in sd.items()
        if k.startswith("backbone.") and "ibot" not in k and "dino_head" not in k
    }
    missing, unexpected = backbone.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: missing={missing} unexpected={unexpected}")


class Flexi_CT_3D(nn.Module):
    def __init__(self, checkpoint_path: str | None = None, device: str | torch.device = "cuda"):
        super().__init__()
        self.backbone = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
        _load_teacher_into_backbone(self.backbone, _resolve_flexict_3d_checkpoint(checkpoint_path))
        self.to(device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, 1, D, H, W] with D, H, W divisible by patch_size (8)."""
        out = self.backbone(x, is_training=True)
        return {
            "cls_token": out["x_norm_clstoken"],          # [B, D]
            "patch_tokens": out["x_norm_patchtokens"],    # [B, N, D]
        }


# --------------------------------------------------------------------------
# Similarity metric used by evaluate.py
# --------------------------------------------------------------------------

# Flexi_CT_3D uses patch_size=8, so each side of the volume fed to it must be
# divisible by 8. 128 matches the cube size used in FlexiCT's own inference
# demo (16^3 = 4096 patch tokens).
FLEXICT_VOLUME_SIZE = 128

MEDSIGLIP_MODEL_ID = "google/medsiglip-448"
MEDSIGLIP_IMAGE_SIZE = (448, 448)
MEDSIGLIP_WINDOW_CLIPS = ((-1024.0, 1024.0), (-135.0, 215.0), (0.0, 80.0))

@dataclass
class SimilarityModels:
    medsiglip_visual: torch.nn.Module
    flexict_visual: torch.nn.Module
    device: str = "cuda"
    batch_size: int = 16
    dtype: torch.dtype = torch.float32  # model/inference precision; embeddings are always returned as fp32
    mask_background_hu: float = -1024.0  # air, matches the HU clamp floor below
    image_size = MEDSIGLIP_IMAGE_SIZE  # 2D slice size for MedSigLIP
    volume_size = (FLEXICT_VOLUME_SIZE,) * 3  # 3D volume size for FlexiCT-3D

    def __post_init__(self):
        # SigLIP normalization stats (maps [0, 1] -> [-1, 1])
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

    def _apply_mask(self, volume: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Replace everything outside the FOV mask with air, so the similarity
        models never see out-of-field background (couch, air ring, etc.) —
        only the actual reconstructed anatomy is compared."""
        if mask is None:
            return volume
        if mask.shape != volume.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume shape {volume.shape}"
            )
        return np.where(mask.astype(bool), volume, self.mask_background_hu)

    def _prepare_medsiglip(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Cheap, whole-volume, numpy-only prep (mask only).
        Deliberately stops here rather than also windowing/resizing the full
        volume to torch tensors: that step is done per-batch in
        _embed_medsiglip instead, so the (3x448x448-per-slice, fp32) prepared
        tensor for the whole volume is never all resident in memory at once —
        only one batch's worth is. For a large volume this is the difference
        between ~600MB and ~35MB of peak extra memory for that tensor alone,
        which matters on memory-constrained (e.g. SLURM --mem=10GB) nodes."""
        volume = self._apply_mask(volume, mask)
        return volume

    def _medsiglip_batch_to_tensor(self, volume_batch: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(volume_batch.astype(np.float32)).unsqueeze(1).to(self.device)
        # HU -> pseudo-RGB via 3 CT windows (wide/mediastinum/brain), each
        # channel independently clamped + min-max normalized to [0, 1].
        x = torch.cat(
            [(x.clamp(lo, hi) - lo) / (hi - lo) for lo, hi in MEDSIGLIP_WINDOW_CLIPS],
            dim=1,
        )
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def _prepare_flexict(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        volume = self._apply_mask(volume, mask)
        # Clip HU then z-score the whole volume, matching FlexiCT's own
        # inference-demo preprocessing, then resize (not crop) to a fixed
        # cube so the metric sees the whole reconstructed volume.
        x = torch.from_numpy(volume.astype(np.float32)).to(self.device)
        x = x.clamp(-1000.0, 1000.0)
        mean, std = x.mean(), x.std()
        x = (x - mean) / std if std > 1e-6 else (x - mean)
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        if x.shape[-3:] != self.volume_size:
            x = F.interpolate(x, size=self.volume_size, mode="trilinear", align_corners=False)
        return x

    def _embed_medsiglip(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        volume = self._prepare_medsiglip(volume, mask)
        embeddings: List[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, volume.shape[0], self.batch_size):
                batch = self._medsiglip_batch_to_tensor(volume[start : start + self.batch_size])
                feat = self.medsiglip_visual(batch.to(self.dtype)).pooler_output
                embeddings.append(feat.float())
        return torch.cat(embeddings, dim=0)

    def _embed_flexict(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        x = self._prepare_flexict(volume, mask)
        with torch.inference_mode():
            feat = self.flexict_visual(x.to(self.dtype))["cls_token"]
        return feat.float()

    def cosine_similarity(
        self, reference: np.ndarray, prediction: np.ndarray, mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Return cosine similarity from MedSigLIP (2D, per-slice cosine similarity across slices) and FlexiCT-3D (whole-volume).

        If `mask` is given (same shape as reference/prediction), both volumes
        are masked to air outside the FOV before embedding, so out-of-field
        background doesn't dilute the similarity score.
        """
        ref_ms = self._embed_medsiglip(reference, mask)
        pred_ms = self._embed_medsiglip(prediction, mask)
        cs_ms = F.cosine_similarity(ref_ms, pred_ms, dim=-1).mean().item()

        ref_flexict = self._embed_flexict(reference, mask)
        pred_flexict = self._embed_flexict(prediction, mask)
        cs_flexict = F.cosine_similarity(ref_flexict, pred_flexict, dim=-1).mean().item()
        return float(cs_ms), float(cs_flexict)

    def embed_reference(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Embed a ground-truth volume once, so the result can be saved to disk
        and reused across every evaluation run instead of recomputing it per
        submission. See precompute_ground_truth_embeddings.py.
        """
        return {
            "medsiglip": self._embed_medsiglip(volume, mask).cpu().numpy(),
            "flexict": self._embed_flexict(volume, mask).cpu().numpy(),
        }

    def cosine_similarity_from_ref_embeddings(
        self,
        ref_embeddings: Dict[str, np.ndarray],
        prediction: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Like cosine_similarity, but takes a precomputed reference embedding
        (from embed_reference) instead of re-embedding the ground truth.

        `mask`, if given, is applied to `prediction` only — the reference
        embedding was already masked (or not) when it was precomputed, so
        this must match whatever precompute_ground_truth_embeddings.py used.
        """
        if "medsiglip" not in ref_embeddings:
            raise KeyError(
                "gt_embeddings.npz has no 'medsiglip' entry Re-run "
                "precompute_ground_truth_embeddings.py to regenerate it."
            )
        ref_ms = torch.from_numpy(ref_embeddings["medsiglip"]).to(self.device)
        pred_ms = self._embed_medsiglip(prediction, mask)
        if ref_ms.shape != pred_ms.shape:
            raise ValueError(
                f"Cached MedSigLIP reference embedding has shape {tuple(ref_ms.shape)}, "
                f"but this run's prediction embeds to {tuple(pred_ms.shape)}. This means "
                "gt_embeddings.npz was precomputed with a incompatible embedding "
                "re-run precompute_ground_truth_embeddings.py to regenerate it."
            )
        cs_ms = F.cosine_similarity(ref_ms, pred_ms, dim=-1).mean().item()

        ref_flexict = torch.from_numpy(ref_embeddings["flexict"]).to(self.device)
        pred_flexict = self._embed_flexict(prediction, mask)
        cs_flexict = F.cosine_similarity(ref_flexict, pred_flexict, dim=-1).mean().item()
        return float(cs_ms), float(cs_flexict)



def load_similarity_models(
        flexict_checkpoint_path: Optional[str] = None,
        medsiglip_cache_dir: Optional[str] = "/opt/app/model_weights",
        device: str = "cuda",
        dtype: Optional[str] = None,
        ) -> SimilarityModels:
    from transformers import AutoModel

    medsiglip_source = os.environ.get("MEDSIGLIP_LOCAL_DIR") or MEDSIGLIP_MODEL_ID
    medsiglip_model = AutoModel.from_pretrained(
        medsiglip_source,
        cache_dir=medsiglip_cache_dir if not os.environ.get("MEDSIGLIP_LOCAL_DIR") else None,
    )

    dtype_name = dtype or os.environ.get("EMBED_DTYPE", "float16")
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype_name]

    medsiglip_visual = medsiglip_model.vision_model.to(device=device, dtype=torch_dtype).eval()
    for p in medsiglip_visual.parameters():
        p.requires_grad = False

    flexict_visual = Flexi_CT_3D(checkpoint_path=flexict_checkpoint_path, device=device).to(torch_dtype).eval()
    for p in flexict_visual.parameters():
        p.requires_grad = False

    return SimilarityModels(
        medsiglip_visual=medsiglip_visual,
        flexict_visual=flexict_visual,
        device=device,
        dtype=torch_dtype,
    )
