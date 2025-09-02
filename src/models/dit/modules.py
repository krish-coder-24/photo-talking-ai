"""
Modules, Inculude:
    - Attention: Attention module used in transformers
    - MLP: MLP module used in transformers
    - PositionalEncoding: Positional encoding module used in transformers
    - ROPE: ROPE module used in transformers
"""
import os
import copy
import logging
import math
import numbers
from itertools import repeat
from collections import OrderedDict
import collections.abc
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List, Final
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange, repeat

from .posemb_layers import apply_rotary_emb
try:
    from apex.normalization.fused_layer_norm import fused_layer_norm_affine
    has_apex = True
except ImportError:
    has_apex = False

try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine, fused_rms_norm
    has_apex_rmsnorm = True
except ImportError:
    has_apex_rmsnorm = False

has_torch_rms_norm = hasattr(F, 'rms_norm')

from .config import use_fused_attn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    # pads a tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n


# RMS_NORM
def get_autocast_dtype(device: str = 'cuda'):
    try:
        return torch.get_autocast_dtype(device)
    except (AttributeError, TypeError):
        # dispatch to older device specific fns, only covering cuda/cpu devices here
        if device == 'cpu':
            return torch.get_autocast_cpu_dtype()
        else:
            assert device == 'cuda'
            return torch.get_autocast_gpu_dtype()


def is_autocast_enabled(device: str = 'cuda'):
    try:
        return torch.is_autocast_enabled(device)
    except TypeError:
        # dispatch to older device specific fns, only covering cuda/cpu devices here
        if device == 'cpu':
            return torch.is_autocast_cpu_enabled()
        else:
            assert device == 'cuda'
            return torch.is_autocast_enabled()  # defaults cuda (only cuda on older pytorch)


_USE_FAST_NORM = False  # defaulting to False for now
def is_fast_norm():
    return _USE_FAST_NORM


def rms_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    norm_ndim = len(normalized_shape)
    v = x.pow(2)
    if torch.jit.is_scripting():
        # ndim = len(x.shape)
        # dims = list(range(ndim - norm_ndim, ndim))  # this doesn't work on pytorch <= 1.13.x
        # NOTE -ve dims cause torchscript to crash in some cases, out of options to work around
        assert norm_ndim == 1
        v = torch.mean(v, dim=-1).unsqueeze(-1)  # ts crashes with -ve dim + keepdim=True
    else:
        dims = tuple(range(-1, -norm_ndim - 1, -1))
        v = torch.mean(v, dim=dims, keepdim=True)
    x = x * torch.rsqrt(v + eps)
    if weight is not None:
        x = x * weight
    return x


def fast_rms_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # this must be by itself, cannot merge with has_apex_rmsnorm
        return rms_norm(x, normalized_shape, weight, eps)

    if has_apex_rmsnorm:
        if weight is None:
            return fused_rms_norm(x, normalized_shape, eps)
        else:
            return fused_rms_norm_affine(x, weight, normalized_shape, eps)

    if is_autocast_enabled(x.device.type):
        # normally native AMP casts LN inputs to float32
        # apex LN does not, this is behaving like Apex
        dt = get_autocast_dtype(x.device.type)
        x, weight = x.to(dt), weight.to(dt)

    with torch.autocast(device_type=x.device.type, enabled=False):
        if has_torch_rms_norm:
            x = F.rms_norm(x, normalized_shape, weight, eps)
        else:
            x = rms_norm(x, normalized_shape, weight, eps)

    return x


class RMSNorm(nn.Module):
    """ RMSNorm w/ fast (apex) norm if available
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    _fast_norm: bool

    def __init__(self, channels, eps=1e-6, elementwise_affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
        # Since there is no built-in PyTorch impl, always use APEX RmsNorm if is installed.
        if self._fast_norm:
            x = fast_rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            x = rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return x


class Mlp(nn.Module):
    """ MLP module used in transformers
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #bias = to_2tuple(bias)
        #drop_probs = to_2tuple(drop)
        bias = [bias, bias]
        drop_probs = [drop, drop]
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class MMsingle_attention(nn.Module):
    """
    Self-Attention module used in transformers
    """
    fused_attn: Final[bool]

    def __init__(
        self, dim: int, 
        num_heads: int = 8, 
        proj_bias: bool = True,
        attn_drop: float = 0., 
        proj_drop: float = 0.,
        qkv_bias: bool = False, 
        qk_norm: Optional[str] = "rms_norm", 
        **block_kwargs
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.attn_drop = nn.Dropout(attn_drop)
        if qk_norm is None:
            self.xs_q_norm = nn.Identity()
            self.xs_k_norm = nn.Identity()
        elif qk_norm == "rms_norm":
            self.xs_q_norm = RMSNorm(self.head_dim, eps=1e-5)
            self.xs_k_norm = RMSNorm(self.head_dim, eps=1e-5)
        elif qk_norm == "layer_norm":
            self.xs_q_norm = nn.LayerNorm(dim, eps=1e-5)
            self.xs_k_norm = nn.LayerNorm(dim, eps=1e-5)
        else:
            raise ValueError(f"Unsupported qk_norm method: {qk_norm}")

    def forward(self, txt_len,x: torch.Tensor, mask: Optional[torch.Tensor] = None,causal: bool = False,freqs_cis=None,freqs_cis2=None) -> torch.Tensor:
        B, N1, C = x.shape
        xs_qkv = x.reshape(B, N1, 3, -1)
        xs_q, xs_k, xs_v = xs_qkv.permute(2, 0, 1, 3).unbind(0)
        N2=N1//4
        q = xs_q.view(B, N1, self.num_heads, self.head_dim)
        k = xs_k.view(B, N1, self.num_heads, self.head_dim)
        v = xs_v.view(B, N1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.xs_q_norm(q), self.xs_k_norm(k)  
        if freqs_cis is not None or freqs_cis2 is not None:
            img_q, txt_q = q[:, :txt_len, :, :], q[:, txt_len:, :, :]
            img_k, txt_k = k[:, :txt_len, :, :], k[:, txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq.transpose(1, 2), img_kk.transpose(1, 2)
            if freqs_cis2 is not None:
                txt_qq, txt_kk = apply_rotary_emb(txt_q, txt_k, freqs_cis2, head_first=False) 
                assert (
                    txt_qq.shape == txt_q.shape and txt_kk.shape == txt_k.shape
                ), f"img_kk: {txt_q.shape}, img_q: {txt_q.shape}, img_kk: {txt_kk.shape}, img_k: {txt_k.shape}"
                txt_q, txt_k = txt_qq, txt_kk
            q = torch.cat((img_q, txt_q.transpose(1, 2)), dim=2)
            k = torch.cat((img_k, txt_k.transpose(1, 2)), dim=2)
        if mask is not None:
            mask = mask[:, None, None, :].expand(-1, self.num_heads,N1, -1)  # (B, num_heads, N, N)
            mask = mask.to(dtype=q.dtype)
        if causal:
            mask2 = torch.ones((N2+3*N2,N2+3*N2), dtype=torch.bool, device=v.device)
            mask2[-N2-N2:, :N2]= 0
            mask2[-N2-N2:-N2,-N2:]=0
            mask2[-N2:,-N2-N2:-N2]=0
            mask = mask2.to(dtype=torch.bool)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N1, -1)
        return x
class MMfour_attention(nn.Module):
    """
    Self-Attention module used in transformers
    """
    fused_attn: Final[bool]

    def __init__(
        self, dim: int, 
        num_heads: int = 8, 
        proj_bias: bool = True,
        attn_drop: float = 0., 
        proj_drop: float = 0.,
        qkv_bias: bool = False, 
        qk_norm: Optional[str] = "rms_norm", 
        **block_kwargs
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv_xs = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_au1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_au2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_au3 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm is None:
            self.xs_q_norm = nn.Identity()
            self.xs_k_norm = nn.Identity()
            self.au_q_norm = nn.Identity()
            self.au_k_norm = nn.Identity()
        elif qk_norm == "rms_norm":
            self.xs_q_norm = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.xs_k_norm = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.au_q_norm1 = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.au_k_norm1 = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)

            self.au_q_norm2 = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.au_k_norm2 = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)

            self.au_q_norm3 = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.au_k_norm3= RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
        elif qk_norm == "layer_norm":
            self.xs_q_norm = nn.LayerNorm(dim, eps=1e-5)
            self.xs_k_norm = nn.LayerNorm(dim, eps=1e-5)
            self.au_q_norm = nn.LayerNorm(dim, eps=1e-5)
            self.au_k_norm = nn.LayerNorm(dim, eps=1e-5)
        else:
            raise ValueError(f"Unsupported qk_norm method: {qk_norm}")

        self.attn_drop = nn.Dropout(attn_drop)
        self.xs_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.au_proj1 =  nn.Linear(dim, dim, bias=proj_bias)
        self.au_proj2 =  nn.Linear(dim, dim, bias=proj_bias)
        self.au_proj3 =  nn.Linear(dim, dim, bias=proj_bias)
        self.xs_proj_drop = nn.Dropout(proj_drop)
        self.au_proj_drop1 = nn.Dropout(proj_drop)
        self.au_proj_drop2 = nn.Dropout(proj_drop)
        self.au_proj_drop3 = nn.Dropout(proj_drop)
    def forward(self, x: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor,y3: torch.Tensor,mask: Optional[torch.Tensor] = None,causal=False,freqs_cis=None,freqs_cis2=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N1, C = x.shape
        xs_qkv = self.qkv_xs(x).reshape(B, N1, 3, -1)
        xs_q, xs_k, xs_v = xs_qkv.permute(2, 0, 1, 3).unbind(0)
        

        B,N2,C= y1.shape
        au_qkv1 = self.qkv_au1(y1).reshape(B, N2, 3, -1)
        au_q1, au_k1, au_v1 = au_qkv1.permute(2, 0, 1, 3).unbind(0)
        
        B,N3,C= y2.shape
        au_qkv2 = self.qkv_au2(y2).reshape(B, N3, 3, -1)
        au_q2, au_k2, au_v2 = au_qkv2.permute(2, 0, 1, 3).unbind(0)

        B,N4,C= y3.shape
        au_qkv3 = self.qkv_au3(y3).reshape(B, N4, 3, -1)
        au_q3, au_k3, au_v3 = au_qkv3.permute(2, 0, 1, 3).unbind(0)


        M=N2//N1        
        xs_q = xs_q.view(B, N1, self.num_heads, self.head_dim)
        xs_k = xs_k.view(B, N1, self.num_heads, self.head_dim)
        xs_v = xs_v.view(B, N1, self.num_heads, self.head_dim).transpose(1, 2)
        xs_q, xs_k = self.xs_q_norm(xs_q), self.xs_k_norm(xs_k)
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(xs_q, xs_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == xs_q.shape and img_kk.shape == xs_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {xs_q.shape}, img_kk: {img_kk.shape}, img_k: {xs_k.shape}"
            xs_q, xs_k = img_qq.transpose(1, 2), img_kk.transpose(1, 2)
        au_q1=au_q1.view(B, N2, self.num_heads, self.head_dim)
        au_k1=au_k1.view(B, N2, self.num_heads, self.head_dim)
        au_v1=au_v1.view(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        au_q1, au_k1 = self.au_q_norm1(au_q1), self.au_k_norm1(au_k1)

        au_q2=au_q2.view(B, N3, self.num_heads, self.head_dim)
        au_k2=au_k2.view(B, N3, self.num_heads, self.head_dim)
        au_v2=au_v2.view(B, N3, self.num_heads, self.head_dim).transpose(1, 2)
        au_q2, au_k2 = self.au_q_norm2(au_q2), self.au_k_norm2(au_k2)

        au_q3=au_q3.view(B, N4, self.num_heads, self.head_dim)
        au_k3=au_k3.view(B, N4, self.num_heads, self.head_dim)
        au_v3=au_v3.view(B, N4, self.num_heads, self.head_dim).transpose(1, 2)
        au_q3, au_k3 = self.au_q_norm3(au_q3), self.au_k_norm3(au_k3)

        if freqs_cis2 is not None:
            au_q11, au_k11 = apply_rotary_emb(au_q1, au_k1, freqs_cis2, head_first=False)
            au_q1, au_k1 = au_q11, au_k11
            assert (
                au_q11.shape == au_q1.shape and au_k11.shape == au_k1.shape
            ), f"au_q11: {au_q11.shape}, img_q: {au_q1.shape}, img_kk: {au_k11.shape}, img_k: {au_k1.shape}"



        q = torch.cat((xs_q, au_q1.transpose(1, 2),au_q2.transpose(1, 2),au_q3.transpose(1, 2)), dim=2)
        k = torch.cat((xs_k, au_k1.transpose(1, 2),au_k2.transpose(1, 2),au_k3.transpose(1, 2)), dim=2)
        v = torch.cat((xs_v, au_v1,au_v2,au_v3), dim=2)

        if mask is not None:
            # mask = mask[:, None, :]  # (B, 1, N)
            mask2 = mask[:, None, :].expand(-1, self.num_heads,-1)
            mask = mask[:, None, None, :].expand(-1, self.num_heads,M, -1) 
            mask = rearrange(mask, "b n m d -> b n (m d)")
            att_mask=torch.cat((mask2,mask),dim=-1)
            att_mask=att_mask[:,:,None,:].expand(-1, -1,N1+N2, -1) 
            mask = att_mask.to(dtype=q.dtype)
        if causal:
            mask2 = torch.ones((N1+3*N2,N1+3*N2), dtype=torch.bool, device=v.device)
            mask2[-N3-N4:, :N1] = 0
            mask2[-N1-N1:-N1,-N1:]=0
            mask2[-N1:,-N1-N1:-N1]=0
            mask = mask2.to(dtype=torch.bool)
        if self.fused_attn:
            # print("yesyes")
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N1+N2+N3+N4, C)
        xs,au1,au2,au3=x[:,:N1],x[:,N1:N1+N2],x[:,N1+N2:N1+N2+N3],x[:,N1+N2+N3:N1+N2+N3+N4]
        xs = self.xs_proj(xs)
        xs = self.xs_proj_drop(xs)
        au1 = self.au_proj1(au1)
        au1 = self.au_proj_drop1(au1)

        au2 = self.au_proj2(au2)
        au2 = self.au_proj_drop2(au2)

        au3 = self.au_proj3(au3)
        au3 = self.au_proj_drop3(au3)
        return xs,au1,au2,au3
class MMdual_attention(nn.Module):
    """
    Self-Attention module used in transformers
    """
    fused_attn: Final[bool]

    def __init__(
        self, dim: int, 
        num_heads: int = 8, 
        proj_bias: bool = True,
        attn_drop: float = 0., 
        proj_drop: float = 0.,
        qkv_bias: bool = False, 
        qk_norm: Optional[str] = "rms_norm", 
        **block_kwargs
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv_xs = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_au = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm is None:
            self.xs_q_norm = nn.Identity()
            self.xs_k_norm = nn.Identity()
            self.au_q_norm = nn.Identity()
            self.au_k_norm = nn.Identity()
        elif qk_norm == "rms_norm":
            self.xs_q_norm = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.xs_k_norm = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.au_q_norm = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
            self.au_k_norm = RMSNorm(self.head_dim, eps=1e-5,elementwise_affine=True)
        elif qk_norm == "layer_norm":
            self.xs_q_norm = nn.LayerNorm(dim, eps=1e-5)
            self.xs_k_norm = nn.LayerNorm(dim, eps=1e-5)
            self.au_q_norm = nn.LayerNorm(dim, eps=1e-5)
            self.au_k_norm = nn.LayerNorm(dim, eps=1e-5)
        else:
            raise ValueError(f"Unsupported qk_norm method: {qk_norm}")

        self.attn_drop = nn.Dropout(attn_drop)
        self.xs_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.au_proj =  nn.Linear(dim, dim, bias=proj_bias)
        self.xs_proj_drop = nn.Dropout(proj_drop)
        self.au_proj_drop = nn.Dropout(proj_drop)
    def forward(self, seq_len,x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None,causal=False,freqs_cis=None,freqs_cis2=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N1, C = x.shape
        xs_qkv = self.qkv_xs(x).reshape(B, N1, 3, -1)
        xs_q, xs_k, xs_v = xs_qkv.permute(2, 0, 1, 3).unbind(0)
        

        B,N2,C= y.shape
        au_qkv = self.qkv_au(y).reshape(B, N2, 3, -1)
        au_q, au_k, au_v = au_qkv.permute(2, 0, 1, 3).unbind(0)    
        xs_q = xs_q.view(B, N1, self.num_heads, self.head_dim)
        xs_k = xs_k.view(B, N1, self.num_heads, self.head_dim)
        xs_v = xs_v.view(B, N1, self.num_heads, self.head_dim).transpose(1, 2)
        xs_q, xs_k = self.xs_q_norm(xs_q), self.xs_k_norm(xs_k)
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(xs_q, xs_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == xs_q.shape and img_kk.shape == xs_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {xs_q.shape}, img_kk: {img_kk.shape}, img_k: {xs_k.shape}"
            xs_q, xs_k = img_qq.transpose(1, 2), img_kk.transpose(1, 2)
        au_q=au_q.view(B, N2, self.num_heads, self.head_dim)
        au_k=au_k.view(B, N2, self.num_heads, self.head_dim)
        au_v=au_v.view(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        au_q, au_k = self.au_q_norm(au_q), self.au_k_norm(au_k)
        if freqs_cis2 is not None:
            img_qq, img_kk = apply_rotary_emb(au_q, au_k, freqs_cis2, head_first=False)
            assert (
                img_qq.shape == au_q.shape and img_kk.shape == au_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {xs_q.shape}, img_kk: {img_kk.shape}, img_k: {xs_k.shape}"
            au_q, au_k = img_qq, img_kk
        q = torch.cat((xs_q, au_q.transpose(1, 2)), dim=2)
        k = torch.cat((xs_k, au_k.transpose(1, 2)), dim=2)
        v = torch.cat((xs_v, au_v), dim=2)

        if mask is not None:
            # mask = mask[:, None, :]  # (B, 1, N)
            mask2 = mask[:, None, :].expand(-1, self.num_heads,-1)
            mask = mask[:, None, None, :].expand(-1, self.num_heads,M, -1) 
            mask = rearrange(mask, "b n m d -> b n (m d)")
            att_mask=torch.cat((mask2,mask),dim=-1)
            att_mask=att_mask[:,:,None,:].expand(-1, -1,N1+N2, -1) 
            mask = att_mask.to(dtype=q.dtype)
        if causal:
            mask2 = torch.ones((N1+3*N1,N1+3*N1), dtype=torch.bool, device=v.device)
            mask2[-N1-N1:, :N1] = 0
            mask2[-N1-N1:-N1,-N1:]=0
            mask2[-N1:,-N1-N1:-N1]=0

            mask = mask2.to(dtype=torch.bool)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            



            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N1+N2, C)
        xs,au=x[:,:N1],x[:,N1:]
        xs = self.xs_proj(xs)
        xs = self.xs_proj_drop(xs)
        au = self.au_proj(au)
        au = self.au_proj_drop(au)
        return xs,au

class SelfAttention(nn.Module):
    """
    Self-Attention module used in transformers
    """
    fused_attn: Final[bool]

    def __init__(
        self, dim: int, 
        num_heads: int = 8, 
        proj_bias: bool = True,
        attn_drop: float = 0., 
        proj_drop: float = 0.,
        qkv_bias: bool = False, 
        qk_norm: Optional[str] = "rms_norm", 
        **block_kwargs
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm is None:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim, eps=1e-5)
            self.k_norm = RMSNorm(dim, eps=1e-5)
        elif qk_norm == "layer_norm":
            self.q_norm = nn.LayerNorm(dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(dim, eps=1e-5)
        else:
            raise ValueError(f"Unsupported qk_norm method: {qk_norm}")

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,freqs_cis=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, -1).permute(2, 0, 1, 3) 
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(q, k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == q.shape and img_kk.shape == k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {q.shape}, img_kk: {img_kk.shape}, img_k: {k.shape}"
            q, k = img_qq, img_kk
        
        if mask is not None:
            mask = mask[:, None, None, :].expand(-1, self.num_heads,N, -1)  # (B, num_heads, N, N)
            mask = mask.to(dtype=q.dtype)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    Cross-Attention module used in transformers
    """
    fused_attn: Final[bool]

    def __init__(
        self, dim: int,
        num_heads: int = 8, 
        proj_bias: bool = True,
        attn_drop: float = 0., 
        proj_drop: float = 0.,
        qkv_bias: bool = False, 
        qk_norm: Optional[str] = "rms_norm", 
        **block_kwargs
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.window_size = int(block_kwargs.get('window_size', 1))
        if self.window_size > 1:
            self.indices = (
                torch.arange(self.window_size) - (self.window_size - 1) // 2
            ).unsqueeze(0)            # 1, window_size, [-3, -2, -1, 0, 1, 2, 3]
            norm_dim = dim
        else:
            self.indices = None
            norm_dim = self.head_dim

        if qk_norm is None:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(norm_dim, eps=1e-5)
            self.k_norm = RMSNorm(norm_dim, eps=1e-5)
        elif qk_norm == "layer_norm":
            self.q_norm = nn.LayerNorm(norm_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(norm_dim, eps=1e-5)
        else:
            raise ValueError(f"Unsupported qk_norm method: {qk_norm}")

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor,mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        '''
        if self.window_size > 1:
            indices = (torch.arange(N).unsqueeze(1) + self.indices).to(x.device)   # N x window_size
            indices = indices.clamp(0, N - 1)
            attn_mask = torch.zeros(N, y.shape[1], dtype=x.dtype, device=x.device)  # N x N
            attn_mask = torch.scatter(attn_mask, dim=1, index=indices, value=1)     # N x N
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(-1)                        # 1 x N x N x 1
            attn_mask = attn_mask.expand(-1, -1, -1, M)                             # 1 x N x N x M
            attn_mask = attn_mask.reshape(1, N, -1)                                 # 1 x N x (NxM)

            #x = rearrange(x, "b n c -> (b n) 1 c")
            y = rearrange(y, "b n m d -> b (n m) d")

            q = self.to_q(x)
            q = self.q_norm(q).reshape(-1, N, self.num_heads, self.head_dim).transpose(1, 2)

            kv = self.to_kv(y).reshape(-1, N*M, 2, self.num_heads*self.head_dim).permute(2, 0, 1, 3)
            k, v = kv.unbind(0)
            k = self.k_norm(k)
            k = k.view(-1, N*M, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(-1, N*M, self.num_heads, self.head_dim).transpose(1, 2)
        else:
        '''
        '''
        # wsize = 1
        attn_mask = None

        x = rearrange(x, "b n c -> (b n) 1 c")
        y = rearrange(y, "b n m d -> (b n) m d")

        q = self.to_q(x).reshape(-1, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.to_kv(y).reshape(-1, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        '''

        # wsize=all
        # attn_mask = None
        if y.shape==4:
            M = y.shape[2]
            y = rearrange(y, "b n m d -> b (n m) d")
        else:
            N2 = y.shape[1]
            M=N2//N
        q = self.to_q(x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.to_kv(y).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if mask is not None:
            mask = mask[:, None, None, :].expand(-1, self.num_heads, M, -1)  # (B, num_heads, N, N)
            mask = rearrange(mask, "b n m d -> b n (m d)")
            mask=mask[:, :, None, :].expand(-1, -1, N, -1)
            mask = mask.to(dtype=q.dtype)
            # mask = mask.masked_fill(mask == 0, float("-inf"))
            
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)         # B x N x (N*M)
            attn = attn.masked_fill(mask == 0, float(-1e-9))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        # B, H, N, C//H
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
