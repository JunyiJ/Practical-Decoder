from types import SimpleNamespace
from pathlib import Path
import sys

import torch
import pytest
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `from src...` imports when running pytest from repo root.
    sys.path.append(str(PROJECT_ROOT))

from src.attention.mha import MHA, MHA_NAIVE


def _build_cfg(attn_bias: bool) -> SimpleNamespace:
    return SimpleNamespace(
        dim=8,
        n_heads=2,
        dropout=0.0,
        attn_dropout=0.0,
        attn_bias=attn_bias,
        causal=True,
    )


def _copy_naive_weights_to_mha(mha: MHA, naive: MHA_NAIVE) -> None:
    head_dim = naive.head_dim
    dim = naive.dim
    dtype = naive.heads[0].qkv.weight.dtype
    # Stitch per-head q/k/v matrices into the fused [3*dim, dim] layout.
    qkv_weight = torch.zeros(3 * dim, dim, dtype=dtype)
    qkv_bias = None
    if naive.heads[0].qkv.bias is not None:
        qkv_bias = torch.zeros(3 * dim, dtype=naive.heads[0].qkv.bias.dtype)

    for i, head in enumerate(naive.heads):
        w_q, w_k, w_v = head.qkv.weight.chunk(3, dim=0)
        start = i * head_dim
        end = (i + 1) * head_dim
        # Map each head's slice into q, k, v blocks of the fused projection.
        qkv_weight[start:end] = w_q
        qkv_weight[dim + start:dim + end] = w_k
        qkv_weight[2 * dim + start:2 * dim + end] = w_v

        if qkv_bias is not None:
            b_q, b_k, b_v = head.qkv.bias.chunk(3, dim=0)
            qkv_bias[start:end] = b_q
            qkv_bias[dim + start:dim + end] = b_k
            qkv_bias[2 * dim + start:2 * dim + end] = b_v

    with torch.no_grad():
        mha.qkv.weight.copy_(qkv_weight)
        if qkv_bias is not None:
            mha.qkv.bias.copy_(qkv_bias)
        mha.out.weight.copy_(naive.out.weight)
        if naive.out.bias is not None:
            mha.out.bias.copy_(naive.out.bias)


def _mha_reference(mha: MHA, x: torch.Tensor, causal: bool) -> torch.Tensor:
    b, s, d = x.size()
    qkv = mha.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(b, s, mha.n_heads, mha.head_dim).transpose(1, 2)
    k = k.view(b, s, mha.n_heads, mha.head_dim).transpose(1, 2)
    v = v.view(b, s, mha.n_heads, mha.head_dim).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
    y = y.transpose(1, 2).contiguous().view(b, s, d)
    return mha.out(y)


@pytest.mark.parametrize("attn_bias", [True, False])
def test_mha_matches_naive(attn_bias: bool) -> None:
    torch.manual_seed(0)
    cfg = _build_cfg(attn_bias)
    naive = MHA_NAIVE(cfg)
    mha = MHA(cfg)
    _copy_naive_weights_to_mha(mha, naive)
    naive.eval()
    mha.eval()

    x = torch.randn(2, 5, cfg.dim)
    y_naive, _ = naive(x)
    y_mha, _ = mha(x)
    torch.testing.assert_close(y_naive, y_mha, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("causal", [True, False])
def test_mha_matches_sdpa_reference(causal: bool) -> None:
    torch.manual_seed(0)
    cfg = _build_cfg(attn_bias=True)
    cfg.causal = causal
    mha = MHA(cfg)
    mha.eval()

    x = torch.randn(2, 6, cfg.dim)
    y_mha, _ = mha(x)
    y_ref = _mha_reference(mha, x, causal=causal)
    torch.testing.assert_close(y_mha, y_ref, rtol=1e-5, atol=1e-6)


def test_mha_output_shape() -> None:
    cfg = _build_cfg(attn_bias=True)
    cfg.causal = True
    mha = MHA(cfg)
    x = torch.randn(3, 4, cfg.dim)
    y, aux = mha(x)
    assert y.shape == x.shape
    assert aux is None
