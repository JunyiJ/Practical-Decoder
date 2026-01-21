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

from src.attention.gqa import GQA
from src.attention.mha import MHA


def _build_cfg(attn_bias: bool, n_heads: int, n_kv_heads: int, causal: bool) -> SimpleNamespace:
    return SimpleNamespace(
        dim=8,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        block_size=16,
        dropout=0.0,
        attn_dropout=0.0,
        attn_bias=attn_bias,
        causal=causal,
    )


def _copy_mha_weights_to_gqa(gqa: GQA, mha: MHA) -> None:
    with torch.no_grad():
        gqa.qkv.weight.copy_(mha.qkv.weight)
        if gqa.qkv.bias is not None:
            gqa.qkv.bias.copy_(mha.qkv.bias)
        gqa.out.weight.copy_(mha.out.weight)
        if gqa.out.bias is not None:
            gqa.out.bias.copy_(mha.out.bias)


def _gqa_reference(gqa: GQA, x: torch.Tensor, causal: bool) -> torch.Tensor:
    b, s, d = x.size()
    qkv = gqa.qkv(x)
    kv_dim = gqa.n_kv_heads * gqa.head_dim
    q, k, v = torch.split(qkv, [gqa.dim, kv_dim, kv_dim], dim=-1)
    q = q.view(b, s, gqa.n_heads, gqa.head_dim).transpose(1, 2)
    k = k.view(b, s, gqa.n_kv_heads, gqa.head_dim).transpose(1, 2)
    v = v.view(b, s, gqa.n_kv_heads, gqa.head_dim).transpose(1, 2)
    if gqa.group_factor > 1:
        k = k.repeat_interleave(gqa.group_factor, dim=1)
        v = v.repeat_interleave(gqa.group_factor, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
    y = y.transpose(1, 2).contiguous().view(b, s, d)
    return gqa.out(y)


@pytest.mark.parametrize("causal", [True, False])
def test_gqa_matches_mha_when_full_kv(causal: bool) -> None:
    torch.manual_seed(0)
    cfg = _build_cfg(attn_bias=True, n_heads=4, n_kv_heads=4, causal=causal)
    gqa = GQA(cfg)
    mha = MHA(cfg)
    _copy_mha_weights_to_gqa(gqa, mha)
    gqa.eval()
    mha.eval()

    x = torch.randn(2, 7, cfg.dim)
    y_gqa, _ = gqa(x)
    y_mha, _ = mha(x)
    torch.testing.assert_close(y_gqa, y_mha, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("causal", [True, False])
def test_gqa_matches_reference_grouped(causal: bool) -> None:
    torch.manual_seed(0)
    cfg = _build_cfg(attn_bias=False, n_heads=4, n_kv_heads=2, causal=causal)
    gqa = GQA(cfg)
    gqa.eval()

    x = torch.randn(3, 5, cfg.dim)
    y_gqa, _ = gqa(x)
    y_ref = _gqa_reference(gqa, x, causal=causal)
    torch.testing.assert_close(y_gqa, y_ref, rtol=1e-5, atol=1e-6)
