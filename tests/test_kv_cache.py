from types import SimpleNamespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `from src...` imports when running pytest from repo root.
    sys.path.append(str(PROJECT_ROOT))

import torch
import pytest

from src.attention.cache import KVCache
from src.models.gpt import GPT


def _build_cfg(attn_type: str, n_heads: int, n_kv_heads: int) -> SimpleNamespace:
    model = SimpleNamespace(
        dim=16,
        n_layers=2,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=32,
        block_size=8,
        dropout=0.0,
        attn_dropout=0.0,
        attn_bias=True,
        causal=True,
        attn_type=attn_type,
        mlp_type="dense",
        moe=SimpleNamespace(num_experts=2, top_k=1, aux_loss_weight=0.0),
    )
    training = SimpleNamespace(batch_size=2)
    return SimpleNamespace(model=model, training=training)


@pytest.mark.parametrize(
    "attn_type,n_heads,n_kv_heads",
    [
        ("mha", 4, 4),
        ("gqa", 4, 2),
    ],
)
def test_kv_cache_matches_full_forward(attn_type: str, n_heads: int, n_kv_heads: int) -> None:
    torch.manual_seed(0)
    cfg = _build_cfg(attn_type, n_heads, n_kv_heads)
    model = GPT(cfg.model)
    model.eval()

    batch = 2
    seq_len = 6
    idx = torch.randint(0, cfg.model.vocab_size, (batch, seq_len))
    full_logits, _ = model(idx)

    caches = [
        KVCache(cfg, device="cpu", batch_size=batch, dtype=next(model.parameters()).dtype)
        for _ in range(cfg.model.n_layers)
    ]
    step_logits = []
    for t in range(seq_len):
        logits, _ = model(idx[:, t:t + 1], cache=caches, start_pos=t)
        step_logits.append(logits[:, -1, :])
    cached_logits = torch.stack(step_logits, dim=1)

    torch.testing.assert_close(full_logits, cached_logits, rtol=1e-5, atol=1e-6)
