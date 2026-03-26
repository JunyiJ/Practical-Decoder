from types import SimpleNamespace
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.gpt import GPT
from src.moe.moe_block import MoEBlock


def _build_nested_moe_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        dim=8,
        n_layers=1,
        n_heads=2,
        vocab_size=16,
        block_size=8,
        dropout=0.0,
        rope=0,
        norm_type="layernorm",
        attn_type="mha",
        attn_dropout=0.0,
        attn_bias=True,
        causal=True,
        mlp_type="moe",
        moe=SimpleNamespace(
            num_experts=3,
            shared_num_experts=1,
            top_k=1,
            aux_loss_weight=0.25,
            ffn="gelu",
        ),
    )


def test_moe_block_reads_nested_moe_config() -> None:
    moe = MoEBlock(_build_nested_moe_cfg())
    assert moe.num_experts == 3
    assert moe.shared_num_experts == 1
    assert moe.top_k == 1
    assert moe.ffn == "gelu"
    assert len(moe.experts) == 4


def test_gpt_excludes_aux_loss_for_validation() -> None:
    torch.manual_seed(0)
    cfg = _build_nested_moe_cfg()
    model = GPT(cfg)

    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    targets = torch.randint(0, cfg.vocab_size, (2, 4))

    _, train_loss, train_breakdown = model(idx, targets, return_loss_breakdown=True)
    _, val_loss, val_breakdown = model(
        idx,
        targets,
        include_aux_loss=False,
        return_loss_breakdown=True,
    )

    assert train_breakdown is not None
    assert val_breakdown is not None
    assert train_breakdown["aux_loss"].item() > 0.0
    torch.testing.assert_close(
        train_loss,
        train_breakdown["ce_loss"] + train_breakdown["aux_loss"],
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(val_breakdown["aux_loss"], torch.zeros_like(val_breakdown["aux_loss"]))
    torch.testing.assert_close(val_loss, val_breakdown["ce_loss"], rtol=0.0, atol=1e-7)
    torch.testing.assert_close(val_breakdown["total_loss"], val_breakdown["ce_loss"], rtol=0.0, atol=1e-7)
