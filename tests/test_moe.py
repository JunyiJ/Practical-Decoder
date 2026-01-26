from types import SimpleNamespace
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `from src...` imports when running pytest from repo root.
    sys.path.append(str(PROJECT_ROOT))

from src.moe.moe_block import MoEBlock


def _build_cfg(num_experts: int, top_k: int, dim: int = 4, moe_hidden_dim: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        dim=dim,
        moe_hidden_dim=moe_hidden_dim,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        dropout=0.0,
    )


def _set_constant_expert_outputs(moe: MoEBlock, values: list[float]) -> None:
    assert len(values) == moe.num_experts
    for expert, value in zip(moe.experts, values):
        linear1 = expert[0]
        linear2 = expert[2]
        with torch.no_grad():
            linear1.weight.zero_()
            if linear1.bias is not None:
                linear1.bias.zero_()
            linear2.weight.zero_()
            if linear2.bias is not None:
                linear2.bias.fill_(value)


def _force_router_to_expert0(moe: MoEBlock) -> None:
    with torch.no_grad():
        w = torch.zeros(moe.num_experts, moe.dim)
        w[0].fill_(1.0)
        for i in range(1, moe.num_experts):
            w[i].fill_(-float(i))
        moe.router.weight.copy_(w)


def test_moe_routes_to_single_expert_and_aux_loss() -> None:
    torch.manual_seed(0)
    cfg = _build_cfg(num_experts=3, top_k=1)
    moe = MoEBlock(cfg)
    moe.eval()

    _set_constant_expert_outputs(moe, [0.25, 0.5, -0.75])
    _force_router_to_expert0(moe)

    x = torch.ones(2, 3, cfg.dim)
    y, aux = moe(x)

    expected = torch.full_like(x, 0.25)
    torch.testing.assert_close(y, expected, rtol=0.0, atol=0.0)

    logits = torch.tensor(
        [cfg.dim * 1.0, cfg.dim * -1.0, cfg.dim * -2.0],
        dtype=aux.dtype,
        device=aux.device,
    )
    expected_aux = cfg.moe_num_experts * torch.softmax(logits, dim=0)[0]
    torch.testing.assert_close(aux, expected_aux, rtol=1e-6, atol=1e-7)
