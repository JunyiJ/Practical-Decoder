from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow `from src...` imports when running pytest from repo root.
    sys.path.append(str(PROJECT_ROOT))

import torch

from src.models.rope import apply_rotary_emb, precompute_freqs_cis


def test_rope_identity_at_pos0() -> None:
    torch.manual_seed(0)
    b, h, s, d = 2, 3, 4, 8
    x = torch.randn(b, h, s, d)
    cos, sin = precompute_freqs_cis(d, s)
    y = apply_rotary_emb(x, cos, sin)
    torch.testing.assert_close(y[:, :, 0, :], x[:, :, 0, :], rtol=1e-5, atol=1e-6)


def test_rope_norm_preserved() -> None:
    torch.manual_seed(0)
    b, h, s, d = 2, 2, 5, 6
    x = torch.randn(b, h, s, d)
    cos, sin = precompute_freqs_cis(d, s)
    y = apply_rotary_emb(x, cos, sin)
    x_norm = torch.linalg.vector_norm(x, dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1)
    torch.testing.assert_close(x_norm, y_norm, rtol=1e-5, atol=1e-6)
