import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponential (cos + i*sin).
    dim: head_dim
    end: max_seq_len (block_size)
    """
    # base.exp(-2i/d)
    i = torch.arange(0, dim, 2)[: (dim//2)].float()
    freqs = 1.0 / (theta ** (2 * i / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (max_seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, n_heads, seq_len, head_dim)
    freqs_cis: (seq_len, head_dim // 2)
    """
    B, n_heads, seq_len, head_dim = x.shape
    cos = freqs_cos.view(1, 1, seq_len, head_dim//2)
    sin = freqs_sin.view(1, 1, seq_len, head_dim//2)
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")
    x = x.view(B, n_heads, seq_len, head_dim // 2, 2)
    x1s = x[..., 0]
    x2s = x[..., 1]
    x1s_new = x1s * cos - x2s * sin
    x2s_new = x1s * sin + x2s * cos
    x_new = torch.stack([x1s_new, x2s_new], dim=-1).flatten(3)
    return x_new