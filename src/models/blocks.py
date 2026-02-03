import torch.nn as nn

from ..attention.mha import MHA
from ..attention.gqa import GQA
from ..moe.moe_block import MoEBlock

from .rope import precompute_freqs_cis

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"cfg.dim must be divisible by cfg.n_heads (got {self.dim}, {self.n_heads})")
        self.head_dim = self.dim // self.n_heads
        self.norm_type = getattr(cfg, "norm_type", "layernorm")
        if self.norm_type == "layernorm":
            self.ln1 = nn.LayerNorm(cfg.dim)
            self.ln2 = nn.LayerNorm(cfg.dim)
        elif self.norm_type == "rmsnorm":
            self.ln1 = nn.RMSNorm(cfg.dim)
            self.ln2 = nn.RMSNorm(cfg.dim)
        else:
            raise ValueError(f"Unknown norm_type: {cfg.norm_type}")

        if cfg.attn_type == "mha":
            self.attn = MHA(cfg)
        elif cfg.attn_type == "gqa":
            self.attn = GQA(cfg)
        else:
            raise ValueError(f"Unknown attn_type: {cfg.attn_type}")
        # TODO add more variants later

        if cfg.mlp_type == "dense":
            self.mlp = nn.Sequential(
                nn.Linear(cfg.dim, cfg.dim * 4),
                nn.GELU(),
                nn.Linear(cfg.dim * 4, cfg.dim),
                nn.Dropout(cfg.dropout)
            )
        elif cfg.mlp_type == "moe":
            self.mlp = MoEBlock(cfg)
        else:
            raise ValueError(f"Unknown mlp_type: {cfg.mlp_type}")
        self.enable_rope = (getattr(cfg, "rope", 0) == 1)
        if self.enable_rope:
            if self.head_dim % 2 != 0:
                raise ValueError("head_dim must be even when rope is enabled")
            freq_cos, freq_sin = precompute_freqs_cis(self.head_dim, cfg.block_size)
            self.register_buffer("freq_cos", freq_cos)
            self.register_buffer("freq_sin", freq_sin)
        else:
            self.freq_cos = None
            self.freq_sin = None

    def forward(self, x, cache=None, start_pos=0):
        residual = x
        x, _ = self.attn(self.ln1(x), cache=cache, start_pos=start_pos, freq_cos=self.freq_cos, freq_sin=self.freq_sin)
        x = residual + x
        mlp_out = self.mlp(self.ln2(x))

        # MoE aux loss if present
        aux_loss = None
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
        x = x + mlp_out
        return x, aux_loss
