import torch.nn as nn

from ..attention.mha import MHA
from ..attention.gqa import GQA
from ..moe.moe_block import MoEBlock

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO consider RMSNorm
        self.ln1 = nn.LayerNorm(cfg.dim)

        if cfg.attn_type == "mha":
            self.attn = MHA(cfg)
        elif cfg.attn_type == "gqa":
            self.attn = GQA(cfg)
        else:
            raise ValueError(f"Unknown attn_type: {cfg.attn_type}")
        # TODO add more variants later

        self.ln2 = nn.LayerNorm(cfg.dim)

        if cfg.mlp_type == "dense":
            self.mlp = nn.Sequential(
                nn.Linear(cfg.dim, cfg.dim * 2),
                nn.GELU(),
                nn.Linear(cfg.dim * 2, cfg.dim),
                nn.Dropout(cfg.dropout)
            )
        elif cfg.mlp_type == "moe":
            self.mlp = MoEBlock(cfg)
        else:
            raise ValueError(f"Unknown mlp_type: {cfg.mlp_type}")

    def forward(self, x):
        residual = x
        x, _ = self.attn(self.ln1(x))
        x = residual + x
        mlp_out = self.mlp(self.ln2(x))

        # MoE aux loss if present
        aux_loss = None
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
        x = x + mlp_out
        return x, aux_loss
