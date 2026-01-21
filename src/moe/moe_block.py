import torch
import torch.nn as nn


class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.hidden_dim = getattr(cfg, "moe_hidden_dim", cfg.dim * 2)
        self.num_experts = getattr(cfg, "moe_num_experts", 4)
        self.top_k = getattr(cfg, "moe_top_k", 2)
        if self.top_k < 1:
            raise ValueError("cfg.moe_top_k must be >= 1")
        pass

    def forward(self, x):
        pass
