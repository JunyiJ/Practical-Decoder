import math

import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError("cfg.dim must be divisible by cfg.n_heads")
        pass

    def forward(self, x):
        pass
