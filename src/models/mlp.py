import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.linear_gate = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.linear_up = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.linear_down = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.linear_down(F.silu(self.linear_gate(x)) * self.linear_up(x)))
        return x