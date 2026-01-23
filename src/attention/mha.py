import math

import torch
import torch.nn as nn

from ..models.rope import apply_rotary_emb


class ScaledAttention(nn.Module):
    def __init__(self, dim, attn_dim, attn_bias, attn_drop, causal=True):
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        self.causal = causal

        self.qkv = nn.Linear(self.dim, self.attn_dim * 3, bias=attn_bias)
        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, x, cache=None, start_pos=0):
        b, s, d = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        score = q @ k.transpose(-2, -1)
        if self.causal:
            mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))
            score = score.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(score / math.sqrt(self.attn_dim), dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v
        return y, None

class MHA_NAIVE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"cfg.dim must be divisible by cfg.n_heads (got {self.dim}, {self.n_heads})")
        self.head_dim = self.dim // self.n_heads

        attn_bias = getattr(cfg, "attn_bias", True)
        drop = getattr(cfg, "dropout", 0.0)
        attn_drop = getattr(cfg, "attn_dropout", drop)
        causal = getattr(cfg, "causal", True)
        self.heads = nn.ModuleList([ScaledAttention(self.dim, self.head_dim, attn_bias, attn_drop, causal) for _ in range(self.n_heads)])
        self.out = nn.Linear(self.dim, self.dim, bias=attn_bias)
        self.out_dropout = nn.Dropout(drop)

    def forward(self, x):
        heads_output = []
        for head in self.heads:
            output, _ = head(x)
            heads_output.append(output)
        y = torch.concat(heads_output, dim=-1)
        y = self.out_dropout(self.out(y))
        return y, None

class MHA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"cfg.dim must be divisible by cfg.n_heads (got {self.dim}, {self.n_heads})")
        self.head_dim = self.dim // self.n_heads

        attn_bias = getattr(cfg, "attn_bias", True)
        drop = getattr(cfg, "dropout", 0.0)
        attn_drop = getattr(cfg, "attn_dropout", drop)
        self.causal = getattr(cfg, "causal", True)

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=attn_bias)
        self.out = nn.Linear(self.dim, self.dim, bias=attn_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_dropout = nn.Dropout(drop)

        self.max_seq_len = getattr(cfg, "block_size", None)
        if self.causal and self.max_seq_len is not None:
            mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool))
            self.register_buffer("causal_mask", mask, persistent=False)
        else:
            self.register_buffer("causal_mask", None, persistent=False)
    
    def _get_causal_mask(self, s, device):
        if self.causal_mask is None or self.causal_mask.size(0) < s:
            self.causal_mask = torch.tril(torch.ones(s, s, device=device, dtype=torch.bool))
        return self.causal_mask[:s, :s]

    def forward(self, x, cache=None, start_pos=0, freq_cos=None, freq_sin=None):
        b, s, d = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        if freq_cos is not None and freq_sin is not None:
            cos = freq_cos[start_pos:start_pos + s].to(device=q.device, dtype=q.dtype)
            sin = freq_sin[start_pos:start_pos + s].to(device=q.device, dtype=q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v, start_pos)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            total_len = k.size(-2)
            mask = self._get_causal_mask(total_len, x.device)
            mask = mask[start_pos:start_pos + s, :total_len]
            att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        y = self.out_dropout(self.out(y))
        return y, None
