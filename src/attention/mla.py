import math

import torch
import torch.nn as nn

from ..models.rope import apply_rotary_emb

# https://arxiv.org/abs/2405.04434
class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.dimc_kv = getattr(cfg, "dimckv", self.dim // 10)
        self.dimc_q = getattr(cfg, "dimcq", self.dim // 10)
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError(f"cfg.dim must be divisible by cfg.n_heads (got {self.dim}, {self.n_heads})")
        self.head_dim = self.dim // self.n_heads
        self.head_dim_r = getattr(cfg, "head_dim_r", self.head_dim)

        attn_bias = getattr(cfg, "attn_bias", True)
        drop = getattr(cfg, "dropout", 0.0)
        attn_drop = getattr(cfg, "attn_dropout", drop)
        self.causal = getattr(cfg, "causal", True)

        self.dkv = nn.Linear(self.dim, self.dimc_kv, bias=attn_bias)
        self.uk = nn.Linear(self.dimc_kv, self.head_dim * self.n_heads, bias=attn_bias)
        self.uv = nn.Linear(self.dimc_kv, self.head_dim * self.n_heads, bias=attn_bias)
        self.dq = nn.Linear(self.dim, self.dimc_q, bias=attn_bias)
        self.uq = nn.Linear(self.dimc_q, self.head_dim * self.n_heads, bias=attn_bias)
        self.qr = nn.Linear(self.dimc_q, self.head_dim_r * self.n_heads, bias=attn_bias)
        self.kr = nn.Linear(self.dim, self.head_dim_r, bias=attn_bias)

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
        c_q = self.dq(x)  # [b, s, dimc_q]
        c_kv = self.dkv(x)  # [b, s, dimc_kv]
        # [b, s, head_dim_r * n_heads] => [b, n_heads, s, head_dim_r]
        c_q_r = self.qr(c_q).view(b, s, self.n_heads, -1).transpose(1, 2)
        # [b, s, head_dim_r] => [b, 1, s, head_dim_r]
        k_r = self.kr(x).unsqueeze(1)
        if freq_cos is not None and freq_sin is not None:
            cos = freq_cos[start_pos:start_pos + s].to(device=c_q.device, dtype=c_q.dtype)
            sin = freq_sin[start_pos:start_pos + s].to(device=c_q.device, dtype=c_q.dtype)
            r_q = apply_rotary_emb(c_q_r, cos, sin)
            r_k = apply_rotary_emb(k_r, cos, sin)
        else:
            r_q = c_q_r
            r_k = k_r
        # [b, n_heads, s, head_dim_r] => [b, s, head_dim_r * n_heads]
        r_q = r_q.transpose(1, 2).contiguous().view(b, s, -1)
        # [b, 1, s, head_dim_r] => [b, s, head_dim_r]
        r_k = r_k.squeeze(1)
        if cache is not None:
            c_kv, r_k = cache.update(c_kv, r_k, start_pos)
        # concat([b, s, head_dim x n_heads], [b, s, head_dim_r x n_heads]) => [b, s, (head_dim + head_dim_r) * n_heads]
        q = torch.concat([self.uq(c_q), r_q], dim=-1)
        total_len = c_kv.size(1)
        # [b, total_len, head_dim x n_heads]
        k_base = self.uk(c_kv)
        v = self.uv(c_kv)
        # [b, total_len, head_dim_r x n_heads]
        r_k = r_k.unsqueeze(2).expand(-1, -1, self.n_heads, -1).contiguous().view(b, total_len, -1)
        # concat([b, total_len, head_dim x n_heads], [b, total_len, head_dim_r x n_heads]) => [b, total_len, (head_dim + head_dim_r) * n_heads]
        k = torch.concat([k_base, r_k], dim=-1)

        q = q.view(b, s, self.n_heads, -1).transpose(1, 2)
        k = k.view(b, total_len, self.n_heads, -1).transpose(1, 2)
        v = v.view(b, total_len, self.n_heads, -1).transpose(1, 2)

        # [b, n_heads, s, s]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim + self.head_dim_r)
        if self.causal:
            total_len = k.size(-2)
            mask = self._get_causal_mask(total_len, x.device)
            mask = mask[start_pos:start_pos + s, :total_len]
            att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # [b, n_heads, s, head_dim]
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        y = self.out_dropout(self.out(y))
        return y, None
