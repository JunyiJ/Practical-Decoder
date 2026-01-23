import math

import torch
import torch.nn as nn

from ..models.rope import apply_rotary_emb


class GQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_heads = getattr(cfg, "n_heads", 1)
        if self.dim % self.n_heads != 0:
            raise ValueError("cfg.dim must be divisible by cfg.n_heads")
        self.head_dim = self.dim // self.n_heads
        self.n_kv_heads = getattr(cfg, "n_kv_heads", getattr(cfg, "n_groups", self.n_heads))
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("cfg.n_heads must be divisible by cfg.n_kv_heads")
        self.group_factor = self.n_heads // self.n_kv_heads


        attn_bias = getattr(cfg, "attn_bias", True)
        drop = getattr(cfg, "dropout", 0.0)
        attn_drop = getattr(cfg, "attn_dropout", drop)
        self.causal = getattr(cfg, "causal", True)

        kv_dim = self.n_kv_heads * self.head_dim
        self.qkv = nn.Linear(self.dim, self.dim + 2 * kv_dim, bias=attn_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out = nn.Linear(self.dim, self.dim, bias=attn_bias)
        self.dropout = nn.Dropout(drop)

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
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [self.dim, kv_dim, kv_dim], dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if freq_cos is not None and freq_sin is not None:
            cos = freq_cos[start_pos:start_pos + s].to(device=q.device, dtype=q.dtype)
            sin = freq_sin[start_pos:start_pos + s].to(device=q.device, dtype=q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v, start_pos)
        q = q.view(b, self.n_kv_heads, self.group_factor, s, self.head_dim)
        k_view = k.unsqueeze(2) # (b, n_kv_heads, 1, k_len, head_dim)
        v_view = v.unsqueeze(2)
        # if self.group_factor > 1:
        #     k_len = k.size(-2)
        #     # Expand KV heads across groups without materializing repeats.
        #     k = k.unsqueeze(2).expand(b, self.n_kv_heads, self.group_factor, k_len, self.head_dim)
        #     v = v.unsqueeze(2).expand(b, self.n_kv_heads, self.group_factor, k_len, self.head_dim)
            # reshape on an expanded tensor can be performance thief.
            # k = k.reshape(b, self.n_heads, k_len, self.head_dim)
            # v = v.reshape(b, self.n_heads, k_len, self.head_dim)
        attn_score = torch.matmul(q, k_view.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            total_len = k.size(-2)
            causal_mask = self._get_causal_mask(total_len, x.device)
            causal_mask = causal_mask[start_pos:start_pos + s, :total_len]
            attn_score = attn_score.masked_fill(~causal_mask, float("-inf"))
        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.attn_dropout(attn_score)
        y = torch.matmul(attn_score, v_view)
        y = y.view(b, self.n_heads, s, self.head_dim)
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        y = self.dropout(self.out(y))
        return y, None
