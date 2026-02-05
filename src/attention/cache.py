import torch
import torch.nn as nn

class KVCache(nn.Module):
    def __init__(self, cfg, device, batch_size=None, dtype=None, n_kv_heads=None):
        super().__init__()
        self.attn_type = getattr(cfg.model, "attn_type", None)
        self.is_mla = self.attn_type == "mla"
        if n_kv_heads is None:
            n_kv_heads = getattr(cfg.model, "n_kv_heads", cfg.model.n_heads)
            if getattr(cfg.model, "attn_type", None) == "mha":
                n_kv_heads = cfg.model.n_heads
        if batch_size is None:
            batch_size = getattr(cfg.training, "batch_size", 1)
        if self.is_mla:
            dimc_kv = getattr(cfg.model, "dimckv", cfg.model.dim // 10)
            head_dim_r = getattr(cfg.model, "head_dim_r", cfg.model.dim // cfg.model.n_heads)
            c_cache_shape = (batch_size, cfg.model.block_size, dimc_kv)
            r_cache_shape = (batch_size, cfg.model.block_size, head_dim_r)
            self.register_buffer("c_cache", torch.zeros(c_cache_shape, device=device, dtype=dtype))
            self.register_buffer("r_cache", torch.zeros(r_cache_shape, device=device, dtype=dtype))
            self.k_cache = None
            self.v_cache = None
            return
        # Cache shape: (Batch, Num_Heads, Max_Seq_Len, Head_Dim)
        head_dim = cfg.model.dim // cfg.model.n_heads
        cache_shape = (
            batch_size,
            n_kv_heads,
            cfg.model.block_size,
            head_dim,
        )
        self.register_buffer("k_cache", torch.zeros(cache_shape, device=device, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, device=device, dtype=dtype))

    def update(self, k_new, v_new, start_pos):
        """
        k_new, v_new: (Batch, Num_Heads, Seq_Len_New, Head_Dim)
        """
        if self.is_mla:
            c_new = k_new
            r_new = v_new
            end_pos = start_pos + c_new.size(1)
            if end_pos > self.c_cache.size(1):
                raise ValueError("Cache size exceeded; increase block_size or reset cache.")
            self.c_cache[:, start_pos:end_pos, :] = c_new
            self.r_cache[:, start_pos:end_pos, :] = r_new
            return self.c_cache[:, :end_pos, :], self.r_cache[:, :end_pos, :]
        end_pos = start_pos + k_new.size(2)
        if end_pos > self.k_cache.size(2):
            raise ValueError("Cache size exceeded; increase block_size or reset cache.")
        self.k_cache[:, :, start_pos : end_pos, :] = k_new
        self.v_cache[:, :, start_pos: end_pos, :] = v_new
        return self.k_cache[:, :, :end_pos, :], self.v_cache[:, :, :end_pos, :]


    def reset(self):
        if self.is_mla:
            self.c_cache.zero_()
            self.r_cache.zero_()
            return
        self.k_cache.zero_()
        self.v_cache.zero_()
