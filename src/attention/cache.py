import torch
import torch.nn as nn

class KVCache(nn.Module):
    def __init__(self, cfg, device, batch_size=None, dtype=None, n_kv_heads=None):
        super().__init__()
        # Cache shape: (Batch, Num_Heads, Max_Seq_Len, Head_Dim)
        head_dim = cfg.model.dim // cfg.model.n_heads
        if n_kv_heads is None:
            n_kv_heads = getattr(cfg.model, "n_kv_heads", cfg.model.n_heads)
            if getattr(cfg.model, "attn_type", None) == "mha":
                n_kv_heads = cfg.model.n_heads
        if batch_size is None:
            batch_size = getattr(cfg.training, "batch_size", 1)
        cache_shape = (
            batch_size,
            n_kv_heads,
            cfg.model.block_size,
            head_dim,
        )
        self.register_buffer("k_cache", torch.zeros(cache_shape, device=device, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, device=device, dtype=dtype))
        self.current_pos = 0

    def update(self, k_new, v_new, start_pos):
        """
        k_new, v_new: (Batch, Num_Heads, Seq_Len_New, Head_Dim)
        """
        end_pos = start_pos + k_new.size(2)
        if end_pos > self.k_cache.size(2):
            raise ValueError("Cache size exceeded; increase block_size or reset cache.")
        self.k_cache[:, :, start_pos : end_pos, :] = k_new
        self.v_cache[:, :, start_pos: end_pos, :] = v_new
        self.current_pos = end_pos
        return self.k_cache[:, :, :end_pos, :], self.v_cache[:, :, :end_pos, :]


    def reset(self):
        self.current_pos = 0
