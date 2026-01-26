import torch
import torch.nn as nn

from .blocks import TransformerBlock

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.aux_loss_weight = getattr(cfg, "aux_loss_weight", 1.0)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.dim),
            "wpe": nn.Embedding(cfg.block_size, cfg.dim),
            "h": nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            "ln_f": nn.LayerNorm(cfg.dim)
        })
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.enable_rope = (getattr(cfg, "rope", 0) == 1)

    def forward(self, idx, targets=None, cache=None, start_pos=0):
        b, t = idx.size()
        if start_pos + t > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {t} with start_pos {start_pos} exceeds block_size {self.cfg.block_size}"
            )
        token_emb = self.transformer.wte(idx)
        if not self.enable_rope:
            pos = torch.arange(start_pos, start_pos + t, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = token_emb + pos_emb
        else:
            x = token_emb
        aux_losses = []
        if cache is not None and not isinstance(cache, (list, tuple)):
            raise TypeError("cache must be a list/tuple of per-layer KVCache objects")
        if cache is not None and len(cache) != len(self.transformer.h):
            raise ValueError("cache length must match number of transformer blocks")
        for layer_idx, block in enumerate(self.transformer.h):
            layer_cache = None if cache is None else cache[layer_idx]
            x, block_aux_loss = block(x, layer_cache, start_pos)
            if block_aux_loss is not None:
                aux_losses.append(block_aux_loss)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if aux_losses:
                loss += self.aux_loss_weight * sum(aux_losses)
        
        return logits, loss
