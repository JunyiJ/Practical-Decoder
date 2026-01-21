import torch
import torch.nn as nn

from .blocks import TransformerBlock

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.dim),
            "h": nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            "ln_f": nn.LayerNorm(cfg.dim)
        })
        self.lm_head = nn.Linear(cfg.dim, cfg,vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.wte(idx)
        aux_losses = []
        for block in self.transformer.h:
            x, block_aux_loss = block(x)
            if block_aux_loss is not None:
                aux_losses.append(block_aux_loss)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets)
            if aux_losses:
                loss += sum(aux_losses)
        
        return logits, loss
