from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
from datetime import datetime
from ..models.gpt import GPT
from ..data.loaders import TinyDataLoader
from ..utils.checkpoint import save_checkpoint

@hydra.main(version_base=None, config_path="../config", config_name="mac_tinyshakespeare")
def train(cfg: DictConfig):
    device = cfg.training.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    loader = TinyDataLoader(
        data_path=to_absolute_path(cfg.data.path),
        batch_size = cfg.training.batch_size,
        block_size = cfg.model.block_size,
        device = device
    )
    cfg.model.vocab_size = loader.vocab_size
    model = GPT(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    checkpoint_dir = to_absolute_path(cfg.training.checkpoint_dir)
    checkpoint_every = cfg.training.checkpoint_every
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"Starting training: {cfg.model.attn_type} + {cfg.model.mlp_type} on {device}")
    model.train()
    for iter in range(cfg.training.max_iters):
        xb, yb = loader.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")
        if checkpoint_dir and checkpoint_every > 0 and (iter + 1) % checkpoint_every == 0:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_checkpoint(
                Path(checkpoint_dir) / f"step_{iter + 1:06d}_{stamp}.pt",
                model,
                optimizer=optimizer,
                step=iter + 1,
                cfg=cfg,
            )
    if checkpoint_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_checkpoint(
            Path(checkpoint_dir) / f"final_{stamp}.pt",
            model,
            optimizer=optimizer,
            step=cfg.training.max_iters,
            cfg=cfg,
        )


if __name__ == "__main__":
    train()
