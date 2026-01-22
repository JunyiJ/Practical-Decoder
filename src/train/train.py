from pathlib import Path
import math
import logging
import sys
import time
import resource

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
from datetime import datetime
from ..models.gpt import GPT
from ..data.loaders import TinyDataLoader
from ..utils.checkpoint import save_checkpoint

log = logging.getLogger(__name__)

def _get_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_bytes = rss
    else:
        rss_bytes = rss * 1024
    return rss_bytes / (1024 * 1024)


def _get_accel_mem_mb(device: str) -> float | None:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device == "mps" and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            return current() / (1024 * 1024)
    return None

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
    log.info("Starting training: %s + %s on %s", cfg.model.attn_type, cfg.model.mlp_type, device)
    
    start_time = time.perf_counter()
    for iter in range(cfg.training.max_iters):
        model.train()
        iter_start = time.perf_counter()
        xb, yb = loader.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        iter_time = time.perf_counter() - iter_start

        if iter % 400 == 0:
            elapsed = time.perf_counter() - start_time
            rss_mb = _get_rss_mb()
            accel_mb = _get_accel_mem_mb(device)
            model.eval()
            eval_losses = 0.0
            with torch.no_grad():
                for _ in range(100):
                    x_eval, y_eval = loader.get_batch('val')
                    _, eval_loss = model(x_eval, y_eval)
                    eval_losses += eval_loss.item()
            avg_eval_loss = eval_losses / 100
            ppl = math.exp(avg_eval_loss)

            
            if accel_mb is None:
                log.info(
                    "Step %d: Loss %.4f | val %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB",
                    iter,
                    loss.item(),
                    avg_eval_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                )
            else:
                log.info(
                    "Step %d: Loss %.4f | val %.4f | ppl %.2f | iter %.3fs | elapsed %.1fs | rss %.1f MB | accel %.1f MB",
                    iter,
                    loss.item(),
                    avg_eval_loss,
                    ppl,
                    iter_time,
                    elapsed,
                    rss_mb,
                    accel_mb,
                )
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
