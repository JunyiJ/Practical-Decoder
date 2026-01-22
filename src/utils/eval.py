"""
python -m src.utils.eval \
  --checkpoint checkpoints/final_20260121_161140.pt \
  --data-path data/raw/shakespeare.txt \
  --eval-iters 100 \
  --device mps
"""
from __future__ import annotations

import argparse
import math
from typing import Optional

import torch
from omegaconf import OmegaConf

from ..data.loaders import TinyDataLoader
from ..models.gpt import GPT


def _select_device(requested: Optional[str]) -> str:
    if requested is None:
        return "cpu"
    device = requested
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


@torch.no_grad()
def evaluate(model: GPT, loader: TinyDataLoader, eval_iters: int) -> tuple[float, float]:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = loader.get_batch("val")
        _, loss = model(xb, yb)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on TinyDataLoader.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--data-path", required=True, help="Path to training text (for vocab).")
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda")
    args = parser.parse_args()

    device = _select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "cfg" not in ckpt:
        raise ValueError("Checkpoint missing cfg; cannot reconstruct model.")
    cfg = OmegaConf.create(ckpt["cfg"])

    loader = TinyDataLoader(
        data_path=args.data_path,
        batch_size=cfg.training.batch_size,
        block_size=cfg.model.block_size,
        device=device,
    )
    cfg.model.vocab_size = loader.vocab_size
    model = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state"])

    avg_loss, ppl = evaluate(model, loader, args.eval_iters)
    print(f"val_loss={avg_loss:.4f} val_ppl={ppl:.2f}")


if __name__ == "__main__":
    main()
