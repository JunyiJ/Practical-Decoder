from __future__ import annotations

import argparse
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


def _encode(prompt: str, stoi: dict[str, int]) -> torch.Tensor:
    try:
        ids = [stoi[ch] for ch in prompt]
    except KeyError as exc:
        raise ValueError(f"Unknown character in prompt: {exc.args[0]!r}") from exc
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            cutoff = values[:, [-1]]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tokens from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--data-path", required=True, help="Path to training text (for vocab).")
    parser.add_argument("--prompt", default="", help="Prompt string.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda")
    args = parser.parse_args()

    device = _select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "cfg" not in ckpt:
        raise ValueError("Checkpoint missing cfg; cannot reconstruct model.")
    cfg = OmegaConf.create(ckpt["cfg"])

    loader = TinyDataLoader(
        data_path=args.data_path,
        batch_size=1,
        block_size=cfg.model.block_size,
        device=device,
    )
    cfg.model.vocab_size = loader.vocab_size
    model = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = _encode(args.prompt, loader.stoi).unsqueeze(0).to(device)
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = loader.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
