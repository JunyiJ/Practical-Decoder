from __future__ import annotations

import argparse
import resource
import sys
import time
from typing import Optional

import torch
from omegaconf import OmegaConf

from ..attention.cache import KVCache
from ..data.loaders import TinyDataLoader
from ..models.gpt import GPT

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency for memory stats
    psutil = None


def _select_device(requested: Optional[str]) -> str:
    if requested is None:
        return "cpu"
    device = requested
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _device_type(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def _get_mem_usage(device) -> float:
    """Returns memory usage in MB."""
    device_type = _device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    if device_type == "mps" and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            return current() / 1e6
        return 0.0
    if psutil is not None:
        return psutil.Process().memory_info().rss / 1e6
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    return rss_bytes / 1e6


def _encode(prompt: str, stoi: dict[str, int]) -> torch.Tensor:
    try:
        ids = [stoi[ch] for ch in prompt]
    except KeyError as exc:
        raise ValueError(f"Unknown character in prompt: {exc.args[0]!r}") from exc
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def _generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    cache: Optional[list[KVCache]] = None,
    greedy: bool = False,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return idx
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    def _sample_next(logits: torch.Tensor) -> torch.Tensor:
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            cutoff = values[:, [-1]]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    if cache is not None:
        if idx.size(1) == 0:
            raise ValueError("Prompt must be non-empty when using KV cache.")
        logits, _ = model(idx, cache=cache, start_pos=0)
        next_id = _sample_next(logits[:, -1, :])
        idx = torch.cat((idx, next_id), dim=1)
        for _ in range(max_new_tokens - 1):
            start_pos = idx.size(1) - 1
            logits, _ = model(idx[:, -1:], cache=cache, start_pos=start_pos)
            next_id = _sample_next(logits[:, -1, :])
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        next_id = _sample_next(logits[:, -1, :])
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def _load_model(
    checkpoint: str,
    data_path: str,
    device: str,
) -> tuple[GPT, OmegaConf, TinyDataLoader]:
    ckpt = torch.load(checkpoint, map_location=device)
    if "cfg" not in ckpt:
        raise ValueError(f"Checkpoint missing cfg: {checkpoint}")
    cfg = OmegaConf.create(ckpt["cfg"])
    loader = TinyDataLoader(
        data_path=data_path,
        batch_size=1,
        block_size=cfg.model.block_size,
        device=device,
    )
    cfg.model.vocab_size = loader.vocab_size
    model = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, loader


def _benchmark(
    model: GPT,
    cfg: OmegaConf,
    idx: torch.Tensor,
    max_new_tokens: int,
    enable_cache: bool,
    temperature: float,
    top_k: Optional[int],
    greedy: bool,
) -> tuple[float, float]:
    if enable_cache:
        max_allowed = cfg.model.block_size - idx.size(1)
        if max_allowed <= 0:
            raise ValueError("Prompt length exceeds or equals block_size when cache is enabled.")
        if max_new_tokens > max_allowed:
            print(
                f"Warning: truncating max_new_tokens from {max_new_tokens} to {max_allowed} "
                "due to block_size limit with KV cache."
            )
            max_new_tokens = max_allowed
    caches = None
    if enable_cache and cfg.model.attn_type in ("mha", "gqa"):
        n_kv_heads = cfg.model.n_kv_heads if cfg.model.attn_type == "gqa" else cfg.model.n_heads
        caches = [
            KVCache(
                cfg,
                idx.device,
                batch_size=idx.size(0),
                dtype=next(model.parameters()).dtype,
                n_kv_heads=n_kv_heads,
            )
            for _ in range(cfg.model.n_layers)
        ]

    # Warmup
    _generate(
        model,
        idx,
        max_new_tokens=5,
        temperature=temperature,
        top_k=top_k,
        cache=caches,
        greedy=greedy,
    )
    if caches is not None:
        for cache in caches:
            cache.reset()

    start_mem = _get_mem_usage(idx.device)
    start_time = time.perf_counter()
    _generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        cache=caches,
        greedy=greedy,
    )
    end_time = time.perf_counter()
    end_mem = _get_mem_usage(idx.device)

    total_time = max(end_time - start_time, 1e-9)
    tokens_sec = max_new_tokens / total_time
    mem_delta = end_mem - start_mem
    return tokens_sec, mem_delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MHA vs GQA and print a Markdown table."
    )
    parser.add_argument("--mha-checkpoint", required=True, help="Path to MHA checkpoint.")
    parser.add_argument("--gqa-checkpoint", required=True, help="Path to GQA checkpoint.")
    parser.add_argument("--data-path", required=True, help="Path to training text (for vocab).")
    parser.add_argument("--prompt", default="", help="Prompt string.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda")
    parser.add_argument("--enable-cache", action="store_true", help="Enable KV cache for decoding")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (argmax)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    device = _select_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    mha_model, mha_cfg, mha_loader = _load_model(args.mha_checkpoint, args.data_path, device)
    gqa_model, gqa_cfg, gqa_loader = _load_model(args.gqa_checkpoint, args.data_path, device)

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    mha_prompt = _encode(args.prompt, mha_loader.stoi).unsqueeze(0)
    gqa_prompt = _encode(args.prompt, gqa_loader.stoi).unsqueeze(0)
    mha_idx = mha_prompt.expand(args.batch_size, -1).contiguous().to(device)
    gqa_idx = gqa_prompt.expand(args.batch_size, -1).contiguous().to(device)

    mha_tokens_sec, mha_mem = _benchmark(
        mha_model,
        mha_cfg,
        mha_idx,
        max_new_tokens=args.max_new_tokens,
        enable_cache=args.enable_cache,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    gqa_tokens_sec, gqa_mem = _benchmark(
        gqa_model,
        gqa_cfg,
        gqa_idx,
        max_new_tokens=args.max_new_tokens,
        enable_cache=args.enable_cache,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )

    cache_label = "on" if args.enable_cache else "off"
    print("| Model | Cache | Tokens/sec | Mem MB |")
    print("| --- | --- | ---: | ---: |")
    print(f"| MHA | {cache_label} | {mha_tokens_sec:.2f} | {mha_mem:.2f} |")
    print(f"| GQA | {cache_label} | {gqa_tokens_sec:.2f} | {gqa_mem:.2f} |")


if __name__ == "__main__":
    main()
