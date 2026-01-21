from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    cfg: Optional[Any] = None,
) -> None:
    ckpt_path = Path(path)
    ckpt = {"model_state": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if step is not None:
        ckpt["step"] = step
    if cfg is not None:
        try:
            from omegaconf import OmegaConf

            ckpt["cfg"] = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            ckpt["cfg"] = cfg

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str | torch.device] = None,
) -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
