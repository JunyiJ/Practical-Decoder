import hydra
from omegaconf import DictConfig
import torch
from llm_lab.models.gpt import GPT

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def train(cfg: DictConfig):
    device = cfg.training.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    model = GPT(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.training.learning_rate)
    print(f"Starting training: {cfg.model.attn_type} + {cfg.model.mlp_type} on {device}")
    model.train()
    # TODO training logic


if __name__ == "__main__":
    train()