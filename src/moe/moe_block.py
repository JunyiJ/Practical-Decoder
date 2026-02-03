import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.mlp import SwiGLU


class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.hidden_dim = getattr(cfg, "moe_hidden_dim", cfg.dim * 4)
        self.shared_num_experts = getattr(cfg, "moe_shared_num_experts", 0)
        self.num_experts = getattr(cfg, "moe_num_experts", 4)
        self.top_k = getattr(cfg, "moe_top_k", 2)
        self.record_expert_hist = getattr(cfg, "moe_record_expert_hist", True)
        self.last_expert_hist = None
        if self.top_k < 1:
            raise ValueError("cfg.moe_top_k must be >= 1")
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        self.ffn = getattr(cfg, "moe_ffn", "gelu")
        if self.ffn == "gelu":
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.dim),
                    nn.Dropout(cfg.dropout)
                ) for _ in range(self.num_experts + self.shared_num_experts)
            ])
        elif self.ffn == "swiglu":
            self.experts = nn.ModuleList([
                SwiGLU(self.dim, self.hidden_dim, cfg.dropout)
                for _ in range(self.num_experts + self.shared_num_experts)
            ])
        else:
            raise ValueError(f"cfg.moe_ffn has unsupported type: {self.ffn}")

    def calculate_aux_loss(self, router_probs, selected_experts):
        """
        router_probs: Softmax output of the router (TotalTokens, NumExperts)
        selected_experts: The indices chosen (TotalTokens, TopK)
        """
        # (TotalTokens, TopK, NumExperts)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        f_i = expert_mask.mean(dim=(0, 1))
        P_i = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        return aux_loss


    def forward(self, x):
        """
        x shape: (batch_size, seq_len, hidden_dim)
        """
        b, s, d = x.shape
        x_flat = x.view(-1, d)
        final_output = torch.zeros_like(x_flat)
        router_x = self.router(x_flat)
        router_probs = F.softmax(router_x, dim=-1)
        top_k_values, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
        aux_loss = self.calculate_aux_loss(router_probs, top_k_indices)
        if self.record_expert_hist:
            with torch.no_grad():
                self.last_expert_hist = torch.bincount(
                    top_k_indices.reshape(-1),
                    minlength=self.num_experts,
                )
        for i in range(self.num_experts):
            indices, experts = torch.where(top_k_indices == i)
            if indices.numel() > 0:
                weight = top_k_probs[indices, experts].unsqueeze(dim=-1)
                expert_output = self.experts[i](x_flat[indices])
                final_output[indices] += weight * expert_output
        if self.shared_num_experts > 0:
            for i in range(self.num_experts, self.num_experts + self.shared_num_experts):
                final_output += self.experts[i](x_flat)
        return final_output.view(b, s, d), aux_loss
