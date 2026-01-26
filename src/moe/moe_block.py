import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.hidden_dim = getattr(cfg, "moe_hidden_dim", cfg.dim * 2)
        self.num_experts = getattr(cfg, "moe_num_experts", 4)
        self.top_k = getattr(cfg, "moe_top_k", 2)
        if self.top_k < 1:
            raise ValueError("cfg.moe_top_k must be >= 1")
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.dim),
                nn.Dropout(cfg.dropout)
            ) for _ in range(self.num_experts)
        ])

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
        # Flatten batch and seq for routing
        x_flat = x.reshape(-1, d)
        
        # TODO: Get router logits
        logits = self.router(x_flat)
        all_probs = F.softmax(logits, dim=-1)
        
        # TODO: Select top-k experts and their weights (probabilities)
        top_k_logits, top_k_inds = torch.topk(logits, k=self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        aux_loss = self.calculate_aux_loss(all_probs, top_k_inds)
        
        # TODO: Dispatch tokens to experts 
        # (Challenge: How do you do this efficiently without a for-loop?)
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices, top_k_level = torch.where(top_k_inds == i)
            if token_indices.numel() > 0:
                expert_input = x_flat[token_indices]
                expert_output = expert(expert_input)
                weights = top_k_probs[token_indices, top_k_level].unsqueeze(-1)
                final_output[token_indices] += expert_output * weights
        
        return final_output.view(b, s, d), aux_loss
