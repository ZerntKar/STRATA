import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

STANCE_NAMES = ["Amplify", "Warning"]


class StrategicPlanSelector(nn.Module):

    def __init__(
        self,
        state_dim: int,
        attribute_embed_dim: int,
        num_attributes: int,
        hidden_dim: int = 128,
        attribute_names: List[str] = None,
    ):
        super().__init__()
        self.num_attributes = num_attributes
        self.attribute_names = attribute_names or [f"attr_{i}" for i in range(num_attributes)]
        self.attribute_embedding = nn.Embedding(num_attributes, attribute_embed_dim)

        self.mlp_style = nn.Sequential(
            nn.Linear(state_dim + attribute_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0]
        attr_ids = torch.arange(self.num_attributes, device=state.device)
        attr_embs = self.attribute_embedding(attr_ids)

        state_exp = state.unsqueeze(1).expand(B, self.num_attributes, -1)
        attr_embs_exp = attr_embs.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([state_exp, attr_embs_exp], dim=-1)

        style_logits = self.mlp_style(combined)

        if validity_mask is not None:
            mask_exp = (validity_mask == 0).unsqueeze(-1).expand_as(style_logits)
            style_logits = style_logits.masked_fill(mask_exp, float("-inf"))

        value = self.value_head(state)
        return style_logits, value

    def select_action(
        self,
        state: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        greedy: bool = False,
    ) -> Tuple[List[Tuple[str, str]], torch.Tensor, torch.Tensor]:
        style_logits, value = self.forward(state, validity_mask)
        style_probs = F.softmax(style_logits, dim=-1)

        if greedy:
            style_selected = style_probs.argmax(dim=-1)
        else:
            style_dist = torch.distributions.Categorical(style_probs)
            style_selected = style_dist.sample()

        log_probs = torch.distributions.Categorical(style_probs).log_prob(style_selected)

        action = []
        total_log_prob = torch.tensor(0.0, device=state.device)

        valid_mask = validity_mask[0] if validity_mask is not None else torch.ones(self.num_attributes, device=state.device)
        for attr_idx in range(self.num_attributes):
            if valid_mask[attr_idx].item() == 0:
                continue
            attr_name = self.attribute_names[attr_idx]
            stance_idx = style_selected[0, attr_idx].item()
            stance_name = STANCE_NAMES[int(stance_idx)]
            action.append((attr_name, stance_name))
            total_log_prob = total_log_prob + log_probs[0, attr_idx]

        return action, total_log_prob, value

    def compute_log_prob(
        self,
        state: torch.Tensor,
        style_targets: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        style_logits, _ = self.forward(state, validity_mask)
        style_probs = F.softmax(style_logits, dim=-1)
        log_probs = torch.distributions.Categorical(style_probs).log_prob(style_targets)
        return log_probs
