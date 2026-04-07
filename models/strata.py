import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

from models.perception import PerceptionLayer, SASRec
from models.memory import RAPM
from models.selector import StrategicPlanSelector
from models.generator import AttributeGroundedGenerator
from utils.kg_utils import KnowledgeGraph


class STRATA(nn.Module):

    def __init__(
        self,
        perception: PerceptionLayer,
        memory: RAPM,
        selector: StrategicPlanSelector,
        generator: AttributeGroundedGenerator,
        user_embed_dim: int,
        item_embed_dim: int,
        num_attributes: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.perception = perception
        self.memory = memory
        self.selector = selector
        self.generator = generator

        mem_dim = num_attributes * 2
        state_input_dim = user_embed_dim + mem_dim + item_embed_dim
        self.state_proj = nn.Linear(state_input_dim, hidden_dim)

    def build_state(
        self,
        user_id: int,
        item_id: str,
        item_seq: torch.Tensor,
        item_attr_names: List[str],
        category: str,
        device: torch.device,
    ) -> torch.Tensor:
        h_u = self.perception.get_user_embedding(item_seq)

        h_mem = self.memory.get_memory_vector(user_id).to(device)

        item_attrs = self.perception.get_item_facts(item_id)
        attr_values = list(item_attrs.values())
        if attr_values:
            h_i = torch.tensor(
                [float(v) if isinstance(v, (int, float)) else 0.0 for v in attr_values[:item_seq.shape[-1]]],
                dtype=torch.float32, device=device
            )
            if h_i.shape[0] < self.state_proj.in_features - h_u.shape[-1] - h_mem.shape[0]:
                pad_size = self.state_proj.in_features - h_u.shape[-1] - h_mem.shape[0] - h_i.shape[0]
                h_i = torch.cat([h_i, torch.zeros(pad_size, device=device)])
        else:
            h_i = torch.zeros(self.state_proj.in_features - h_u.shape[-1] - h_mem.shape[0], device=device)

        h_mem_b = h_mem.unsqueeze(0)
        h_i_b = h_i.unsqueeze(0)
        state_raw = torch.cat([h_u, h_mem_b, h_i_b], dim=-1)
        state = self.state_proj(state_raw)
        return state

    def forward(
        self,
        user_id: int,
        item_id: str,
        item_seq: torch.Tensor,
        item_attr_names: List[str],
        category: str,
        device: torch.device,
        greedy: bool = False,
    ) -> Tuple[List[Tuple[str, str]], str, torch.Tensor, torch.Tensor]:
        state = self.build_state(user_id, item_id, item_seq, item_attr_names, category, device)

        validity_mask_list = self.perception.get_validity_mask(item_id)
        validity_mask = torch.tensor(validity_mask_list, dtype=torch.float32, device=device).unsqueeze(0)

        action, log_prob, value = self.selector.select_action(state, validity_mask, greedy=greedy)

        item_facts = self.perception.serialize_facts(item_id)
        explanation = self.generator.generate(action, item_facts)

        return action, explanation, log_prob, value
