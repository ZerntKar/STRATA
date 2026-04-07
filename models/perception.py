import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from utils.kg_utils import KnowledgeGraph


class SASRec(nn.Module):

    def __init__(self, num_items: int, embedding_dim: int = 64, num_heads: int = 2, num_layers: int = 2, max_seq_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=embedding_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embedding_dim, num_items)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        B, T = item_seq.shape
        positions = torch.arange(T, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(item_seq) + self.pos_embedding(positions)
        x = self.dropout(self.layer_norm(x))
        causal_mask = torch.triu(torch.ones(T, T, device=item_seq.device), diagonal=1).bool()
        x = self.transformer(x, mask=causal_mask)
        return self.output_proj(x[:, -1, :])

    def get_user_embedding(self, item_seq: torch.Tensor) -> torch.Tensor:
        B, T = item_seq.shape
        positions = torch.arange(T, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(item_seq) + self.pos_embedding(positions)
        x = self.layer_norm(x)
        x = self.transformer(x)
        return x[:, -1, :]


class PerceptionLayer(nn.Module):

    def __init__(self, rec_model: SASRec, kg: KnowledgeGraph, top_k: int = 20, all_attributes: List[str] = None):
        super().__init__()
        self.rec_model = rec_model
        self.kg = kg
        self.top_k = top_k
        self.all_attributes = all_attributes or []
        for param in self.rec_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_candidates(self, item_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.rec_model(item_seq)
        scores, indices = torch.topk(logits, self.top_k, dim=-1)
        return indices, scores

    def get_item_facts(self, item_id: str) -> Dict:
        return self.kg.lookup(item_id)

    def get_validity_mask(self, item_id: str) -> List[int]:
        return self.kg.get_validity_mask(item_id, self.all_attributes)

    def serialize_facts(self, item_id: str) -> str:
        return self.kg.serialize_facts(item_id)

    def get_user_embedding(self, item_seq: torch.Tensor) -> torch.Tensor:
        return self.rec_model.get_user_embedding(item_seq)
