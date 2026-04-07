import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict

from utils.absa import ABSAResult
from utils.kg_utils import KnowledgeGraph


class RAPM(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_attributes: int,
        attribute_names: List[str],
        kg: KnowledgeGraph,
        eta_pos: float = 0.1,
        eta_neg: float = 1.0,
        eta_decay: float = 0.2,
        rho_absa: float = 0.75,
        tau: float = 5.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_attributes = num_attributes
        self.attribute_names = attribute_names
        self.attr2idx = {a: i for i, a in enumerate(attribute_names)}
        self.kg = kg
        self.eta_pos = eta_pos
        self.eta_neg = eta_neg
        self.eta_decay = eta_decay
        self.rho_absa = rho_absa
        self.tau = tau

        self.pref_strength = np.full((num_users, num_attributes), 0.5, dtype=np.float32)
        self.risk_sensitivity = np.zeros((num_users, num_attributes), dtype=np.float32)

    def overall_appeal(self, user_id: int, item_attr_names: List[str]) -> float:
        if not item_attr_names:
            return 0.5
        indices = [self.attr2idx[a] for a in item_attr_names if a in self.attr2idx]
        if not indices:
            return 0.5
        return float(np.mean(self.pref_strength[user_id, indices]))

    def risk_of_rejection(self, user_id: int, item_id: str, item_attr_names: List[str], category: str) -> float:
        if not item_attr_names:
            return 0.0
        max_risk = 0.0
        for attr in item_attr_names:
            if attr not in self.attr2idx:
                continue
            idx = self.attr2idx[attr]
            r_uk = self.risk_sensitivity[user_id, idx]
            g_ik = self.kg.weakness_score(item_id, attr, category, tau=self.tau)
            max_risk = max(max_risk, r_uk * g_ik)
        return float(max_risk)

    def get_memory_vector(self, user_id: int) -> torch.Tensor:
        pref_vec = torch.tensor(self.pref_strength[user_id], dtype=torch.float32)
        risk_vec = torch.tensor(self.risk_sensitivity[user_id], dtype=torch.float32)
        return torch.cat([pref_vec, risk_vec], dim=0)

    def update(self, user_id: int, absa_results: List[ABSAResult]) -> None:
        for result in absa_results:
            if result.confidence < self.rho_absa:
                continue
            if result.attribute not in self.attr2idx:
                continue
            idx = self.attr2idx[result.attribute]
            v = result.intensity
            kappa = result.confidence
            if result.polarity == -1:
                self.risk_sensitivity[user_id, idx] = np.clip(
                    self.risk_sensitivity[user_id, idx] + self.eta_neg * v * kappa, 0.0, 1.0
                )
            elif result.polarity == 1:
                self.pref_strength[user_id, idx] = np.clip(
                    self.pref_strength[user_id, idx] + self.eta_pos * v * kappa, 0.0, 1.0
                )
                self.risk_sensitivity[user_id, idx] = np.clip(
                    self.risk_sensitivity[user_id, idx] * (1.0 - self.eta_decay * v * kappa), 0.0, 1.0
                )
