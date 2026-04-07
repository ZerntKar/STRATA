import json
import math
from typing import Dict, List, Optional


class KnowledgeGraph:

    def __init__(self, kg_path: str, stats_path: str):
        with open(kg_path, "r") as f:
            self.kg: Dict[str, Dict] = json.load(f)
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.category_stats: Dict = stats["category_stats"]
        self.direction: Dict[str, int] = stats.get("direction", {})
        self.hard_attrs: set = set(stats.get("hard_attrs", []))
        self.soft_attrs: set = set(stats.get("soft_attrs", []))

    def lookup(self, item_id: str) -> Dict:
        return self.kg.get(item_id, {})

    def get_validity_mask(self, item_id: str, all_attributes: List[str]) -> List[int]:
        item_attrs = self.lookup(item_id)
        mask = []
        for attr in all_attributes:
            if attr in self.hard_attrs and attr not in item_attrs:
                mask.append(0)
            else:
                mask.append(1)
        return mask

    def weakness_score(self, item_id: str, attribute: str, category: str, epsilon: float = 1e-6, tau: float = 5.0) -> float:
        item_attrs = self.lookup(item_id)
        if attribute not in item_attrs:
            return 0.0
        val = float(item_attrs[attribute])
        cat_stats = self.category_stats.get(category, {}).get(attribute, {})
        mu = float(cat_stats.get("mean", val))
        sigma = float(cat_stats.get("std", 1.0))
        s_k = self.direction.get(attribute, 1)
        z = tau * s_k * (mu - val) / (sigma + epsilon)
        return 1.0 / (1.0 + math.exp(-z))

    def serialize_facts(self, item_id: str) -> str:
        attrs = self.lookup(item_id)
        if not attrs:
            return "No factual attributes available."
        return "; ".join(f"{k}: {v}" for k, v in attrs.items())
