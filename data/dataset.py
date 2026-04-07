import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class RecommendationDataset(Dataset):

    def __init__(self, data_path: str, kg_path: str, split: str = "train", max_history_len: int = 10):
        self.split = split
        self.max_history_len = max_history_len
        self.data = self._load_data(data_path, split)
        self.kg = self._load_kg(kg_path)

    def _load_data(self, path: str, split: str) -> List[Dict]:
        samples = []
        with open(f"{path}/{split}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples

    def _load_kg(self, kg_path: str) -> Dict:
        with open(kg_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        history = sample["history"][-self.max_history_len:]
        item_attrs = self.kg.get(sample["item_id"], {})
        return {
            "user_id": sample["user_id"],
            "item_id": sample["item_id"],
            "category": sample.get("category", ""),
            "history": history,
            "item_attrs": item_attrs,
            "review": sample.get("review", ""),
            "rating": sample.get("rating", 3.0),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "user_ids": [b["user_id"] for b in batch],
        "item_ids": [b["item_id"] for b in batch],
        "categories": [b["category"] for b in batch],
        "histories": [b["history"] for b in batch],
        "item_attrs": [b["item_attrs"] for b in batch],
        "reviews": [b["review"] for b in batch],
        "ratings": torch.tensor([b["rating"] for b in batch], dtype=torch.float),
    }
