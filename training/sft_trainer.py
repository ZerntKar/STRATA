import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List

from models.selector import StrategicPlanSelector
from models.generator import AttributeGroundedGenerator
from utils.absa import ABSAModel, extract_aspect_labels


class SFTTrainer:

    def __init__(
        self,
        selector: StrategicPlanSelector,
        generator: AttributeGroundedGenerator,
        absa_model: ABSAModel,
        attribute_names: List[str],
        learning_rate: float = 1e-5,
        device: str = "cuda",
    ):
        self.selector = selector
        self.generator = generator
        self.absa_model = absa_model
        self.attribute_names = attribute_names
        self.device = device

        self.selector_optimizer = torch.optim.AdamW(selector.parameters(), lr=learning_rate)
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in generator.parameters() if p.requires_grad], lr=learning_rate
        )

    def train_selector_step(self, state: torch.Tensor, gt_style_targets: torch.Tensor, validity_mask: torch.Tensor) -> float:
        self.selector.train()
        self.selector_optimizer.zero_grad()

        style_logits, _ = self.selector.forward(state, validity_mask)
        B, num_attrs, _ = style_logits.shape

        valid_mask_bool = validity_mask.bool()
        loss = nn.CrossEntropyLoss()(
            style_logits[valid_mask_bool].view(-1, 2),
            gt_style_targets[valid_mask_bool].view(-1).long()
        )
        loss.backward()
        self.selector_optimizer.step()
        return loss.item()

    def train_generator_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> float:
        self.generator.train()
        self.generator_optimizer.zero_grad()
        loss = self.generator.forward(input_ids, attention_mask, labels)
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def prepare_selector_targets(self, reviews: List[str], validity_masks: List[List[int]]) -> torch.Tensor:
        B = len(reviews)
        num_attrs = len(self.attribute_names)
        targets = torch.zeros(B, num_attrs, dtype=torch.long)
        for b, review in enumerate(reviews):
            _, pol_map = extract_aspect_labels(review, self.absa_model, self.attribute_names)
            for attr, pol in pol_map.items():
                if attr in self.attribute_names:
                    idx = self.attribute_names.index(attr)
                    targets[b, idx] = 0 if pol == 1 else 1
        return targets

    def run(self, dataloader: DataLoader, model, epochs: int = 5) -> None:
        for epoch in range(epochs):
            total_sel_loss = 0.0
            total_gen_loss = 0.0
            for batch in dataloader:
                reviews = batch["reviews"]
                user_ids = batch["user_ids"]
                item_ids = batch["item_ids"]
                categories = batch["categories"]

                for i in range(len(user_ids)):
                    item_seq = torch.zeros(1, 10, dtype=torch.long).to(self.device)
                    state = model.build_state(
                        user_ids[i], item_ids[i], item_seq,
                        list(batch["item_attrs"][i].keys()), categories[i], self.device
                    )
                    validity_mask_list = model.perception.get_validity_mask(item_ids[i])
                    validity_mask = torch.tensor(validity_mask_list, dtype=torch.float32, device=self.device).unsqueeze(0)
                    gt_targets = self.prepare_selector_targets([reviews[i]], [validity_mask_list])
                    gt_targets = gt_targets.to(self.device)
                    sel_loss = self.train_selector_step(state, gt_targets, validity_mask)
                    total_sel_loss += sel_loss

            print(f"Epoch {epoch+1} | Selector Loss: {total_sel_loss:.4f} | Generator Loss: {total_gen_loss:.4f}")
