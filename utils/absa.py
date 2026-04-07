from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class ABSAResult:
    attribute: str
    polarity: int
    intensity: float
    confidence: float


class ABSAModel(nn.Module):

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.label2pol = {0: -1, 1: 0, 2: 1}

    def forward(self, reviews: List[str], attributes: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encodings = self.tokenizer(
            reviews, attributes,
            padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        outputs = self.encoder(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)
        return outputs.logits, probs

    @torch.no_grad()
    def extract(self, review: str, candidate_attributes: List[str], confidence_threshold: float = 0.75) -> List[ABSAResult]:
        if not candidate_attributes:
            return []
        reviews_rep = [review] * len(candidate_attributes)
        _, probs = self.forward(reviews_rep, candidate_attributes)
        results = []
        for i, attr in enumerate(candidate_attributes):
            prob = probs[i]
            pred_label = prob.argmax().item()
            confidence = prob.max().item()
            if confidence < confidence_threshold:
                continue
            polarity = self.label2pol[pred_label]
            if polarity == 0:
                continue
            intensity = abs(prob[2].item() - prob[0].item())
            results.append(ABSAResult(
                attribute=attr,
                polarity=polarity,
                intensity=float(intensity),
                confidence=float(confidence),
            ))
        return results


def extract_aspect_labels(review: str, absa_model: ABSAModel, candidate_attributes: List[str], threshold: float = 0.75) -> Tuple[List[str], Dict[str, int]]:
    results = absa_model.extract(review, candidate_attributes, threshold)
    attrs = [r.attribute for r in results]
    pol_map = {r.attribute: r.polarity for r in results}
    return attrs, pol_map
