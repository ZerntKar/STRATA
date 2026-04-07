import torch
import numpy as np
from typing import List, Dict, Tuple, Set, Optional

from utils.lexicon import RuleAnchoredVerifier, AMPLIFY_PHRASES, WARNING_PHRASES
from utils.kg_utils import KnowledgeGraph


class SelectorReward:

    def __init__(self, attribute_names: List[str], pmi_weights: Dict[str, float], lambda1: float = 0.5, lambda2: float = 0.5):
        self.attribute_names = attribute_names
        self.pmi_weights = pmi_weights
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def feature_reward(self, selected_attrs: Set[str], gt_attrs: Set[str]) -> float:
        intersection = selected_attrs & gt_attrs
        union = selected_attrs | gt_attrs
        if not union:
            return 0.0
        num = sum(self.pmi_weights.get(k, 1.0) for k in intersection)
        den = sum(self.pmi_weights.get(k, 1.0) for k in union)
        return float(num / (den + 1e-8))

    def style_reward(self, action: List[Tuple[str, str]], gt_attrs: Set[str], gt_polarities: Dict[str, int]) -> float:
        action_dict = {attr: stance for attr, stance in action}
        overlap = set(action_dict.keys()) & gt_attrs
        if not overlap:
            return 0.0
        total_match = 0.0
        for attr in overlap:
            stance = action_dict[attr]
            polarity = gt_polarities.get(attr, 0)
            if polarity == 1 and stance == "Amplify":
                total_match += 1.0
            elif polarity == -1 and stance == "Warning":
                total_match += 1.0
        return total_match / len(overlap)

    def compute(self, action: List[Tuple[str, str]], gt_attrs: Set[str], gt_polarities: Dict[str, int]) -> float:
        selected_attrs = {attr for attr, _ in action}
        r_fea = self.feature_reward(selected_attrs, gt_attrs)
        r_style = self.style_reward(action, gt_attrs, gt_polarities)
        return self.lambda1 * r_fea + self.lambda2 * r_style


class GeneratorReward:

    def __init__(self, kg: KnowledgeGraph, verifier: RuleAnchoredVerifier, lambda3: float = 0.4, lambda4: float = 0.4, lambda5: float = 0.2):
        self.kg = kg
        self.verifier = verifier
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5

    def follow_reward(self, generated_text: str, action: List[Tuple[str, str]]) -> float:
        return self.verifier.compute_follow_reward(generated_text, action)

    def semantic_reward(self, generated_text: str, gt_review: str) -> float:
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn([generated_text], [gt_review], model_type="roberta-large", verbose=False)
            return float(F1[0].item())
        except Exception:
            gen_tokens = set(generated_text.lower().split())
            gt_tokens = set(gt_review.lower().split())
            if not gt_tokens:
                return 0.0
            return len(gen_tokens & gt_tokens) / len(gt_tokens)

    def factual_anchoring_reward(self, generated_text: str, action: List[Tuple[str, str]], item_id: str, category: str) -> float:
        warning_attrs = [(attr, stance) for attr, stance in action if stance == "Warning"]
        if not warning_attrs:
            return 1.0
        correct = 0.0
        for attr, _ in warning_attrs:
            kg_weakness = self.kg.weakness_score(item_id, attr, category)
            kg_direction = "negative" if kg_weakness > 0.5 else "positive"
            gen_direction = self._extract_text_direction(generated_text, attr)
            if gen_direction is not None:
                correct += 1.0 if kg_direction == gen_direction else 0.0
            else:
                correct += 0.5
        return correct / len(warning_attrs)

    def _extract_text_direction(self, text: str, attribute: str) -> Optional[str]:
        text_lower = text.lower()
        sentences = text_lower.split(".")
        for sent in sentences:
            if attribute.lower() in sent:
                has_warning = any(p in sent for p in WARNING_PHRASES)
                has_amplify = any(p in sent for p in AMPLIFY_PHRASES)
                if has_warning and not has_amplify:
                    return "negative"
                elif has_amplify and not has_warning:
                    return "positive"
        return None

    def compute(self, generated_text: str, action: List[Tuple[str, str]], gt_review: str, item_id: str, category: str) -> Tuple[float, Dict[str, float]]:
        r_follow = self.follow_reward(generated_text, action)
        r_sem = self.semantic_reward(generated_text, gt_review)
        r_fact = self.factual_anchoring_reward(generated_text, action, item_id, category)
        total = self.lambda3 * r_follow + self.lambda4 * r_sem + self.lambda5 * r_fact
        return total, {"r_follow": r_follow, "r_sem": r_sem, "r_fact": r_fact, "total": total}
