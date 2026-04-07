import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import copy

from models.strata import STRATA
from training.rewards import SelectorReward, GeneratorReward
from utils.absa import ABSAModel, extract_aspect_labels


class CooperativePPOTrainer:

    def __init__(
        self,
        model: STRATA,
        selector_reward: SelectorReward,
        generator_reward: GeneratorReward,
        absa_model: ABSAModel,
        attribute_names: List[str],
        beta_kl_init: float = 0.1,
        beta_kl_final: float = 0.01,
        ppo_clip_eps: float = 0.2,
        ppo_value_coef: float = 0.1,
        ppo_epochs: int = 3,
        alternating_rounds: int = 3,
        learning_rate: float = 1e-5,
        device: str = "cuda",
    ):
        self.model = model
        self.selector_reward_fn = selector_reward
        self.generator_reward_fn = generator_reward
        self.absa_model = absa_model
        self.attribute_names = attribute_names
        self.beta_kl_init = beta_kl_init
        self.beta_kl_final = beta_kl_final
        self.ppo_clip_eps = ppo_clip_eps
        self.ppo_value_coef = ppo_value_coef
        self.ppo_epochs = ppo_epochs
        self.alternating_rounds = alternating_rounds
        self.device = device

        self.ref_selector = copy.deepcopy(model.selector)
        for p in self.ref_selector.parameters():
            p.requires_grad = False

        self.selector_optimizer = torch.optim.AdamW(model.selector.parameters(), lr=learning_rate)
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in model.generator.parameters() if p.requires_grad], lr=learning_rate
        )

    def _get_beta(self, current_step: int, total_steps: int) -> float:
        progress = current_step / max(total_steps, 1)
        return self.beta_kl_init + (self.beta_kl_final - self.beta_kl_init) * progress

    def _compute_selector_kl(self, state: torch.Tensor, validity_mask: torch.Tensor) -> torch.Tensor:
        style_logits_cur, _ = self.model.selector.forward(state, validity_mask)
        with torch.no_grad():
            style_logits_ref, _ = self.ref_selector.forward(state, validity_mask)

        cur_probs = F.softmax(style_logits_cur, dim=-1)
        ref_probs = F.softmax(style_logits_ref, dim=-1)
        kl = F.kl_div(cur_probs.log(), ref_probs, reduction="batchmean")
        return kl

    def _selector_ppo_step(
        self,
        state: torch.Tensor,
        validity_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        style_targets: torch.Tensor,
        advantages: torch.Tensor,
        values_old: torch.Tensor,
        returns: torch.Tensor,
        beta: float,
    ) -> float:
        self.model.selector.train()
        self.selector_optimizer.zero_grad()

        new_log_probs = self.model.selector.compute_log_prob(state, style_targets, validity_mask)
        new_log_prob_sum = new_log_probs.sum(dim=-1)
        old_log_prob_sum = old_log_probs.sum(dim=-1)

        ratio = torch.exp(new_log_prob_sum - old_log_prob_sum)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        _, values_new = self.model.selector.forward(state, validity_mask)
        value_loss = F.mse_loss(values_new.squeeze(-1), returns)

        kl_penalty = self._compute_selector_kl(state, validity_mask)

        loss = policy_loss + self.ppo_value_coef * value_loss + beta * kl_penalty
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.selector.parameters(), 1.0)
        self.selector_optimizer.step()
        return loss.item()

    def _generator_ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        old_gen_log_probs: torch.Tensor,
        gen_advantages: torch.Tensor,
        beta: float,
    ) -> float:
        self.model.generator.train()
        self.generator_optimizer.zero_grad()

        new_gen_log_probs = self.model.generator.compute_log_probs(input_ids, attention_mask, response_ids)
        ratio = torch.exp(new_gen_log_probs - old_gen_log_probs)
        surr1 = ratio * gen_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * gen_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        loss = policy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.generator.parameters() if p.requires_grad], 1.0
        )
        self.generator_optimizer.step()
        return loss.item()

    def train_step(
        self,
        user_id: int,
        item_id: str,
        item_seq: torch.Tensor,
        item_attr_names: List[str],
        category: str,
        gt_review: str,
        gt_attrs: set,
        gt_polarities: Dict[str, int],
        current_step: int,
        total_steps: int,
    ) -> Dict[str, float]:
        beta = self._get_beta(current_step, total_steps)

        state = self.model.build_state(user_id, item_id, item_seq, item_attr_names, category, self.device)
        validity_mask_list = self.model.perception.get_validity_mask(item_id)
        validity_mask = torch.tensor(validity_mask_list, dtype=torch.float32, device=self.device).unsqueeze(0)

        action, explanation, old_log_prob, value_old = self.model.forward(
            user_id, item_id, item_seq, item_attr_names, category, self.device
        )

        sel_reward = self.selector_reward_fn.compute(action, gt_attrs, gt_polarities)
        gen_reward, gen_components = self.generator_reward_fn.compute(
            explanation, action, gt_review, item_id, category
        )

        sel_advantage = torch.tensor(sel_reward, dtype=torch.float32, device=self.device) - value_old.squeeze()
        gen_advantage = torch.tensor(gen_reward, dtype=torch.float32, device=self.device)

        num_attrs = len(self.attribute_names)
        style_targets = torch.zeros(1, num_attrs, dtype=torch.long, device=self.device)
        for attr, stance in action:
            if attr in self.attribute_names:
                idx = self.attribute_names.index(attr)
                style_targets[0, idx] = 0 if stance == "Amplify" else 1

        old_log_probs_per_attr = self.model.selector.compute_log_prob(
            state.detach(), style_targets, validity_mask
        ).detach()

        returns = torch.tensor(sel_reward, dtype=torch.float32, device=self.device).unsqueeze(0)

        sel_loss_total = 0.0
        for _ in range(self.ppo_epochs):
            sel_loss = self._selector_ppo_step(
                state.detach(), validity_mask, old_log_probs_per_attr,
                style_targets, sel_advantage.detach(), value_old.detach(),
                returns, beta
            )
            sel_loss_total += sel_loss

        item_facts = self.model.perception.serialize_facts(item_id)
        prompt = self.model.generator.build_input(action, item_facts)
        gen_inputs = self.model.generator.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        response_ids = self.model.generator.tokenizer(
            explanation, return_tensors="pt", truncation=True, max_length=256
        ).input_ids.to(self.device)

        old_gen_log_probs = self.model.generator.compute_log_probs(
            gen_inputs["input_ids"], gen_inputs["attention_mask"], response_ids
        ).detach()

        gen_loss_total = 0.0
        for _ in range(self.ppo_epochs):
            gen_loss = self._generator_ppo_step(
                gen_inputs["input_ids"], gen_inputs["attention_mask"],
                response_ids, old_gen_log_probs, gen_advantage.detach(), beta
            )
            gen_loss_total += gen_loss

        absa_results = self.absa_model.extract(gt_review, self.attribute_names)
        self.model.memory.update(user_id, absa_results)

        return {
            "sel_reward": sel_reward,
            "gen_reward": gen_reward,
            "sel_loss": sel_loss_total / self.ppo_epochs,
            "gen_loss": gen_loss_total / self.ppo_epochs,
            "beta_kl": beta,
            **gen_components,
        }
