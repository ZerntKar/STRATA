import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Set

from config.config import STRATAConfig
from data.dataset import RecommendationDataset, collate_fn
from models.perception import SASRec, PerceptionLayer
from models.memory import RAPM
from models.selector import StrategicPlanSelector
from models.generator import AttributeGroundedGenerator
from models.strata import STRATA
from training.rewards import SelectorReward, GeneratorReward
from training.sft_trainer import SFTTrainer
from training.ppo_trainer import CooperativePPOTrainer
from utils.absa import ABSAModel
from utils.kg_utils import KnowledgeGraph
from utils.lexicon import RuleAnchoredVerifier
from evaluation.metrics import evaluate_batch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_attribute_names(data_dir: str) -> List[str]:
    path = os.path.join(data_dir, "attributes.json")
    with open(path, "r") as f:
        return json.load(f)


def load_pmi_weights(data_dir: str) -> Dict[str, float]:
    path = os.path.join(data_dir, "pmi_weights.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_synonym_map(data_dir: str) -> Dict[str, List[str]]:
    path = os.path.join(data_dir, "synonym_map.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_user_item_counts(data_dir: str) -> tuple:
    path = os.path.join(data_dir, "meta.json")
    with open(path, "r") as f:
        meta = json.load(f)
    return meta["num_users"], meta["num_items"]


def build_model(cfg: STRATAConfig, attribute_names: List[str], num_users: int, num_items: int, kg: KnowledgeGraph, device: torch.device) -> STRATA:
    rec_model = SASRec(
        num_items=num_items,
        embedding_dim=cfg.embedding_dim,
        max_seq_len=cfg.max_history_len,
    )

    rec_ckpt = os.path.join(cfg.data_dir, "sasrec.pt")
    if os.path.exists(rec_ckpt):
        rec_model.load_state_dict(torch.load(rec_ckpt, map_location=device))
        print(f"[Perception] Loaded SASRec checkpoint from {rec_ckpt}")
    else:
        print("[Perception] No SASRec checkpoint found, using random initialization.")

    rec_model = rec_model.to(device)

    perception = PerceptionLayer(
        rec_model=rec_model,
        kg=kg,
        top_k=cfg.top_k_candidates,
        all_attributes=attribute_names,
    )

    memory = RAPM(
        num_users=num_users,
        num_attributes=cfg.num_attributes,
        attribute_names=attribute_names,
        kg=kg,
        eta_pos=cfg.eta_pos,
        eta_neg=cfg.eta_neg,
        eta_decay=cfg.eta_decay,
        rho_absa=cfg.rho_absa,
        tau=cfg.tau,
    )

    mem_dim = cfg.num_attributes * 2
    state_dim = cfg.embedding_dim + mem_dim + cfg.embedding_dim

    selector = StrategicPlanSelector(
        state_dim=state_dim,
        attribute_embed_dim=cfg.attribute_embed_dim,
        num_attributes=cfg.num_attributes,
        hidden_dim=cfg.selector_hidden_dim,
        attribute_names=attribute_names,
    ).to(device)

    generator = AttributeGroundedGenerator(
        model_name=cfg.llm_model_name,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        load_in_8bit=cfg.load_in_8bit,
        max_gen_length=cfg.max_gen_length,
    )

    model = STRATA(
        perception=perception,
        memory=memory,
        selector=selector,
        generator=generator,
        user_embed_dim=cfg.embedding_dim,
        item_embed_dim=cfg.embedding_dim,
        num_attributes=cfg.num_attributes,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    return model


def run_sft(cfg: STRATAConfig, model: STRATA, train_loader: DataLoader, attribute_names: List[str], absa_model: ABSAModel, device: torch.device):
    print("\n" + "=" * 50)
    print("Phase 1: Supervised Fine-Tuning (SFT Warm-up)")
    print("=" * 50)

    sft_trainer = SFTTrainer(
        selector=model.selector,
        generator=model.generator,
        absa_model=absa_model,
        attribute_names=attribute_names,
        learning_rate=cfg.learning_rate,
        device=cfg.device,
    )

    sft_trainer.run(
        dataloader=train_loader,
        model=model,
        epochs=cfg.sft_epochs,
    )

    ckpt_path = os.path.join(cfg.data_dir, "checkpoints", "sft_selector.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.selector.state_dict(), ckpt_path)
    print(f"[SFT] Selector checkpoint saved to {ckpt_path}")


def run_rl(cfg: STRATAConfig, model: STRATA, train_loader: DataLoader, attribute_names: List[str], pmi_weights: Dict[str, float], kg: KnowledgeGraph, absa_model: ABSAModel, synonym_map: Dict[str, List[str]], device: torch.device):
    print("\n" + "=" * 50)
    print("Phase 2: Cooperative PPO (KL-Regularized RL)")
    print("=" * 50)

    verifier = RuleAnchoredVerifier(synonym_map=synonym_map)

    selector_reward_fn = SelectorReward(
        attribute_names=attribute_names,
        pmi_weights=pmi_weights,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
    )

    generator_reward_fn = GeneratorReward(
        kg=kg,
        verifier=verifier,
        lambda3=cfg.lambda3,
        lambda4=cfg.lambda4,
        lambda5=cfg.lambda5,
    )

    ppo_trainer = CooperativePPOTrainer(
        model=model,
        selector_reward=selector_reward_fn,
        generator_reward=generator_reward_fn,
        absa_model=absa_model,
        attribute_names=attribute_names,
        beta_kl_init=cfg.beta_kl_init,
        beta_kl_final=cfg.beta_kl_final,
        ppo_clip_eps=cfg.ppo_clip_eps,
        ppo_epochs=cfg.ppo_epochs,
        alternating_rounds=cfg.alternating_rounds,
        learning_rate=cfg.learning_rate,
        device=cfg.device,
    )

    total_steps = cfg.rl_epochs * len(train_loader)
    current_step = 0

    for epoch in range(cfg.rl_epochs):
        epoch_metrics = {
            "sel_reward": [],
            "gen_reward": [],
            "sel_loss": [],
            "gen_loss": [],
            "r_follow": [],
            "r_sem": [],
            "r_fact": [],
        }

        for batch in train_loader:
            user_ids = batch["user_ids"]
            item_ids = batch["item_ids"]
            categories = batch["categories"]
            reviews = batch["reviews"]
            item_attrs_list = batch["item_attrs"]

            for i in range(len(user_ids)):
                item_seq = torch.zeros(1, cfg.max_history_len, dtype=torch.long).to(device)
                item_attr_names = list(item_attrs_list[i].keys())

                gt_attrs_raw, gt_polarities = _extract_gt_labels(
                    reviews[i], absa_model, attribute_names
                )
                gt_attrs = set(gt_attrs_raw)
                gt_neg_attrs = {a for a, p in gt_polarities.items() if p == -1}

                step_metrics = ppo_trainer.train_step(
                    user_id=user_ids[i],
                    item_id=item_ids[i],
                    item_seq=item_seq,
                    item_attr_names=item_attr_names,
                    category=categories[i],
                    gt_review=reviews[i],
                    gt_attrs=gt_attrs,
                    gt_polarities=gt_polarities,
                    current_step=current_step,
                    total_steps=total_steps,
                )

                for k in epoch_metrics:
                    if k in step_metrics:
                        epoch_metrics[k].append(step_metrics[k])

                current_step += 1

        avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_metrics.items()}
        print(
            f"[RL Epoch {epoch+1}/{cfg.rl_epochs}] "
            f"SelR={avg['sel_reward']:.4f} | GenR={avg['gen_reward']:.4f} | "
            f"R_follow={avg['r_follow']:.4f} | R_sem={avg['r_sem']:.4f} | R_fact={avg['r_fact']:.4f} | "
            f"SelLoss={avg['sel_loss']:.4f} | GenLoss={avg['gen_loss']:.4f}"
        )

    ckpt_sel = os.path.join(cfg.data_dir, "checkpoints", "rl_selector.pt")
    os.makedirs(os.path.dirname(ckpt_sel), exist_ok=True)
    torch.save(model.selector.state_dict(), ckpt_sel)
    print(f"[RL] Selector checkpoint saved to {ckpt_sel}")


def run_evaluation(cfg: STRATAConfig, model: STRATA, test_loader: DataLoader, attribute_names: List[str], absa_model: ABSAModel, kg: KnowledgeGraph, synonym_map: Dict[str, List[str]], device: torch.device):
    print("\n" + "=" * 50)
    print("Phase 3: Evaluation")
    print("=" * 50)

    model.eval()
    all_generated = []
    all_references = []
    all_neg_attrs = []
    all_kg_facts = []
    all_pref_attrs = []

    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch["user_ids"]
            item_ids = batch["item_ids"]
            categories = batch["categories"]
            reviews = batch["reviews"]
            item_attrs_list = batch["item_attrs"]

            for i in range(len(user_ids)):
                item_seq = torch.zeros(1, cfg.max_history_len, dtype=torch.long).to(device)
                item_attr_names = list(item_attrs_list[i].keys())

                _, explanation, _, _ = model.forward(
                    user_id=user_ids[i],
                    item_id=item_ids[i],
                    item_seq=item_seq,
                    item_attr_names=item_attr_names,
                    category=categories[i],
                    device=device,
                    greedy=True,
                )

                gt_attrs_raw, gt_polarities = _extract_gt_labels(reviews[i], absa_model, attribute_names)
                neg_attrs = {a for a, p in gt_polarities.items() if p == -1}
                pref_attrs = {a for a, p in gt_polarities.items() if p == 1}
                kg_facts = kg.lookup(item_ids[i])

                all_generated.append(explanation)
                all_references.append(reviews[i])
                all_neg_attrs.append(neg_attrs)
                all_kg_facts.append(kg_facts)
                all_pref_attrs.append(pref_attrs)

    results = evaluate_batch(
        generated_texts=all_generated,
        reference_texts=all_references,
        negative_attrs_list=all_neg_attrs,
        kg_facts_list=all_kg_facts,
        user_pref_attrs_list=all_pref_attrs,
        synonym_map=synonym_map,
    )

    print("\n[Evaluation Results]")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    results_path = os.path.join(cfg.data_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Evaluation] Results saved to {results_path}")

    sample_path = os.path.join(cfg.data_dir, "generated_samples.jsonl")
    with open(sample_path, "w") as f:
        for gen, ref in zip(all_generated[:50], all_references[:50]):
            f.write(json.dumps({"generated": gen, "reference": ref}) + "\n")
    print(f"[Evaluation] Sample outputs saved to {sample_path}")


def _extract_gt_labels(review: str, absa_model: ABSAModel, attribute_names: List[str]):
    from utils.absa import extract_aspect_labels
    return extract_aspect_labels(review, absa_model, attribute_names)


def main():
    cfg = STRATAConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Config] Dataset: {cfg.dataset} | Device: {device}")

    attribute_names = load_attribute_names(cfg.data_dir)
    pmi_weights = load_pmi_weights(cfg.data_dir)
    synonym_map = load_synonym_map(cfg.data_dir)
    num_users, num_items = load_user_item_counts(cfg.data_dir)
    cfg.num_attributes = len(attribute_names)

    print(f"[Data] Users: {num_users} | Items: {num_items} | Attributes: {cfg.num_attributes}")

    kg = KnowledgeGraph(
        kg_path=os.path.join(cfg.data_dir, "kg.json"),
        stats_path=os.path.join(cfg.data_dir, "kg_stats.json"),
    )

    absa_model = ABSAModel(model_name="bert-base-uncased").to(device)
    absa_ckpt = os.path.join(cfg.data_dir, "absa.pt")
    if os.path.exists(absa_ckpt):
        absa_model.load_state_dict(torch.load(absa_ckpt, map_location=device))
        print(f"[ABSA] Loaded checkpoint from {absa_ckpt}")
    absa_model.eval()

    train_dataset = RecommendationDataset(cfg.data_dir, os.path.join(cfg.data_dir, "kg.json"), split="train", max_history_len=cfg.max_history_len)
    val_dataset = RecommendationDataset(cfg.data_dir, os.path.join(cfg.data_dir, "kg.json"), split="valid", max_history_len=cfg.max_history_len)
    test_dataset = RecommendationDataset(cfg.data_dir, os.path.join(cfg.data_dir, "kg.json"), split="test", max_history_len=cfg.max_history_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    print(f"[Data] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = build_model(cfg, attribute_names, num_users, num_items, kg, device)
    print(f"[Model] STRATA built successfully.")

    run_sft(cfg, model, train_loader, attribute_names, absa_model, device)

    run_rl(cfg, model, train_loader, attribute_names, pmi_weights, kg, absa_model, synonym_map, device)

    run_evaluation(cfg, model, test_loader, attribute_names, absa_model, kg, synonym_map, device)


if __name__ == "__main__":
    main()
