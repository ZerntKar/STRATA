
# STRATA: Risk-Aware Explainable Recommendation

> **Mitigating Selective Disclosure in Recommender Explanations via Risk-Aware Memory and Hierarchical Planning**

---

## Overview

Existing explainable recommendation systems tend to selectively highlight positive aspects while omitting potential concerns, leading to biased and untrustworthy explanations. **STRATA** addresses this problem through a hierarchical plan-and-generate framework that decouples strategic planning from natural language generation.

STRATA consists of four core components:

- **Perception Layer** — Candidate generation via SASRec + factual attribute retrieval from Wikidata-aligned KG
- **RAPM (Risk-Aware Preference Memory)** — Dual-state user memory tracking preference strength and risk sensitivity at the attribute level
- **Strategic Plan Selector** — Attribute-level Amplify/Warning strategy planning via MLP policy with validity masking
- **Attribute-Grounded Generator** — LoRA-finetuned LLM that converts strategy plans into fluent, fact-anchored explanations

Training follows a two-phase pipeline: supervised warm-up (SFT) followed by cooperative PPO with KL-annealing regularization.

---

## Project Structure

```
strata/
├── config/
│   └── config.py              # Global hyperparameters
├── data/
│   └── dataset.py             # Dataset loading and preprocessing
├── models/
│   ├── perception.py          # SASRec + KG perception layer
│   ├── memory.py              # RAPM dual-state memory
│   ├── selector.py            # Strategic plan selector
│   ├── generator.py           # Attribute-grounded generator (LoRA)
│   └── strata.py              # Full model integration
├── training/
│   ├── rewards.py             # Selector and generator reward functions
│   ├── sft_trainer.py         # Supervised fine-tuning trainer
│   └── ppo_trainer.py         # Cooperative PPO trainer
├── evaluation/
│   └── metrics.py             # N-FCR / F-EHR / P-EHR / BERTScore
├── utils/
│   ├── absa.py                # ABSA sentiment extraction
│   ├── kg_utils.py            # Knowledge graph utilities
│   └── lexicon.py             # ToneMatch / KeyAnchor lexicons
└── main.py                    # Training and evaluation entry point
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version |
|---------|---------|
| torch | ≥ 2.0.0 |
| transformers | ≥ 4.40.0 |
| peft | ≥ 0.10.0 |
| bert-score | ≥ 0.3.13 |
| nltk | ≥ 3.8.1 |

---

## Data Preparation

Place the following files under `data/`:

```
data/
├── train.jsonl          # Training samples
├── valid.jsonl          # Validation samples
├── test.jsonl           # Test samples
├── kg.json              # Item-attribute knowledge graph
├── kg_stats.json        # Category-level attribute statistics
├── attributes.json      # Global attribute name list
├── pmi_weights.json     # PMI-based attribute weights
├── synonym_map.json     # Attribute synonym mappings
├── meta.json            # Dataset metadata (num_users, num_items)
└── sasrec.pt            # Pre-trained SASRec checkpoint
```

Each `.jsonl` sample follows this format:

```json
{
  "user_id": "U001",
  "item_id": "I042",
  "category": "book",
  "history": ["I010", "I021", "I033"],
  "review": "This book has great storytelling but the pacing is slow.",
  "rating": 4.0
}
```

---

## Reward Design

**Selector rewards:**

$$R_{sel} = \lambda_1 \cdot R_{fea} + \lambda_2 \cdot R_{style}$$

- $$R_{fea}$$: Weighted Jaccard similarity between selected and GT attributes
- $$R_{style}$$: Stance-polarity alignment reward

**Generator rewards:**

$$R_{gen} = \lambda_3 \cdot R_{follow} + \lambda_4 \cdot R_{sem} + \lambda_5 \cdot R_{fact}$$

- $$R_{follow}$$: Rule-anchored instruction following (KeyAnchor × ToneMatch)
- $$R_{sem}$$: BERTScore semantic consistency
- $$R_{fact}$$: Factual direction anchoring for Warning-stance attributes

---

## Training Objective

The overall KL-regularized PPO objective:

$$\mathcal{L} = -\mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t, 1\pm\epsilon) A_t\right)\right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

where $$\beta$$ anneals linearly from $$\beta_{init}$$ to $$\beta_{final}$$ over training.
