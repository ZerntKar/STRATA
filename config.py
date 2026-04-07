from dataclasses import dataclass


@dataclass
class STRATAConfig:
    dataset: str = "book"
    data_dir: str = "data/"
    max_history_len: int = 10
    top_k_candidates: int = 20

    rec_model_name: str = "sasrec"
    llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    embedding_dim: int = 64
    hidden_dim: int = 256

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    eta_pos: float = 0.1
    eta_neg: float = 1.0
    eta_decay: float = 0.2
    rho_absa: float = 0.75
    tau: float = 5.0
    epsilon_weak: float = 1e-6

    num_attributes: int = 20
    attribute_embed_dim: int = 64
    selector_hidden_dim: int = 128

    lambda1: float = 0.5
    lambda2: float = 0.5
    lambda3: float = 0.4
    lambda4: float = 0.4
    lambda5: float = 0.2

    beta_kl_init: float = 0.1
    beta_kl_final: float = 0.01
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 3
    alternating_rounds: int = 3

    batch_size: int = 8
    learning_rate: float = 1e-5
    sft_epochs: int = 5
    rl_epochs: int = 3
    warmup_steps: int = 100
    max_gen_length: int = 256
    seed: int = 42

    device: str = "cuda"
    fp16: bool = True
    load_in_8bit: bool = True
