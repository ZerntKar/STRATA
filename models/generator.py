import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


def build_control_prompt(action: List[Tuple[str, str]]) -> str:
    amplify_attrs = [a for a, z in action if z == "Amplify"]
    warning_attrs = [a for a, z in action if z == "Warning"]
    parts = []
    if amplify_attrs:
        parts.append(f"Highlight these strengths: {', '.join(amplify_attrs)}.")
    if warning_attrs:
        parts.append(f"Warn about these concerns using hedging phrases: {', '.join(warning_attrs)}.")
    return " ".join(parts)


class AttributeGroundedGenerator(nn.Module):

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = True,
        max_gen_length: int = 256,
    ):
        super().__init__()
        self.max_gen_length = max_gen_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        self.model = get_peft_model(base_model, lora_config)

    def build_input(self, action: List[Tuple[str, str]], item_facts: str, user_profile: Optional[str] = None) -> str:
        i_sys = "You are a trustworthy recommendation assistant. Generate a balanced explanation that honestly highlights both strengths and potential concerns."
        i_ctrl = build_control_prompt(action)
        user_ctx = f"User profile: {user_profile}\n" if user_profile else ""
        return (
            f"<|system|>\n{i_sys}\n"
            f"<|user|>\n"
            f"{user_ctx}"
            f"Item facts: {item_facts}\n"
            f"Instructions: {i_ctrl}\n"
            f"<|assistant|>\n"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, action: List[Tuple[str, str]], item_facts: str, user_profile: Optional[str] = None, temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = self.build_input(action, item_facts, user_profile)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_gen_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def compute_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        full_mask = torch.cat([attention_mask, torch.ones_like(response_ids)], dim=1)
        outputs = self.model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits
        resp_len = response_ids.shape[1]
        resp_logits = logits[:, -(resp_len + 1):-1, :]
        log_probs = torch.log_softmax(resp_logits, dim=-1)
        token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)
