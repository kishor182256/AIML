from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def load_causal_lm(model_name: str):
    return AutoModelForCausalLM.from_pretrained(model_name)


def attach_lora(model, lora_config: dict):
    config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )

    return get_peft_model(model, config)
