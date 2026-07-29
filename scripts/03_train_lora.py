from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
import yaml
from peft import (
    LoraConfig,
    get_peft_model,
)

from trl import SFTTrainer

# --------------------
# Configuration
# --------------------
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# OUTPUT_DIR = "adapters/qwen-lora"

# MAX_SEQ_LENGTH = 2048

# BATCH_SIZE = 2

# LEARNING_RATE = 2e-4

# EPOCHS = 1

# --------------------
# Load Dataset
# --------------------
print("Loading dataset...")

def load_config(config_path="configs/lora.yaml"):
    """
    Load the YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

config = load_config()

MODEL_NAME = config["model"]["name"]

OUTPUT_DIR = config["training"]["output_dir"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
EPOCHS = config["training"]["epochs"]
MAX_SEQ_LENGTH = config["training"]["max_seq_length"]

print(config)

def format_chat(example):
    """
    Convert a conversation into Qwen's chat format.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/processed/train.jsonl",
        "test": "data/processed/test.jsonl",
    },
)

dataset = dataset.map(format_chat)
print(dataset["train"][0]["text"][:300])

# --------------------
# Load Tokenizer
# --------------------
# print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# print("Tokenizer loaded:", tokenizer.name_or_path)

# --------------------
# Test the tokenizer
# --------------------
sample = dataset["train"][0]

# print("\nFirst sample:")
# print(sample)

formatted_text = tokenizer.apply_chat_template(
    sample["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

# print("\nFormatted Conversation:")
# print(formatted_text)

tokens = tokenizer(
    formatted_text,
    truncation=True,
    max_length=MAX_SEQ_LENGTH,
)

# print("\nToken Count:", len(tokens["input_ids"]))

# print("\nFirst 20 Token IDs:")
# print(tokens["input_ids"][:20])

# print("\nDecoded Text:")
# print(tokenizer.decode(tokens["input_ids"]))
print("\nLoading Qwen Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
)

print("Model loaded successfully!")

print("\nAttaching LoRA...")

lora_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["alpha"],
    lora_dropout=config["lora"]["dropout"],
    bias=config["lora"]["bias"],
    task_type=config["lora"]["task_type"],
)

model = get_peft_model(model, lora_config)

print("LoRA attached successfully!")
model.print_trainable_parameters()