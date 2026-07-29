from pathlib import Path
import sys

from transformers import TrainingArguments
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import load_config, normalize_training_config
from utils.dataset import format_chat_dataset, load_chat_dataset
from utils.modeling import attach_lora, load_causal_lm
from utils.tokenization import load_tokenizer, tokenize_text


CONFIG_PATH = PROJECT_ROOT / "configs" / "lora.yaml"


def main():
    config = normalize_training_config(load_config(CONFIG_PATH))

    model_name = config["model"]["name"]
    output_dir = config["training"]["output_dir"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    max_seq_length = config["training"]["max_seq_length"]
    train_file = str(PROJECT_ROOT / config["dataset"]["train"])
    test_file = str(PROJECT_ROOT / config["dataset"]["test"])

    print("Configuration loaded.")
    print(config)

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(model_name)

    print("\nLoading dataset...")
    dataset = load_chat_dataset(train_file, test_file)
    dataset = format_chat_dataset(dataset, tokenizer)
    print(f"Train samples: {len(dataset['train']):,}")
    print(f"Test samples: {len(dataset['test']):,}")

    print("\nFirst formatted sample:")
    print(dataset["train"][0]["text"][:300])

    sample = dataset["train"][0]
    tokens = tokenize_text(
        sample["text"],
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    print("\nSample token count:", len(tokens["input_ids"]))

    print("\nLoading model...")
    model = load_causal_lm(model_name)
    print("Model loaded successfully.")

    print("\nAttaching LoRA...")
    model = attach_lora(model, config["lora"])
    print("LoRA attached successfully.")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
