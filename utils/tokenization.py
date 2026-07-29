from transformers import AutoTokenizer


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def format_chat(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def tokenize_text(text: str, tokenizer, max_seq_length: int) -> dict:
    return tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
    )
