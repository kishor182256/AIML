from pathlib import Path

from datasets import load_dataset

from utils.tokenization import format_chat


def load_chat_dataset(train_file: str, test_file: str):
    missing_files = [
        str(path)
        for path in (Path(train_file), Path(test_file))
        if not path.exists()
    ]

    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Processed dataset files are missing:\n"
            f"{missing}\n\n"
            "Create the full UltraChat processed dataset with:\n"
            "python scripts/02_prepare_dataset.py"
        )

    return load_dataset(
        "json",
        data_files={
            "train": train_file,
            "test": test_file,
        },
    )


def format_chat_dataset(dataset, tokenizer):
    return dataset.map(lambda example: format_chat(example, tokenizer))

