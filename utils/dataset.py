from datasets import load_dataset

from utils.tokenization import format_chat


def load_chat_dataset(train_file: str, test_file: str):
    return load_dataset(
        "json",
        data_files={
            "train": train_file,
            "test": test_file,
        },
    )


def format_chat_dataset(dataset, tokenizer):
    return dataset.map(lambda example: format_chat(example, tokenizer))


def select_demo_samples(dataset, sample_size: int = 50):
    for split in ("train", "test"):
        limit = min(sample_size, len(dataset[split]))
        dataset[split] = dataset[split].select(range(limit))

    return dataset
