from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_training_config(config: dict) -> dict:
    training = config["training"]

    training["batch_size"] = int(training["batch_size"])
    training["learning_rate"] = float(training["learning_rate"])
    training["epochs"] = int(training["epochs"])
    training["max_seq_length"] = int(training["max_seq_length"])

    return config
