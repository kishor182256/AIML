import argparse
import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare UltraChat SFT data as local JSONL files."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples per split. Default 0 writes the full dataset.",
    )

    return parser.parse_args()


def limit_split(split, max_samples: int):
    if max_samples <= 0:
        return split

    return split.select(range(min(max_samples, len(split))))


def save_jsonl(split, output_file: Path):
    with output_file.open("w", encoding="utf-8") as file:
        for sample in split:
            record = {"messages": sample["messages"]}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(split):,} examples to {output_file}")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HuggingFaceH4/ultrachat_200k...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

    train = limit_split(dataset["train_sft"], args.max_samples)
    test = limit_split(dataset["test_sft"], args.max_samples)

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    print("Done.")


if __name__ == "__main__":
    main()
