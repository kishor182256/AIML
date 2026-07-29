import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

train = dataset["train_sft"]
test = dataset["test_sft"]


def save_jsonl(split, filename):
    output_file = OUTPUT_DIR / filename

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in split:
            record = {
                "messages": sample["messages"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(split):,} examples to {output_file}")


save_jsonl(train, "train.jsonl")
save_jsonl(test, "test.jsonl")

print("Done!")