from datasets import load_dataset

print("Downloading UltraChat 200K...")

dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

print(dataset)

print("\nTrain SFT:", len(dataset["train_sft"]))
print("Test SFT :", len(dataset["test_sft"]))

print("\nColumns:")
print(dataset["train_sft"].column_names)

print("\nFirst Example:")
print(dataset["train_sft"][0])