from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {"role": "user", "content": "What is Artificial Intelligence?"},
    {"role": "assistant", "content": "Artificial Intelligence is the simulation of human intelligence by machines."}
]

# Apply Qwen chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print("=" * 80)
print("Formatted Prompt")
print("=" * 80)
print(text)

tokens = tokenizer(text)

print("\nNumber of tokens:", len(tokens["input_ids"]))

print("\nFirst 30 token IDs:")
print(tokens["input_ids"][:30])

print("\nDecoded back:")
print(tokenizer.decode(tokens["input_ids"]))