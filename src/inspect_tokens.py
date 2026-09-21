from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Redis is fast."

encoded = tokenizer(text)

print("Original text:")
print(text)

print("\nToken IDs:")
print(encoded["input_ids"])

tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

print("\nTokens:")
print(tokens)

decoded = tokenizer.decode(encoded["input_ids"])

print("\nDecoded back to text:")
print(decoded)