import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to(device)

text = "Redis"

encoded = tokenizer(text, return_tensors="pt").to(device)

token_id = encoded["input_ids"][0][0]

print("Text:", text)
print("Token ID:", token_id.item())

embedding_layer = model.get_input_embeddings()

embedding = embedding_layer(token_id)

print("Embedding shape:", embedding.shape)
print("First 10 values:")
print(embedding[:10])