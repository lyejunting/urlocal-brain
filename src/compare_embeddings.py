import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to(device)

embedding_layer = model.get_input_embeddings()


def get_embedding(text):
    encoded = tokenizer(text, return_tensors="pt").to(device)

    token_id = encoded["input_ids"][0][0]

    return embedding_layer(token_id)


redis = get_embedding("Redis")
memcached = get_embedding("Memcached")
banana = get_embedding("Banana")


def similarity(a, b):
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0)
    ).item()


print("Redis vs Memcached:", similarity(redis, memcached))
print("Redis vs Banana:", similarity(redis, banana))