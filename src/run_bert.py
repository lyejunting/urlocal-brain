from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

MODEL_NAME = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

text = "Redis is a [MASK] database."

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

mask_token_index = (
    inputs["input_ids"][0] == tokenizer.mask_token_id
).nonzero(as_tuple=True)[0]

mask_logits = outputs.logits[0, mask_token_index, :]

probabilities = torch.softmax(mask_logits, dim=-1)

top = probabilities.topk(5, dim=1)

top_probs = top.values[0]
top_token_ids = top.indices[0]

for token_id, probability in zip(top_token_ids, top_probs):
    token = tokenizer.decode(token_id)
    print(f"{token}: {probability.item():.4f}")

for token_id in top_tokens:
    print(tokenizer.decode(token_id))