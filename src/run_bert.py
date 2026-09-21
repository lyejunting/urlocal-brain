from transformers import AutoTokenizer, AutoModelForMaskedLM

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

top_tokens = mask_logits.topk(5, dim=1).indices[0]

for token_id in top_tokens:
    print(tokenizer.decode(token_id))