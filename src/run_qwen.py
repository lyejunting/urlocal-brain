import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(device)

messages = [
    {
        "role": "user",
        "content": "Explain Redis in one short paragraph."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
)

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("\nAnswer:\n")
print(answer)