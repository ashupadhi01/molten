from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

MAX_NEW_TOKENS = 1000

resource_path = os.path.join(os.path.expanduser('~'), 'models/gpt2')
print(resource_path)


# Initialise the model
model = AutoModelForCausalLM.from_pretrained(
    resource_path,
    dtype = torch.float16,
)

# Initialise the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    resource_path
)

text = "This is a sample text"
inputs = tokenizer(text, return_tensors = "pt")
print("Shape of tokenized prompt tensor: ", inputs)

# output = model.generate(**inputs, max_new_tokens = 10)


# calling the model class as a `callable` without using .generate() construct

output = model(
    input_ids = inputs["input_ids"],
    attention_mask = inputs["attention_mask"]
)
print(output.logits.shape, type(output))

print("Input sequence KV caches: ", output.past_key_values)