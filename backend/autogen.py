from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

MAX_NEW_TOKENS = 5

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

def print_token_logits(input_ids: torch.Tensor, logits: torch.Tensor, num_tokens: int):
    print("-" * 50)
    for i in range(num_tokens):
        print(f"{tokenizer.convert_ids_to_tokens(input_ids[0][i].unsqueeze(0))[0]}: {logits[0][i][:3].tolist()}...")
    print("-" * 50)


print_token_logits(inputs["input_ids"], output.logits, inputs["input_ids"].shape[1])

for _ in range(MAX_NEW_TOKENS):
    output = model(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"]
    )

    token_id = output.logits[0][-1].argmax()
    token = tokenizer.convert_ids_to_tokens(token_id.unsqueeze(0))[0]

    # add the new token_id to input sequence and extend the attention mask
    inputs["input_ids"] = torch.cat([inputs["input_ids"], token_id.unsqueeze(0).unsqueeze(0)], dim = 1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.tensor([[1]])], dim = 1)

    print(token, end = "", flush = True)


    # print(f"New Input IDs: {inputs["input_ids"]}")
    # print(f"New attention max: {inputs["attention_mask"]}")


# print("Input sequence KV caches: ", output.past_key_values)