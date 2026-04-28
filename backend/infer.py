import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import asyncio

resource_path = "/home/molyb/code/test_models/gpt2"
device = torch.accelerator.current_accelerator()

print(f"Current Device: {device}")

# Initialise the model
model = AutoModelForCausalLM.from_pretrained(
    resource_path,
    dtype = torch.float16,
    device_map = device
)

# Initialise the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    resource_path
)


# Define the .generate() method
async def generate(text: str):
    # Initialise the async iterator for streaming tokens
    streamer = TextIteratorStreamer(tokenizer = tokenizer)
    
    inputs = tokenizer(text, return_tensors = "pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    loop = asyncio.get_event_loop()

    loop.run_in_executor(
        None,
        lambda: model.generate(
            **inputs,
            streamer = streamer,
            max_new_tokens = 1024,
            do_sample = True

        )
    )
    
    for chunk in streamer:
        yield chunk


if __name__ == "__main__":
    import torch
    d = torch.accelerator.current_accelerator()
    model_path = "/home/molyb/code/test_models/gpt2"
    device = torch.accelerator.current_accelerator()

    print(f"Current Device: {device}")

    # Initialise the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype = torch.float16,
        device_map = device
    )

    # Initialise the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    
    from pprint import pprint
    pprint(model.get_memory_footprint()/1e6)