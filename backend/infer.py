import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import asyncio
import json
import time
import os
from models import GenerationEvent, FinishReason, EventType

MAX_NEW_TOKENS = 1000

resource_path = os.path.join(os.path.expanduser('~'), 'models/SmolLM2-360M-Instruct')
print(resource_path)
device = torch.accelerator.current_accelerator()

print(f"Current Device: {device}")

# Initialise the model
model = AutoModelForCausalLM.from_pretrained(
    resource_path,
    dtype = torch.float16,
    device_map = device,
)

# Initialise the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    resource_path
)


# Define the .generate() method
async def generate(text: str):
    # Initialise the async iterator for streaming tokens
    streamer = TextIteratorStreamer(tokenizer = tokenizer, skip_prompt = True)
    
    inputs = tokenizer(text, return_tensors = "pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tokens = []

    loop = asyncio.get_event_loop()
    is_first = True
    ttft = None
    start_time, t0 = time.perf_counter(), time.perf_counter()

    loop.run_in_executor(
        None,
        lambda: model.generate(
            **inputs,
            streamer = streamer,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample = True,
            return_dict_in_generate = True
        )
    )

    for chunk in streamer:
        t1 = time.perf_counter()
        itl = t1 - t0

        if is_first:
            ttft = itl
            itl = None
            is_first = False

        event = json.dumps(
            GenerationEvent(
                token = chunk,
                itl = itl,
                event_type = EventType.TOKEN
            ).model_dump(exclude_none = True)
        )

        t0 = t1

        tokens.append(chunk)
        yield f"data: {event}\n\n"

    end_time = time.perf_counter()
    total_generation_time = round(end_time - start_time, 2)

    event = json.dumps(
        GenerationEvent(
            prompt_tokens = inputs['input_ids'].shape[1],
            completion_tokens = len(tokens),
            ttft = ttft,
            average_tps = round(len(tokens) / total_generation_time, 2),
            total_generation_time = total_generation_time,
            finish_reason = FinishReason.MAX_TOKEN_REACHED if len(tokens) >= MAX_NEW_TOKENS else FinishReason.EOS,
            event_type = EventType.COMPLETION
        ).model_dump(exclude_unset = True)
    )

    yield f"data: {event}\n\n"

if __name__ == "__main__":
    import torch
    d = torch.accelerator.current_accelerator()
    resource_path = os.path.join(os.path.expanduser('~'), 'models/SmolLM2-360M-Instruct')

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

    from pprint import pprint
    pprint(model.get_memory_footprint()/1e6)


    text = "existence?<|im_end|>"
    inputs = tokenizer(text, return_tensors = "pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
            **inputs,
            # streamer = streamer,
            max_new_tokens = 10,
            do_sample = True,
            return_dict_in_generate = True,
            output_scores = True
        )

    print("Outputs: ", outputs)
    # for t in inputs['input_ids'][0]:
    #     print(t.item(), ' :: ', tokenizer.convert_ids_to_tokens(t.unsqueeze(0))[0])


"""
I have a subtle bug which I don't understand quite well. I have certain query for which one generation event of type `TOKEN` looks like this:
data: {"token": "existence?<|im_end|>", "itl": 0.0005101159913465381, "event_type": "TOKEN"}
It seems like it is a single token. But when I manually call the tokeniser to deconstruct it, it is composed of 3 tokens:
43694  ::  existence
47  ::  ?
2  ::  <|im_end|>

What is going on? can you explain me. 
"""