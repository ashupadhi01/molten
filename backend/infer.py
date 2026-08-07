import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import asyncio
import json
import time
from models import GenerationEvent, FinishReason, EventType

MAX_NEW_TOKENS = 50
resource_path = "/home/ashutosh/models/gpt2"
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
            do_sample = True
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
            finish_reason = FinishReason.EOS  if len(tokens) >= MAX_NEW_TOKENS else FinishReason.MAX_TOKEN_REACHED,
            event_type = EventType.COMPLETION
        ).model_dump(exclude_unset = True)
    )

    yield f"data: {event}\n\n"

if __name__ == "__main__":
    # import torch
    # d = torch.accelerator.current_accelerator()
    # model_path = "/home/molyb/code/test_models/gpt2"
    # device = torch.accelerator.current_accelerator()

    # print(f"Current Device: {device}")

    # # Initialise the model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     dtype = torch.float16,
    #     device_map = device
    # )

    # # Initialise the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path
    # )
    
    # from pprint import pprint
    # pprint(model.get_memory_footprint()/1e6)


    import asyncio

    async def main():
        query = "Given a scenario when you are in perfect danger. What would you do?"
        async for data in generate(query):
            print(data)

    asyncio.run(main())