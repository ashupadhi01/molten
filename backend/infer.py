import os
import time
import json
import asyncio
from asyncio import Queue
from models import GenerationConfig, GenerationEvent, EventType, FinishReason
from runtime import generator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop


executor = ThreadPoolExecutor(max_workers = 24)

def _stream_response(
    loop: AbstractEventLoop,
    queue: Queue,
    text: str,
    generation_config: GenerationConfig
):
    tokens, is_first, ttft = [], True, None
    start_time = t0 = time.perf_counter()
    
    for token in generator.generate(text, generation_config):
        t1 = time.perf_counter()
        itl = t1 - t0

        if is_first:
            ttft = itl
            itl = None
            is_first = False

        event = json.dumps(
            GenerationEvent(
                token = token,
                itl = itl,
                event_type = EventType.TOKEN
            ).model_dump(exclude_none = True)
        )

        t0 = t1

        tokens.append(token)
        loop.call_soon_threadsafe(queue.put_nowait, event)
    
    end_time = time.perf_counter()
    total_generation_time = round(end_time - start_time, 2)

    event = json.dumps(
        GenerationEvent(
            prompt_tokens = generator.count_prompt_tokens(text),
            completion_tokens = len(tokens),
            ttft = ttft,
            average_tps = round(len(tokens) / total_generation_time, 2),
            total_generation_time = total_generation_time,
            finish_reason = FinishReason.MAX_TOKEN_REACHED if len(tokens) >= generation_config.max_new_tokens else FinishReason.EOS,
            event_type = EventType.COMPLETION
        ).model_dump(exclude_unset = True)
    )

    loop.call_soon_threadsafe(queue.put_nowait, event)
    loop.call_soon_threadsafe(queue.put_nowait, None)


async def stream_response(
    text: str,
    generation_config: GenerationConfig
):
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    future = loop.run_in_executor(
        executor,
        _stream_response,
        loop,
        queue,
        text,
        generation_config
    )

    while True:
        event = await queue.get()

        if event is None:
            break

        yield f"data: {event}\n\n"
