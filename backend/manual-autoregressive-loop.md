# Milestone: Manual Autoregressive Loop (per-token streaming)

## Goal

Replace `TextIteratorStreamer` + `model.generate(..., streamer=...)` with a hand-written autoregressive loop so that:

- each generated token becomes its own `TOKEN` event,
- `itl` (inter-token latency) and `ttft` (time-to-first-token) are real measurements,
- `completion_tokens` reflects the true token count (not the number of streamer chunks),
- `<|im_end|>` no longer leaks into the output text.

Later this can be refactored into a reusable `TokenStreamer` class that adapts to different models. For now: learn by doing.

---

## Why we're doing this (the bug this fixes)

`TextIteratorStreamer` inherits from `TextStreamer`. Its `put()` does **not** emit one token at a time. On each new token it:

1. appends the new id to `token_cache` (the whole generated sequence so far),
2. re-decodes the **entire** cache: `text = tokenizer.decode(token_cache)`,
3. releases text only up to the **last space**: `text[self.print_len : text.rfind(" ") + 1]`.

If there is no space, `rfind(" ")` returns `-1`, the slice becomes `""`, and the token is **buffered silently**. The cache is only flushed on a trailing `"\n"`, a trailing CJK char, a space, or finally `end()` at generation stop.

Observed example with SmolLM2-360M-Instruct:

```
43694 :: 'existence'
47    :: '?'
2     :: '<|im_end|>'
```

None of these contain a space, so all three were buffered and flushed as a single chunk `"existence?<|im_end|>"` by `end()`. That single chunk became one `TOKEN` event — even though three forward passes produced it. The `itl` on that event (`0.0005s`) was just the gap to the final flush, not a real inter-token latency.

Reproduction confirmed:

```
cumulative decode (what the streamer sees):
  [43694]        :: 'existence'
  [43694, 47]    :: 'existence?'
  [43694, 47, 2] :: 'existence?<|im_end|>'
rfind(space) on final: -1
contains space? False
```

Takeaway: `TextIteratorStreamer` is a **text** streamer, not a **token** streamer. It deliberately coalesces tokens until a word boundary.

---

## Learning path — concepts in order

### 0. Read what `model.generate` hides

Open `transformers/generation/utils.py` and follow `generate` → `_sample` (the branch taken when `do_sample=True`). You don't need all of it. Just notice the skeleton:

forward pass → take last-position logits → apply warpers (temperature/top-k/top-p) → sample → append → repeat, with `past_key_values` threaded through.

Your loop will be a stripped-down version of this.

### 1. The forward pass and what it returns

Call the model directly:

```
outputs = model(input_ids=..., attention_mask=..., use_cache=True)
```

`outputs` is a `CausalLMOutput`. Two things matter:

- `outputs.logits` — what's its shape? Which axis is the sequence, which is the vocab? You only ever need **one** position's logits per step — which one, and why?
- `outputs.past_key_values` — this is the KV cache. What happens if you don't pass it next step? What happens if you do?

### 2. The two-phase loop structure (the key insight)

- **Step 1 (prefill):** feed the **entire prompt**. You get logits for every prompt position, but you only use the last. You also get a `past_key_values` representing all that work.
- **Steps 2..N (decode):** feed **only the one token you just generated**, plus the `past_key_values` from before. The model only computes for that single new position.

That's why streaming is cheap after prefill — and it's exactly what gives you a meaningful per-token `itl`.

Get this shape right: prompt in once, then one token at a time. If you accidentally re-feed the whole sequence each step, it'll still work but be O(n²) and your `itl` will be meaningless.

### 3. Sampling (you chose `do_sample=True`)

`generate` doesn't just take `argmax`; it samples. Replicate the minimum:

- What does `temperature` do to logits? (default 1.0 — your current runs are basically sampling from the raw distribution)
- How do you turn logits into a probability distribution, then draw one id? Think `softmax` → `multinomial`.
- Optional later: top-k / top-p. Skip them first.

**Start with greedy (`argmax`) even though your goal is sampling.** Confirm the loop produces sane text, then swap in sampling. Don't fight two bugs at once — greedy is deterministic and far easier to debug.

### 4. Stopping conditions

Two ways generation ends:

- you hit `MAX_NEW_TOKENS`,
- the model emits a stop token.

For SmolLM2-Instruct the stop token is `<|im_end|>` (id `2`, the one observed). Check `tokenizer.eos_token_id` and also look at the model's `generation_config` — there may be multiple `eos_token_id`s. Decide: do you stop on *any* of them?

Keep the existing `FinishReason.EOS` vs `MAX_TOKEN_REACHED` distinction — just base it on the real id you observed, not on `len(tokens)`.

### 5. Per-token decoding

Each step you have exactly one new id. Decode **just that id** with `skip_special_tokens=True`. Notice what you get for `<|im_end|>`: an empty string. That's your cue to (a) stop generation and (b) not emit an empty `TOKEN` event.

This is the whole reason the original bug existed — `TextStreamer` decoded the *accumulated* cache; you'll decode *one id at a time*.

### 6. Timing

Now that each iteration is genuinely one token, `itl` and `ttft` become real:

- `ttft` = time from start to just before you yield the first token (dominated by the prefill forward pass),
- `itl` = gap between consecutive token yields (dominated by each decode forward pass).

Think about *where* to put the `perf_counter` calls so they measure the model work, not your Python overhead.

### 7. The async/threading question (design fork — decide before coding)

Your current code runs `model.generate` in a thread via `run_in_executor` and reads from a queue, because `generate` is blocking and you're in an `async` function. With a manual loop you have a choice:

- **Keep it in an executor:** run your whole loop in a thread, push each decoded token to an `asyncio.Queue`, and `await queue.get()` in the async generator. Same architecture as now, you just replaced `generate` with your loop.
- **Run each forward pass with `asyncio.to_thread`:** the loop itself is async, and each blocking `model(...)` call is offloaded. Cleaner, but you pay a thread hop per token.

Pick one and be consistent. The first is closer to what you have; the second is more "async-native." Either is fine — decide deliberately, not accidentally.

---

## Suggested order of work

1. Open a **scratch file** (not `infer.py` yet).
2. Do steps 1–3 **synchronously, greedily, no streaming, no async** — just a `for` loop that prints each decoded token as it's generated. Get that producing correct text for a fixed prompt.
3. Then add sampling.
4. Then add EOS handling.
5. Then port it into the async `generate()` with timing.

Each layer on top of a working foundation.

---

## Common pitfalls to watch for

- **Wrong logits position / wrong "feed one token + past_key_values" shape.** If your first token is garbage but the rest is fine, or vice versa, this is almost certainly it. Print shapes liberally until you trust the loop.
- **Re-feeding the whole sequence each step** — works but O(n²) and meaningless `itl`.
- **Forgetting `skip_special_tokens=True`** — `<|im_end|>` leaks into text.
- **Emitting an empty `TOKEN` event for the EOS token** — decode of `<|im_end|>` is `""`; skip it.
- **Counting chunks instead of ids** — `completion_tokens` must come from the real generated id count, not from `len(tokens)`.

---

## Reference: the original (buggy) streaming setup

Located in `infer.py`, `generate()`:

- `streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)`
- `loop.run_in_executor(None, lambda: model.generate(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, do_sample=True))`
- `for chunk in streamer:` yields `TOKEN` events with `itl`
- `completion_tokens = len(tokens)` ← undercounts due to buffering
- `finish_reason` based on `len(tokens) >= MAX_NEW_TOKENS` ← wrong basis

The manual loop replaces the `run_in_executor` + `for chunk in streamer` block. Everything around it (the `GenerationEvent` schema, the SSE `yield f"data: {event}\n\n"`, the final `COMPLETION` event) can stay structurally the same — just feed it real per-token data.

---

## Environment reference

- transformers version: 5.14.1
- streamer source: `.venv/lib/python3.14/site-packages/transformers/generation/streamers.py` (`TextStreamer.put` is the heuristic; `TextIteratorStreamer.on_finalized_text` pushes to a `Queue`)
- generate internals: `.venv/lib/python3.14/site-packages/transformers/generation/utils.py` (`generate` → `_sample`)
- model: `~/models/SmolLM2-360M-Instruct`, dtype float16, on `torch.accelerator.current_accelerator()`
- observed token ids for the bug: `43694` (`existence`), `47` (`?`), `2` (`<|im_end|>`)
