# NOTES

## KV Cache & Memory Growth (2026-08-13)

**Symptom:** During autoregressive generation with `SmolLM2-360M-Instruct`, process USS (from `/proc/<pid>/smaps_rollup`) grew ~224 MB over a 50-token generation, while the live KV cache was only ~2 MB and the cumulative-sum-of-KV metric was ~50 MB. Growth was non-linear and only triggered when output seq length exceeded any previous run.

**Root cause: autograd was on during inference.**
- The `model(...)` forward was called with no `torch.inference_mode()` / `torch.no_grad()`.
- Because model parameters have `requires_grad=True`, every op saves tensors for a backward that never comes.
- `past_key_values` fed back into the next forward chains the new graph onto the previous one, so the *entire generation's* graph stays alive until `generate()` returns.

**Why the cumulative-sum metric (~50 MB) matched but undercounted:**
- `DynamicCache.update()` does `torch.cat([old, new])` each step; `cat`'s backward saves both inputs.
- This forms a chain: step *n*'s cat holds step *n−1*'s cat output, which holds *n−2*'s, ... so all cat outputs (sizes 1..50) stay alive.
- Sum of cat-output sizes ≈ cumulative KV sum (coincidence of the cat pattern, not a general rule).
- Slight undercount: misses the per-step size-1 `new_kv` tensors the cats also save.

**Remaining ~170 MB gap:** other per-layer activations saved across all 50 steps × 32 layers (Q/K/V-proj inputs, MLP gate/up/down, attention weights, residuals, logits `[1,1,49152]`, etc.) — invisible to the KV metric.

**The "no growth under previous max" quirk:** allocator behavior, not model behavior. Freed graph tensors go into PyTorch/glibc caching pools (USS counts private pages, which stay mapped). Blocks are reused for same-sized future allocations; only seq lengths exceeding the prior peak force fresh OS allocation.

**Fix:** decorate `generate()` with `torch.inference_mode()`. Diff collapsed from ~224 MB to roughly KV-live size. `inference_mode` (modern, faster) vs `no_grad` (older, more permissive) — either works for pure inference.

**Lesson:** Inference without an inference-mode context silently builds and retains the autograd graph. Always wrap forward calls in `torch.inference_mode()` (or `no_grad()`) for serving/inference paths.

---

## Concurrency, Async & the GIL (2026-08-14)

**Question:** Can `.generate()` be made "truly async" to serve concurrent requests? And is `self.kv_cache_size` (a class attribute on the singleton `CustomGenerator`) safe across requests?

**Async is a model for *waiting*, not *doing*.** Sprinkling `async`/`await` on CPU-bound code doesn't make it concurrent — the event loop only gains control at an `await` where something actually suspends (I/O, a future, a threadpool offload). The generation loop is dominated by the forward pass; making the bookkeeping (sampling, attention-mask update, EOS check) async is pure ceremony — there's nothing to wait on.

**The only lever for concurrency is `run_in_executor` (offload the forward pass to a threadpool).** Everything else is microseconds.

**The GIL does NOT serialize torch compute.** The GIL gates Python bytecode, but torch's heavy ops (matmul, attention) are C extensions that **release the GIL** while running. So two Python threads can run two forwards genuinely in parallel on two cores. The GIL only serializes the Python glue *between* ops.

**Two distinct thread layers — don't conflate them:**
- **Python executor threads** (from `run_in_executor`): one per offloaded request, long-lived, *own* the request's activations + KV cache in their call frame.
- **OpenMP worker threads**: spawned *inside* a single torch op to parallelize it across cores, die when the op returns. They don't own request state — they're ephemeral compute workers.

**The shared model is safe during inference because it's read-only.** Weights are shared but never mutated (no gradients, no in-place updates) → concurrent reads are fine. Per-call state (activations, KV) lives in the caller's frame, not on the model. A mid-forward thread switch just pauses the frame (locals intact, tensors still referenced); nothing model-specific is saved/restored. The mental model of "threads take turns using the model and its state must be saved" is wrong — the model is a pure function (weights in → activations out) during inference.

**The one real shared-mutable offender is `self.kv_cache_size`.** Unlike weights (shared read-only) and activations (per-call), the instance attribute is shared mutable state. With offloading, two threads write it → last-writer-wins → wrong stat read in the `COMPLETION` event. Correct *today* only because we don't offload; brittle by design. Fix: compute the stat in `infer.py` from per-call data (`final_seq_len` + architecture constants), not via shared instance state.

**KV cache size is O(1) analytic, no tensor loop needed.** `size = 2 · L · h_kv · d · s · b_dtype` (layers · KV heads · head dim · seq len · bytes/dtype). Computing it per-token via a tensor loop pollutes ITL for no benefit. For a generation-level stat, compute once at the end.

**The OS thread layer is request-agnostic.** Both Python executor threads and OpenMP workers are just threads to the OS (CFS on Linux schedules them fairly, preemptively, with no concept of "request" or "generation step"). That semantic mapping lives entirely at the application layer. This is why production serving engines build their own scheduler on top — the OS can't prioritize a short request over a long one, can't detect starvation, can't batch.

**Intra-op vs inter-request parallelism — same at the OS layer, asymmetric above it:**
- **Intra-op (A):** N OpenMP workers cooperate on one op, barrier-sync, no GIL between them → efficient, balanced, but serves one request at a time (low latency, zero concurrency).
- **Inter-request (B):** N Python threads run N requests, each op slow → high concurrency, worse per-request latency. At op boundaries they serialize on the GIL for Python glue.
- OS sees both as "N threads on N cores" — equivalent. But (A) optimizes latency, (B) optimizes concurrency. Not interchangeable.

**The real lever is batching, and neither (A) nor (B) reaches it.** Batch-1 decode is **memory-bandwidth bound**, not compute-bound — you load full weight matrices to produce one token. Adding cores (A) barely helps decode (memory-bound); (B) has N requests each hammering the same bandwidth, so no N× throughput. Batching reads weights *once* for N requests → N tokens for one memory pass. That's the move that actually changes throughput. Prefill is more compute-bound, so (A) helps there; decode doesn't benefit from either parallelism strategy.

**Lesson:** For a single local model, async-ifying the bookkeeping is ceremony, offloading the forward pass has GIL-release-enabled overlap but trades memory for concurrency, and the throughput lever that dominates both is batching. Keep per-request stats out of shared instance state.

---

## asyncio Concurrency Primitives — `create_task` vs `to_thread` vs `run_in_executor` (2026-08-16)

**Question:** What's the real contract of `asyncio.create_task()`, `asyncio.to_thread()`, and `loop.run_in_executor()` — and why do they seem to take/return different things?

**The first principle: what each API takes as input.**
- `create_task(coro)` takes a **coroutine object** — the *result* of calling an `async def` (`f()`, not `f`). A coroutine object only exists *after* you call the async function; the function itself is not a coroutine.
- `to_thread(func)` takes a **callable** — the sync function itself (`f`, not `f()`). It does the calling itself, inside the worker thread.
- `run_in_executor(executor, func)` — the low-level primitive `to_thread` wraps — also takes a callable.

**The failure modes are symmetric and reveal the contract.**
- `create_task(f)` → `TypeError: a coroutine was expected` (you passed a function, not a coroutine).
- `to_thread(f())` → runs `f` *now in the current thread*, passes its return value (e.g. `None`) → `TypeError: 'NoneType' object is not callable` (you passed a result, not a callable).
- Both errors are the same mistake: passing the wrong *kind* of thing. The API tells you exactly what it wants.

**The unifying rule for inputs: who does the calling?**
- `create_task` → **you** call `f()` to produce the coroutine; the loop schedules it.
- `to_thread` / `run_in_executor` → **the thread pool** calls `f`; you hand it the recipe.

**The second principle: what each returns, and the laziness distinction.**
- `to_thread(f)` returns a **coroutine** — *lazy*. It does nothing until awaited. Without `await` → `RuntimeWarning: coroutine was never awaited` (a leak detector: you created an awaitable and discarded it).
- `create_task(f())` returns a **`Task`** — *eagerly scheduled*. Creating the task immediately schedules it on the loop, so it runs on its own. `await` is only needed to *get the result* or wait for completion, not to start it.
- `run_in_executor(...)` returns a **`Future`** — a passive result container, *not* scheduled on the loop as a coroutine. The thread fills it in; the loop just watches it.

**`Future` vs `Task` vs coroutine — the object hierarchy.**
- A **coroutine** is a recipe (lazy; runs only when awaited/scheduled).
- A **`Task`** is a coroutine + scheduling (a `Future` subclass that also drives a coroutine).
- A **`Future`** is a result holder (doesn't run code; something else fills it).
- All three are **awaitable**. `await` parks the current coroutine and resumes it when the awaitable resolves — it blocks *this coroutine*, not the loop (other tasks keep running).

**`to_thread` is a thin wrapper over `run_in_executor`.**
```python
async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
```
So the stack is: `to_thread` → `run_in_executor` → `ThreadPoolExecutor.submit` → `threading.Thread` → OS thread. `to_thread` is just `run_in_executor` with the `await` baked in — which is *why* it must be awaited.

**`await` blocks the coroutine, not the loop.** `await future` suspends the current coroutine until the thread finishes; the loop stays alive to run other tasks. This is the cooperative model: `await` yields control back to the loop. Contrast with `time.sleep()` inside a coroutine, which freezes *everything*.

**Fire-and-forget patterns and their costs.**
- `create_task(to_thread(f))` — wraps the `to_thread` coroutine in a Task; control returns immediately, thread runs in background. Cost: one extra parked coroutine on the loop (a watcher that just awaits the Future).
- `run_in_executor(None, f)` — returns a Future you can ignore; no coroutine parked on the loop at all. The raw form, one layer down.
- `to_thread` alone is a *join* primitive (it awaits), not a *spawn* primitive — wrong tool for a daemon.

**`ThreadPoolExecutor` is lazy; `max_workers` is a cap, not a pre-allocation.** The pool spawns threads on demand, only as many as concurrent tasks require, up to the ceiling. Submitting 1 task to a `max_workers=5` pool → 1 OS thread. Submitting 5 at once → 5 threads. Verified via `pstree -ap <pid>`.

**The pool is the unit of thread management.**
- `run_in_executor(None, ...)` → always the loop's **one shared default pool**.
- `run_in_executor(custom_ex, ...)` → that executor's pool; same executor → reused threads; different executors → **separate** thread sets.
- This lets you partition threading: dedicate pools to different workloads (different `max_workers`, isolation so a stuck task in one pool can't starve another).

**Loop-in-a-thread is possible but coroutines are loop-bound.** You can run `asyncio.run(inner())` inside a `to_thread` worker to get a per-thread event loop. The constraint: a coroutine/Task/Future is bound to the loop that created it — you cannot `await` a coroutine from a different loop (`RuntimeError: attached to a different loop`). Coroutines must be created *inside* the thread's loop, not passed in from outside.

**Lesson:** The asyncio concurrency surface looks confusing because three APIs (`create_task`, `to_thread`, `run_in_executor`) seem to do similar things. They're actually a clean stack: `create_task` schedules a coroutine on the loop (cooperative, single-threaded); `to_thread`/`run_in_executor` offload a callable to an OS thread (parallel, preemptive). The input contract (coroutine object vs callable) and the output contract (lazy coroutine vs eager Task vs passive Future) fall out directly from *who does the calling* and *where the work runs*. Pools are lazy and partitionable. `await` blocks the coroutine, never the loop.

---
