# NOTE — Project Insights Log

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
