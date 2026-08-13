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
