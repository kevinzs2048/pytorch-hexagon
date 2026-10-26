# KV cache: upstream example, downstream extension and remaining gap

## What the public example demonstrates

We inspected the official [GPT-2 Python example at public commit 9ec15e8](https://github.com/qualcomm/hexagon-mlir/blob/9ec15e80d9d53aa7fb97654da00bfd7047a77ead/test/python/torch-mlir/gpt2lmheadmodel.py). It sets `config.n_layer = 2`, exports a model from `input_ids`, executes one compiled forward, and compares the top-five last-position predictions with the CPU result.

The example does not pass previous K/V tensors into a subsequent model call, expose cache positions as runtime inputs, or demonstrate a `generate()` loop that reuses KV. A Hugging Face forward may internally produce cache outputs; the example does not feed those outputs into another decode step. This describes the example's coverage, not a claim that every Hexagon-MLIR operator or internal implementation lacks KV support.

## What we added

We retained Transformers' native `StaticCache` and `generate()` semantics. We did not invent a replacement caching algorithm. The extension is the compilation and ARM/DSP integration needed to execute cache-bearing forwards correctly:

| Area | Downstream implementation |
| --- | --- |
| Model | Complete Qwen3-0.6B, 28 decoder layers |
| Logical cache | 28 K tensors and 28 V tensors, managed by native Transformers on ARM |
| Physical cache shape | Each tensor is FP32 `[1, 8, 1024, 128]`; total KV storage 224 MiB |
| Forward inputs | Current tokens, positions, masks and all existing K/V tensors are runtime values |
| Updates | DSP computes the current forward and indexed KV updates; cache positions identify the entries to update |
| Compiler integration | AOT functionalizes mutations; downstream TmTensor scatter support covers indexed updates; the backend returns the resulting tensors for framework mutation/writeback handling |
| Generation | Chunked query=2–32 prefill and query=1 decode reuse compiled artifacts; next-token selection and stopping remain native framework behavior |
| Request boundaries | Native cache reset is retained; tests verify old-request KV does not leak into a new request |
| Validation | Same-input logits and all 56 K/V tensors, previous/future positions, capacity boundaries and request-to-request state |

The native cache setup uses `_get_cache(cache_implementation='static', batch_size=1, max_cache_len=1024, ...)` and native cache initialization. The captured backend validates and packs runtime tensors, executes the DSP library, reconstructs the tensor outputs and returns them to AOT/framework handling. Reusing the compiled artifact across prompts does not mean reusing another prompt's KV contents.

## Current data flow

```text
ARM native StaticCache (all 56 K/V tensors)
    -> complete KV + current tokens/positions/mask
DSP compiled model forward and indexed KV updates
    -> all 56 updated full-capacity K/V tensors + logits
ARM framework cache writeback -> select next token -> next forward
```

The model is not recomputing the old tokens' K/V projections at every step; it reuses the cached values. However, the current backend still transfers the complete physical cache across the boundary each step. Persistent transport buffers do not make the logical model KV state DSP-resident.

## What remains and what we need from the team

For one decoded token, the newly added K/V data across all layers is 224 KiB. The existing interface nevertheless passes the full 224 MiB KV state in each direction, about 448 MiB of KV traffic per forward, plus logits and other inputs. The output packet is 235,488,768 bytes. Tracing also observes 56 allocations of 4 MiB per decode forward and the corresponding large copies.

Our delivered extension therefore provides correct stateful generation through a tensor input/output interface. It does not yet provide DSP-resident KV or a delta-update interface.

We need assistance aligning frontend functionalization, compiler bufferization/aliasing and runtime buffer ownership so that KV can be reused or updated safely on DSP. With persistent state, the boundary could be designed around tokens/positions, logits and necessary state updates, while maintaining native Transformers semantics. Such an interface requires correctness and lifetime validation; a transfer reduction or speedup has not been demonstrated yet.
