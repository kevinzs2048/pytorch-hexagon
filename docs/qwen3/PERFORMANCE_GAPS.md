# Runtime gaps requiring compiler and runtime support

We have extended the PyTorch examples to complete Qwen3-0.6B inference on QCS9075, with native Transformers generation and reusable bounded-dynamic prefill/decode. Numerical results and generation behavior are validated within the pinned downstream configuration. The next requirement is efficient inference execution.

The maintainer email now includes absolute latency and the named INT8 callee table. See [operator and interval timings](OPERATOR_TIMINGS.md) and [the precise KV baseline, extension and remaining gap](KV_CACHE_EXTENSION.md).

## Measurement boundary

On 2026-10-26 we ran two native generation sessions using the accepted downstream DSP libraries. Each completed 16 model forwards and executed zero torch-mlir frontend or model machine-code compilations. Compilation, imports, model loading, initial graph capture, first library loading and the first decode transition were excluded from the steady decode result. Twenty-six decode intervals remained. The timing runs did not execute an inline CPU reference.

We then replayed identical, previously ARM-qualified full-model input packets through the current accepted libraries and instrumented variants, in baseline/instrumented/instrumented/baseline order. Four diagnostic groups measured decode external calls, prefill external calls, attention boundaries and the local prefill guard. All full logits and 56 KV outputs matched the accepted output hashes. The original model primary objects, constants, mathematical callees and power-vote configuration were retained; only diagnostic wrappers were compiled and linked.

These observations apply to Qwen3-0.6B, batch=1, W8A8 linear layers, FP32 intermediates/KV, physical KV capacity 1024, two short native-generation prompts, and a same-input query=8 prefill diagnostic. They describe a locally patched compiler path and numerical policies. Reproduction on the recommended current upstream baseline remains pending. This is a support case, not a benchmark of unmodified public main.

## 1. Attention lowering and generated execution

In the query=8 complete-model timeline, the attention intervals between the V and O projections account for approximately **84%** of instrumented DSP model time. All 197 INT8 linear calls together account for approximately **8%**. The boundaries are marked by the existing projection calls and first/last Softmax exponential calls in all 28 layers. The intervals contain floating-point arithmetic, layout processing and runtime guard work; they must not be labeled as isolated QK, Softmax or PV operator costs.

Native generation also shows that DSP model execution dominates the warmed prefill request. During steady decoding, framework scheduling, output writeback and token selection outside the measured forward callback are a small fraction of the token interval. Improving the framework generation loop alone therefore does not address the measured DSP attention bottleneck.

**Requested support:** review our numerical-policy transformations, resulting Linalg/LLVM loops and target code; identify whether the selected pipeline or options prevent intended optimization; provide recommended attention lowering, scheduling, vectorization and math-runtime patterns that satisfy an agreed W8A8 accuracy contract. We can supply input/output identities, retained model objects, code excerpts and raw timers. An efficient, reviewed small attention example on the supported target would be particularly useful.

## 2. KV ownership, allocation and copies

The current ABI returns all 56 full-capacity K/V tensors on every model forward. The packet output is 235,488,768 bytes. External-call tracing observes 56 allocations of 4 MiB per decode forward, along with associated large copies. The same tensors also cross the ARM/DSP boundary. This combines compiler bufferization/ownership choices with our backend ABI and framework writeback requirements.

**Requested support:** a supported buffer/state representation that permits safe reuse or in-place KV updates while retaining native Transformers cache semantics, together with examples for persistent DSP-side state. We need guidance across frontend functionalization, bufferization and runtime ownership. Eliminating semantically required outputs is not assumed to be a compiler-only change.

## 3. Numerical policy and helper integration

The implementation preserves specific reduction order, rounding and FMA behavior because small floating-point changes before requantization can alter W8A8 outputs. We need to align the accuracy contract with efficient compiler transformations instead of accumulating special-case workarounds.

Separate instrumentation places our local Q2 integer guard at approximately **7%** of representative uninstrumented prefill model time. Millions of small copies were traced to that local helper, including scalar alignment and tail handling. We are tracking those implementation/compilation choices locally. They do not establish an upstream compiler defect.

**Requested support:** recommended numerical options and helper interfaces, clarification of which transformations belong in torch-mlir versus Hexagon-MLIR/LLVM, and review of efficient helper integration. ARM-specific bitwise reproduction should remain an explicit policy rather than a universal default.

## Deliverables we would like to agree on

1. A reviewed and reproducible attention lowering example or patch, with target code and numerical checks.
2. A documented ownership/state approach for reusable KV buffers and persistent execution.
3. A recommended public compiler commit and tested dependency combination for this model path.
4. An engineering contact and review route for small patches and complete-model regressions.

The public main already includes dependency and executor updates. Our request is for guidance and applicable source/tests supporting this full-model execution path. We can continue to maintain the Qwen3 reproduction and same-input regression harness while collaborating on these gaps.
