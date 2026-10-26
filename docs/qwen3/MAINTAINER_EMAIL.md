To: hexagon-mlir.support@qti.qualcomm.com

Subject: Qwen3 on QCS9075: measured attention bottleneck, KV integration and compiler support

Hi Hexagon-MLIR team,

We have extended the Hexagon-MLIR PyTorch examples to run the complete 28-layer Qwen3-0.6B forward pass on QCS9075 DSP. ARM retains PyTorch/Transformers tokenization and native generation. Our additions include bounded-dynamic prefill/decode, compiled artifact reuse across prompts, W8A8 numerical fixes, KV update integration and ARM/DSP runtime support. We have validated same-input logits and every K/V tensor, continuous generation, EOS, request reset and capacity boundaries.

The principal execution gap is in the DSP attention path. We excluded compilation, model loading and initial capture/first-use costs. Across 26 steady decode intervals, the mean was 2.570 seconds/token: 1.478 seconds in the DSP model kernel and 0.038 seconds in framework scheduling, AOT output writeback and token selection outside the callback. A warmed 13-token prefill took 8.695 seconds in the DSP kernel and 9.709 seconds to the first generated token.

For a same-input query=8 complete-model diagnostic, the uninstrumented DSP kernel took 6.726 seconds. The low-overhead timeline took 6.906 seconds, about 2.7% instrumentation overhead. Its all-layer breakdown is:

| Measured interval | Seconds | Fraction of timeline |
| --- | ---: | ---: |
| QK-side preparation and QK-related interval, before the first Softmax exponential | 2.972 | 43.0% |
| First through last Softmax exponential interval | 0.310 | 4.5% |
| PV-related processing and remaining work up to O projection | 2.508 | 36.3% |
| All INT8 matrix-multiply callees | 0.583 | 8.4% |
| Other intervals | 0.534 | 7.7% |
| Total | 6.906 | 100% |

The individual INT8 callee times are:

| Call | Calls across the model | Total seconds |
| --- | ---: | ---: |
| Q projection | 28 | 0.0691 |
| K projection | 28 | 0.0237 |
| V projection | 28 | 0.0231 |
| O projection | 28 | 0.0693 |
| MLP gate projection | 28 | 0.1039 |
| MLP up projection | 28 | 0.1040 |
| MLP down projection | 28 | 0.1041 |
| LM head | 1 | 0.0855 |

These callee measurements exclude quantization/dequantization outside the calls. The attention intervals include layouts, cache updates, normalization and local checks where applicable. Pure QK, complete Softmax and pure PV have not yet been timed separately. Our local Q2 integer guard was separately measured at 0.474 seconds and is being tracked as a local optimization item.

For KV, the public GPT-2 example configures two layers and executes a single input_ids-to-logits forward; it does not demonstrate feeding the previous KV into subsequent generation steps. We retained native Transformers StaticCache and connected all 56 Qwen3 K/V tensors to the DSP forward as runtime inputs/outputs. The extension covers position-indexed updates, chunked prefill, token-by-token decode, framework writeback and request reset. AOT-functionalized index_copy/scatter updates are included in the compiled graph.

The current KV owner is still ARM. Each FP32 cache tensor is [1,8,1024,128], making 224 MiB across 56 tensors. Each forward transfers the full cache to DSP and back, about 448 MiB of KV traffic, although one decoded token adds only 224 KiB of K/V. Decode tracing also observes 56 allocations of 4 MiB and corresponding copies each step. DSP-resident KV and a reduced state-update interface are not yet delivered.

We would like to work with you on:

1. **Attention lowering/code generation:** review the numerical policies and resulting Linalg/LLVM loops, separate QK/Softmax/PV from layout/update costs, and provide a recommended pipeline, efficient target example or applicable patches under an agreed W8A8 accuracy contract.
2. **KV ownership and reuse:** align frontend functionalization, bufferization/aliasing and runtime ownership to reduce full-cache allocation, copying and transfer while retaining native Transformers cache semantics.
3. **Source baseline and engineering contacts:** confirm the recommended public commit and dependencies, identify compiler/runtime owners, and publish applicable source/tests if newer internal versions already support these paths.

These measurements describe our pinned downstream implementation and numerical policies, not an unmodified current-main benchmark. We can share the exact options, model objects, raw timing packets and numerical checks, maintain the full-model Qwen3 regression, and contribute small fixes. An initial constant-partitioning patch on public main already passes an actual C++ pass regression.

Best regards,
Kevin
