# Qwen native extensions

These sources are owned by the Qwen application, including the C++/MLIR code.
They are not part of the general Hexagon-MLIR backend.

- `compiler/Profiles.cmake`: q1/q2 and effective-length `dynamic-kv` compiler targets, loaded through CMake's
  standard `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` entry.
- `buffer_reuse/`: the selected attention/layout transformations.
- `policy/`: first/sums numerical stages, INT8 matching and SiLU effects.
  Its tests use `qwen-policy-sums-opt`, not the general backend optimizer.
- `runtime/`: model-linked DSP helpers and their include closure.
- `runtime/resident_session.cpp`: DynamicCache session with one DSP KV
  owner. ABI declarations are generated from actual captured input provenance by
  `../runtime/arm/prepare_qwen_resident_bindings.py`; no fixed-capacity history
  packets are required. Shared allocation lives in `hexagon_runtime/native/`.
- `arm_rope_trig_runtime/`: pinned SLEEF preparation and RoPE adaptation.
- `diagnostics/` and `runtime/profile/model_*`: optional profiling support.

Build with `python -m models.qwen3.build` from the application root. General MLIR types,
dialects and lowering support are taken from the explicitly selected compiler
checkout. Build receipts check both that checkout and these native sources.
There is no dependency from the compiler repository back into this directory.

The first/sums stages retain their existing numerical implementations. This
ownership change does not merge algorithms or change their floating-point order.

### Immutable INT8 prepacking (opt-in)

A resident build plan may set `"integer_weights": "hvx-i8-prepacked-v1"` at
its root. The default is `"plain"`. All execution domains must have captured
Linalg and matching compiled object bundles. `build_model --compile-resident`
prepares and compiles all four captured domains before linking, without requiring
preexisting objects. Without that flag, the linker rejects old
objects whose complete prepared IR/weights do not match.

The optional `hexagon-prepack-i8-weights` pass runs after integer-call lowering
and attention selection. It admits only a constant INT8 weight behind a rank-2
transpose, with K divisible by 4 and N divisible by 128. It permutes bytes into
`[N/128, K/4, 128, 4]` order and calls the versioned
`__hexagon_i8_prepacked_v1_matmul` ABI. The existing W8A8 scales, floating-point
processing and modulo-INT32 accumulation semantics remain unchanged. Unsupported
or mutable weights fail the required 197-call coverage check.

Packing occurs during model preparation, not at each forward or prompt. The
packed data replaces the weight used by the integer kernel and has the same
byte count; compiler canonicalization removes unused transpose paths. The
runtime consumes the immutable packed constants directly, without a pointer-keyed
cache. The format can serve every admitted query length 1..32.

The packed path passed full-model device byte comparisons and paired timing,
including the original FP32-attention path. See
[kernel contracts](../../../docs/qwen3/INT8_PREPACK.md).

`"integer_threads": 4` additionally selects the pinned llama.cpp worker pool
and DMA queue primitives, and partitions output columns into independent groups
of 128. Two weight tiles per worker overlap DMA transfers into VTCM with exact
integer computation. The cached VTCM lease ends after each matrix multiplication;
unavailable resources retain the DDR path. It requires
`--llama-source`; the default is one worker. This option also supports FP32
attention without linking llama.cpp's fused attention kernels.
The four-worker session passed 72 paired model forwards and all four execution
domains. `"silu_effects": true` also exposes the existing scoped helper-effect
policy to resident graphs, eliminating redundant SiLU exponent calls without
changing arithmetic. Objects must be recompiled from those prepared graphs.

### Ordered FP32 attention and activation reuse (opt-in)

A domain's `"attention": "fp32-ordered-workers-v1"` selects the ordered FP32
runtime for initial prefill, continuation prefill or decode. It reuses the pinned
llama.cpp worker pool, but implements attention arithmetic locally. HVX shuffles
pack occupied FP32 K/V rows without conversion; blocked and paired-query dot
products preserve each output's FMA order. Prefill pairs query rows; decode
pairs the two GQA query heads sharing one K/V head. Softmax and PV retain the
admitted ARM reduction order, including the unsplit decode reduction. Dispatch
uses four compute workers, or eight for at least eight query rows and 256
occupied KV rows. Packing uses four workers: each computes its own head range
in the four-worker path; the eight-worker path waits for packing and cache
publication before computing. Scratch maintenance covers the used rows. The
empty-cache singleton keeps its original graph.

Root `"activation_reuse": true` enables `qwen-reuse-activation` in the existing
packed optimizer, followed by CSE. It shares 84 identical checked reciprocals
for Q/K/V and gate/up activation preparation. The first nonzero assertion stays;
reuse requires the same dominating tensor input, initializer, maps, iterators,
attributes and scalar arithmetic. It neither drops the initial check nor changes
quantization scales or rounding. The default is false. Changed graphs require
fresh compilation and numerical acceptance, just as weight prepacking does.

`"parallel_silu": true` distributes the original scalar SiLU expression over
four workers. `"vectorize_activation": true` hoists checked row-scale assertions
and expands the existing ties-to-even rounding into exact integer operations.
Both options apply to all four resident domains, including single-token decode;
the two single-token entry points require a static query length of one. These
options preserve scales and arithmetic and require recompilation of changed
graphs. Prefill and decode use the same runtime implementations.

Root `"batch_arm_rope": true` enables four-lane RoPE result reuse through the
existing policy optimizer (`--qwen-batch-arm-rope=expected-count=2`). The default
is false. It replaces repeated lane selections with one destination-buffer call
per group, retaining the pinned SLEEF sin/cos algorithm and its shared range
reduction decisions. The matcher requires the original marked expressions,
identical four-lane input groups and FP32 `[1, query, 128]` tensors. It rejects
changed indexing or helper interfaces. This requires rebuilding both
`qwen-silu-effects-opt` and the DSP `trig_runtime.o`, then recompiling captured
resident graphs; an older runtime does not provide the new entry points.
The preparation record and session manifest record the option. Numerical
qualification remains separate from building these artifacts.

Single-row INT8 groups prefetch up to 16 KiB of the next packed-weight tile.
The lookahead is bounded by the current tile; weight layout, values, scales and
modulo-INT32 accumulation are unchanged. Multi-row groups keep the existing loop.
