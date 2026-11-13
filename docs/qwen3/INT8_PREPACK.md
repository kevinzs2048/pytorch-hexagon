# Qwen3 execution optimizations

QCS9075 DSP, Qwen3-0.6B, all 28 layers, batch one. Weights and activations use
W8A8; attention intermediates and KV storage remain FP32. Query lengths are
1–32 with a maximum of 1,024 occupied KV rows. This page describes the validated
FP32 path; FP16/HMX attention is not part of this runtime.

## What changed

1. **Persistent weight packing.** Preparation permutes immutable INT8 weights to
   `[N/128, K/4, 128, 4]`. Every forward reuses the prepared constants. There is
   no per-prompt packing and no cache keyed by temporary tensor addresses.
   Single-row groups prefetch a bounded 16 KiB packed tile ahead; weights
   remain immutable and the general-stride path is retained.
   Weight values and scales are unchanged. The pass admits K divisible by four
   and N divisible by 128 and checks all 197 contractions in each model graph.
   The runtime retains modulo-INT32 accumulation. For more than four query
   rows, it reuses each 128-column weight tile across all row groups before
   moving to the next tile. Checked aligned loads have a general-stride fallback.
   Complete four-row groups have a separate loop from the final partial group,
   allowing the compiler to remove row-count branches from the inner reduction.
   This changes scheduling only; arbitrary admitted query lengths retain their
   bounded tail handling.
2. **Four integer workers.** Independent output-column tiles run through the
   pinned llama.cpp worker pool. The integer arithmetic is implemented here;
   llama.cpp's block-quantized matrix kernels are not substituted for torchao.
   Each worker pipelines two packed weight tiles through llama.cpp's DMA
   queue into VTCM. The largest supported reduction uses 3 MiB across four
   workers. A cached VTCM lease is acquired and released for each matrix
   multiplication; unavailable resources or larger reductions retain the DDR
   kernel. VTCM tiles disable the DDR prefetch instruction. Accumulation order,
   weight bytes and quantization remain unchanged.
3. **One SiLU exponential per element.** Explicit pure-helper attributes let
   the compiler eliminate repeated calls. A q32 diagnostic verified that
   13,762,560 calls became 2,752,512, without changing arithmetic or outputs.
   See the [compiler reproducer](repro/silu-helper-effects.mlir). Prefill and
   decode SiLU run on four independent workers, preserving exp/add/division.
4. **Shared activation preparation.** Identical checked reciprocals for shared
   Q/K/V and gate/up inputs are reused before ordinary CSE. The original
   assertion and division are retained; scales, rounding and INT8 thresholds
   are unchanged. Coverage requires exactly 84 duplicate reciprocals. For
   all four domains, scale assertions move before the loop and ties-to-even
   rounding uses equivalent integer operations, exposing vectorization. A
   compiler optimization implements 32-lane IEEE min/max reductions with
   ordered integer keys, preserving NaNs and signed zero.
5. **Ordered FP32 attention.** Occupied K/V rows are packed with bit-preserving
   HVX shuffles. Four workers process independent heads; prefill uses eight
   workers when both query length is at least eight and occupied KV length is
   at least 256. Dot helpers pair query rows in prefill and the two query heads
   sharing one KV head in decode. Each output retains its QK
   FMA sequence, four-lane softmax sum and ARM PV block boundaries. For q1 with
   KV below four, QK retains the original unfused multiply/add. For q1 with KV
   at most 128, V is read directly from the caller's strides to avoid packing.
   Longer KV uses the packed path. During decode, each worker packs only the
   KV heads it consumes. No persistent secondary KV cache is added.

The attention arithmetic is implemented in this repository. The worker pool
and DMA queue primitives are reused from llama.cpp; no FP16/HMX attention kernel is linked
into this path. The dependency is pinned to
`f818065d753700341b5355ce36ca676ecfb5d139`.

## Build

In the resident plan, set these root fields:

```json
{
  "integer_weights": "hvx-i8-prepacked-v1",
  "integer_threads": 4,
  "silu_effects": true,
  "activation_reuse": true,
  "parallel_silu": true,
  "vectorize_activation": true
}
```

Set `"attention": "fp32-ordered-workers-v1"` in the `initial`, `prefill` and
`decode` domain entries. `initial_single` retains its original attention graph.
All four domain entries must identify captured `linalg` / `linalg_sha256` inputs
and their model bindings. These options are explicit opt-ins.

Use `models.qwen3.build_model --compile-resident` with `--resident-spec` and
`--llama-source` as described in [USAGE](../../USAGE.md). This maintained command
prepares, compiles and links every domain; an existing object bundle is not
required. The generated `resident-plan.json` records the compiled bundles and
can subsequently be passed without `--compile-resident` for a verified relink.
The builder checks complete prepared IR and immutable weight content. Plain
objects cannot be relabeled as packed objects. Frontend capture/export remains
a separate step.

## Exact INT8 HMX prefill

The optional `triton_int8_prefill` plan entry selects HMX for 32-row linear
operations with K in {1024, 2048, 3072} and N divisible by 128, up to 3072.
It also admits 16–31-row prompt tails when N is at least 1024. Smaller queries,
narrow tails, unsupported strides and unavailable HMX/VTCM resources use the
existing HVX implementation. The conservative tail threshold follows same-input
QCS9075 measurements including resource acquisition, packing, DMA and accumulation:
HMX loses at 1–8 rows and wins at 16–31 rows for the model's large linear layers.
This does not change attention,
KV precision, prompt chunking, quantization scales or model generation semantics.

- `kernels/int8_prefill.py` expresses activation packing and output accumulation
  in Triton. The compiler lowers four 128-byte rows to HVX byte shuffles and
  admits small tiles within the explicitly bounded stack budget.
- `native/runtime/int8/hmx_prepacked.cpp` leases VTCM/HMX, launches the kernels
  and calls the vendor HexKL INT8 micro API. `hmx_dma.c` uses the pinned
  llama.cpp queue for two-buffer DMA; neither file reimplements the matrix math.
- Existing `[N/128,K/4,128,4]` weights are read using 2D DMA into HMX's
  `[K/4,32,4]` tile layout. There is no additional model-sized weight copy.
  The runtime publishes each immutable weight once for DMA visibility.
- Real activation rows are converted from signed INT8 to unsigned `A+128`.
  Padding rows contain 128, so HMX output row 32 equals `128*sum(W)`.
  Subtracting that row recovers the exact signed dot product, without a separate
  weight-sum pass or cache. The kernel preserves `C += A@W`, including nonzero C.
  K≤3072 keeps both the raw HMX dot and zero-point correction within INT32.
- The resource descriptor is cached, but the physical VTCM/HMX lease is released
  after every call. The weight identities live only for the immutable model
  session. An error after output accumulation starts aborts; it cannot safely
  repeat the operation through a fallback.

Build both kernels with the same reviewed compiler tools. Each command emits
`kernel.o` and a manifest recording source/tool hashes. LWP-instrumented objects
are rejected by the model build. Rebuild both kernels when upgrading from the
32-row-only runtime: the new manifests must declare `query_rows: [1, 32]`.
An older manifest declaring only 32 rows cannot authorize the tail path.

```sh
python -m models.qwen3.kernels.compile strict-int8-pack \
  --output-dir /path/to/int8/pack \
  --libtriton /path/to/build/compiler/profiles/dynamic-kv/libtriton.so \
  --triton-shared-opt /path/to/triton-shared-opt \
  --linalg-hexagon-opt /path/to/build/compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt
python -m models.qwen3.kernels.compile strict-int8-accumulate \
  --output-dir /path/to/int8/accumulate \
  --libtriton /path/to/build/compiler/profiles/dynamic-kv/libtriton.so \
  --triton-shared-opt /path/to/triton-shared-opt \
  --linalg-hexagon-opt /path/to/build/compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt
```

Keep `integer_weights="hvx-i8-prepacked-v1"` and `integer_threads=4`.
Add `triton_int8_prefill` with `pack` and `accumulate` records, each containing
`object` (absolute `kernel.o` path), `object_sha256`, and `manifest_sha256`.
Use SHA-256 of the actual object and adjacent `manifest.json`; do not edit the
manifest to admit a different object. The existing `models.qwen3.build_model`
resident build selects the CMake runtime archive, links HexKL after its users,
records hashes, and preserves the selection when generating a relink plan.
The toolchain configuration's `hexkl` must identify the v73 HexKL addon.
Omitting this entry retains the current HVX path.

The November 8 device checks cover exact INT64 matrix references, nonzero C,
full-model logits/KV and retained Chinese/code generation. The public build
entry has rebuilt all four model execution domains from retained captured Linalg,
with fresh strict Triton kernels. This does not claim a fresh Transformers capture
or a clean-machine dependency installation. Historical numerical references
were independently checked against ARM W8A8; they were not recomputed for each build.
A source/object build receipt alone is not numerical model acceptance.

The HMX model link uses the matching SDK `hexagon-link` because the HexKL micro
archive contains SDK-specific HMX bitcode. Constant libraries still use the
configured source-built LLD. The builder records the SDK linker identity and
rejects unresolved LLVM intrinsic names before emitting a session receipt.
