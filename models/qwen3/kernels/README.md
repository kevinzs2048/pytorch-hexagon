# Triton model kernels on Hexagon

Attention arithmetic is expressed in Triton and lowered through Triton Shared
and Hexagon-MLIR. The C++ adapter only validates arguments, allocates reusable
scratch and schedules programs. The normal model build still selects the
accepted FP32 implementation; these kernels require an explicit build choice.

| Source | Contract | Precision and status |
| --- | --- | --- |
| `strict_decode.py` | q1, grid `(1, 4)` | Ordered FP32 QK/PV, compatible exponential and softmax sum. Full-model logits/new KV and four 64-token generation requests matched the accepted baseline in the development comparison. |
| `hvx_decode.py` | q1, grid `(1, 4)` | FP32 storage, QFloat QK/PV and tree QK reduction; compatible softmax. Explicit `qfloat-decode` profile; changes rounding. |
| `hmx_decode.py` | q1, grid `(1, 8)` | FP32 storage, FP16 HMX dots/probabilities/results, two live softmax rows; shares the prefill HMX owner. Explicit `experimental-decode-fp16` profile. |
| `attention.py` | q2–32, grid `(1, 8)` | Experimental FP16 Q/K/V/probabilities **and dot outputs**; FP32 online state, explicit pairwise sums, approximate exponential and reciprocal normalization. Full-model outputs differ from the accepted baseline. |
| `silu.py` | 1–32 rows × 3072 columns, grid `(4,)` | FP32 compatibility exponential and strict division; optional activation replacement. Reuses `_exp_compat` from `strict_decode.py`. |
| `quantized_silu_up.py` | 4–32 rows × 3072 columns, grid `(4,)` | Experimental SiLU × dequantized up with strict row/max/boundary repair. Its FP32 output may only feed the specified symmetric row quantizer; independent numerical review and model acceptance remain pending. |

All attention kernels require batch 1, 16 query heads, 8 KV heads, head dimension 128, and
`query_length <= kv_length <= 1024`. Q/K/V storage is FP32 with explicit
head/token strides and contiguous features. Output is contiguous
`[query, 16, 128]`. The queries occupy the final consecutive positions in the
occupied KV sequence. Arbitrary padding masks are unsupported. Effective KV
length controls the loops; allocated capacity does not determine work.

Strict decode shares each packed K/V tile between two query heads. For
`kv_length <= 128`, it reads V directly and avoids V packing. Its scratch is
1,064,960 FP32 elements (4,259,840 bytes), allocated once by the adapter and
partitioned across four programs. The exponential adaptation retains its
SLEEF/Boost license in the source. It does not call llama.cpp attention.

Experimental prefill processes the two heads sharing a KV head together and
keeps softmax state online. The same compiled object selects 32 rows for
q2–16 and 64 rows for q17–32, including both heads; each key block has 128
positions. This runtime choice avoids padding every short query to 64 rows.
It preserves each row's arithmetic and does not compile for individual prompts.
The precision policy is fixed and checked.

## Build the kernel objects

First prepare the compiler/toolchain described in [USAGE.md](../../../USAGE.md).
Use a compiler containing the tile-copy, row-reduction, pointer-select and HMX
dispatch changes. The frontend requires the corresponding installed Triton
Python package. Select an isolated `libtriton.so`; this command does not replace
the installed extension or build/deploy an entire model.

Run from the repository root, setting these paths to your built tools:

```bash
kernel_compiler=/absolute/path/to/libtriton.so
triton_shared_opt=/absolute/path/to/triton-shared-opt
hexagon_opt=/absolute/path/to/linalg-hexagon-opt
kernel_output="$PWD/.local/attention-kernels"

python -m models.qwen3.kernels.compile strict-decode \
  --libtriton "$kernel_compiler" \
  --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" \
  --output-dir "$kernel_output/decode"

python -m models.qwen3.kernels.compile experimental-prefill-fp16 \
  --libtriton "$kernel_compiler" \
  --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" \
  --output-dir "$kernel_output/prefill-fp16"
```

Each output directory must be new. It contains `kernel.o`, the TTIR/Linalg
intermediates, stage logs, complete backend options, and a manifest with source
and compiler hashes. Preparation folds tile copies before reducing the explicit
sum/max trees. Neither step rewrites arithmetic by matching temporary SSA names.
Compilation is excluded from inference timing.

### Optional hybrid decode

Use HVX for short effective KV and HMX for longer sequences. The current
QCS9075 crossover measurement supports `min_kv: 256`; this is an explicit build
choice, not an input-content specialization. FP32 KV ownership and capacity stay
unchanged. These profiles change arithmetic and require model quality checks.
The strict decode profile remains available for rollback.

```bash
llvm_bin=/absolute/path/to/qualified/llvm/bin
python -m models.qwen3.kernels.compile qfloat-decode \
  --libtriton "$kernel_compiler" --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" --schedule-fma-chains \
  --output-dir "$kernel_output/decode-qfloat"

python -m models.qwen3.kernels.compile experimental-decode-fp16 \
  --libtriton "$kernel_compiler" --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" \
  --llc "$llvm_bin/llc" --llvm-extract "$llvm_bin/llvm-extract" \
  --llvm-opt "$llvm_bin/opt" \
  --reuse-hmx-workspace --prepare-hmx-matmul --external-allocator \
  --output-dir "$kernel_output/decode-hmx"
```

Build `experimental-prefill-fp16` with the same three HMX/workspace flags.
The HMX decode compiler extracts its entry and internal helpers, leaving the
shared runtime unresolved for the prefill object to supply. It must not bring
another HMX owner or independent VTCM pool into the model.

The existing resident plan accepts `triton_hmx_decode` alongside
`triton_decode` and `triton_hmx_prefill`. Supply the absolute `object` path,
`object_sha256`, `manifest_sha256`, and integer `min_kv` (1..1024).
`build_model.py` preserves this selection and its hashes when relinking.
The attention CMake entry exposes the equivalent
`TRITON_HMX_DECODE_OBJECT` and `TRITON_HMX_DECODE_MIN_KV` options and requires
`SHARE_HMX_RUNTIME=ON`. Instrumented objects are rejected by model builds.

Operator compilation and an attention archive build are distinct from a full
model source rebuild. Reusing accepted graph objects must be recorded as a
relink; it does not refresh the graph/compiler source acceptance seal.

### Experimental quantized SiLU × up object

```bash
predicate_fixed_llc=/absolute/path/to/llc
python -m models.qwen3.kernels.compile experimental-quantized-silu-up \
  --libtriton "$kernel_compiler" \
  --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" \
  --llc "$predicate_fixed_llc" \
  --output-dir "$kernel_output/quantized-silu-up"
```

The explicit LLVM must include both the native-predicate packing fix
HEX-PRED-001 and the wide-lane truncation fix HEX-PRED-003, described in
[LLVM_ISSUES.md](../../../docs/qwen3/LLVM_ISSUES.md). This profile emits LLVM IR,
runs that `llc`, and records its hash; it does not replace the default compiler.
The build requires no files from a previous profiling scratch directory.

For the measured interleaved implementation, rebuild `libtriton` with the
multi-result vector-writeback fix documented as MLIR-VEC-001 in that issue log.
The first pass keeps its maximum and domain-check accumulators in vectors;
older compiler builds may allocate temporary tensors inside that loop.
Two independent 32-lane products are interleaved before their output stores.
This follows the independent-vector loop organization used by llama.cpp's
sigmoid, while retaining this kernel's own FP32 expressions and quantization
repairs. A four-vector variant was slower in the measured raw-input pairing
and is not retained. Native 64-lane predicates also expose HEX-PRED-002.

On QCS9075, replaying the 28 saved q32 layers, including ordered up
dequantization, reduced the sum of layer medians from 20.457 to 14.274 ms.
A separate 128-token model pairing measured 950.317 to 926.274 ms, with the
resident API's first argmax token at 1022.325 to 1001.014 ms. The 12 requests'
logits/KV payload hashes and eight-token sequences match the prior experimental
baseline; the five quantization-floor fixtures also preserve scale/INT8 results.
These figures exclude compilation/loading/checks and are not full Transformers
end-to-end measurements or independent FP16-attention acceptance. Evidence:
`.local/swiglu-interleave-20261107/REPORT.zh-CN.md`.

The subsequent block-repair refinement keeps the same approximation domain and
per-element arithmetic. If a row contains rejected inputs, it strictly
recomputes only their 32-element blocks, rebuilds the row peak, then applies the
existing maximum/rounding-boundary repairs to the mixed exact/approximate row.
A nonfinite result still triggers a complete strict row. Finite intermediates
can therefore differ from the former whole-row fallback; only the specified
scale and INT8 consumer are the equivalence target.

Against the interleaved version, the raw 28-layer sum decreased from 14.314 to
12.072 ms. A separate full-prompt pair measured 927.786 to 918.022 ms of model
computation and 997.635 to 986.844 ms to the resident first argmax. All 132
forward payload hashes and 12 eight-token sequences match the experimental
baseline. Mixed-domain, quantization-floor and nonfinite fixtures also pass;
nonfinite rows match the previous guarded kernel byte for byte. This does not
resolve the independent numerical-bound or FP16-attention acceptance work.
Evidence: `.local/swiglu-block-repair-20261107/REPORT.zh-CN.md`.

The launcher supplies contiguous FP32 gate and expanded row-scale matrices,
INT32 up-projection accumulators, FP32 per-channel weight scales, and a separate
FP32 output matrix. The up expression retains `(row_scale * float(accumulator))
* weight_scale`. The supported consumer uses
`scale = max(max(abs(output))/127, float32(1e-5))` and
`clamp(roundeven(output * (1/scale)), -127, 127)`. The intermediate FP32 output
is approximate and must have no other consumers. Queries below four rows retain
the original strict model path.

This command builds only the kernel object. To build the optional adapter,
first build the `strict-silu` object described below, then configure:

```bash
cmake -G Ninja -S models/qwen3/native/runtime/attention \
  -B "$kernel_output/quantized-silu-runtime" \
  -DLLAMA_SOURCE="$llama_source" \
  -DHEXAGON_SDK="$hexagon_sdk" \
  -DHEXAGON_TOOLS="$hexagon_tools" \
  -DTRITON_SILU_OBJECT="$kernel_output/silu/kernel.o" \
  -DTRITON_SILU_UP_OBJECT="$kernel_output/quantized-silu-up/kernel.o"
cmake --build "$kernel_output/quantized-silu-runtime" \
  --target qwen-triton-quantized-silu-static -j4
```

`libqwen_triton_quantized_silu.a` contains both kernels and a single shared
worker pool / concurrency guard. When explicitly linking a candidate graph,
use it **instead of** `libqwen_triton_silu.a`. The strict entry remains
`_mlir_ciface_qwen_ordered_silu_v1`; the quantizer-specific entry is
`_mlir_ciface_qwen_quantized_silu_up_v1`. Its arguments are strided memref
descriptors for gate/output `[1,rows,3072]`, accumulator/expanded scale
`[rows,3072]`, and weight scale `[3072]`, in gate/accumulator/scale/weight/output
order. Noncontiguous matrices, queries outside 4–32 and any output overlap
with an input are rejected before dispatch. All four workers use Triton
arithmetic. CMake checks both object hashes and profiles against their adjacent
manifests and rejects LWP objects, whose shared trace storage is unsuitable here.

This is an explicit build option, not automatic model selection. The graph
must retain strict computation below four rows and validate the complete
quantizer consumer contract. A maintained model-graph matcher, numerical-contract
review and independent model acceptance are still required before deployment.
No default model or arithmetic policy selects this candidate.

For experimental prefill, `--reuse-hmx-workspace` additionally enables the
compiler's `reuseHMXWorkspace` option (default: off). It retains one exclusive
VTCM scratch buffer on the HMX owner thread instead of allocating/freeing it
for each outlined matmul. The compiler only transforms a fixed allocation
whose pointer does not escape and whose release follows every use on every
returning path. Unsupported functions retain their existing allocation calls.
The buffer grows as needed and is freed on owner-thread shutdown; it reserves
VTCM between calls. This option requires `dispatchHMX`, does not change the
attention arithmetic, and does not qualify the experimental FP16 model as
equivalent to the accepted FP32 model.

## Build the strict decode adapter

The following optional archive uses Triton for q1 and retains the accepted C
FP32 implementation for q2–32. It provides the existing
`_mlir_ciface_qwen_ordered_fp32_attention_v1` ABI, so the model graph and weights
do not need recompilation. Use it **instead of** the existing attention archive
when linking a development model library.

```bash
cmake -G Ninja -S models/qwen3/native/runtime/attention \
  -B "$kernel_output/runtime" \
  -DLLAMA_SOURCE="$llama_source" \
  -DHEXAGON_SDK="$hexagon_sdk" \
  -DHEXAGON_TOOLS="$hexagon_tools" \
  -DTRITON_DECODE_OBJECT="$kernel_output/decode/kernel.o"
cmake --build "$kernel_output/runtime" \
  --target qwen-triton-strict-attention-static -j4
```

Here `llama_source` is the pinned checkout already required by the worker pool;
`hexagon_sdk` is the SDK directory and `hexagon_tools` contains `bin/`.
The result is `libqwen_triton_strict_attention.a`. It includes the adapter,
Triton object, renamed FP32 prefill and worker pool. It validates the q1 tensor
shapes and strides, rejects a nonzero additive mask, and uses four reusable
32 KiB worker stacks. Concurrent entry on the same loaded adapter returns an
error instead of sharing its scratch. Existing model helpers and SDK libraries
are still required by the surrounding model link.

### Optional strict decode scheduling

**Requires the LLVM fix [HEX-PRED-003](../../../docs/qwen3/LLVM_ISSUES.md) in
the selected compiler library.** Earlier compilers can generate incorrect
32-lane rounding at negative half-integer boundaries, producing NaN. The
unchanged eight-element default does not expose that vector lowering.

With that LLVM fix, a compiler containing `ScheduleFMAChains.h` also supports
`--schedule-fma-chains` on the `strict-decode` object command. This selects
32-element exponential/normalization blocks and schedules independent FP32
FMA lanes. QK/PV accumulation, four-lane denominator reduction, KV storage and
the compatibility exponential formula remain unchanged. The default is still
eight-element blocks without that scheduling option.

With corrected decode, strict SiLU and quantized repair objects, a strict
FP32-attention model ABBA measured seven decode forwards at 471.402 → 453.491 ms
and their resident packet calls at 501.080 → 484.877 ms, following a128-token
prompt. All132 logits/new-KV payloads and12 eight-token sequences matched.
A separate replay matches262 Chinese/code records previously independently
checked against ARM W8A8, including130 full-KV snapshots. Strict SiLU also
matches ARM at half-integer exponential-rounding boundaries and adjacent FP32
values; the guarded product preserves its scale and INT8 consumer results.

Earlier standalone timings (KV129:0.966 → 0.882 ms; KV1024:8.151 → 7.361 ms)
used the pre-fix compiler and finite fixtures that did not expose its bug.
They are diagnostic history, not acceptance of those old objects. KV8 also
regressed from72 to79 us, so this remains an explicit option. Timings exclude
compilation/loading/checks and are not full Transformers end-to-end results.
The complete public model source-build receipt remains separate. Current
model evidence: `.local/strict-current-20261107/current-evidence.tar.xz`;
earlier operator/LWP evidence: `.local/decode-softmax-20261107/`.

## Build the experimental prefill adapter

This optional archive uses strict Triton FP32 decode for q1 and experimental
Triton FP16 attention for q2–32. It is **not** an accepted FP32 model replacement.
The default model build continues to select the accepted implementation.

Compile a fresh prefill object with generated-code allocation hooks:

```bash
python -m models.qwen3.kernels.compile experimental-prefill-fp16 \
  --libtriton "$kernel_compiler" \
  --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" \
  --reuse-hmx-workspace --external-allocator \
  --output-dir "$kernel_output/prefill-pooled"

cmake -G Ninja -S models/qwen3/native/runtime/attention \
  -B "$kernel_output/runtime-prefill" \
  -DLLAMA_SOURCE="$llama_source" \
  -DHEXAGON_SDK="$hexagon_sdk" \
  -DHEXAGON_TOOLS="$hexagon_tools" \
  -DTRITON_DECODE_OBJECT="$kernel_output/decode/kernel.o" \
  -DTRITON_PREFILL_OBJECT="$kernel_output/prefill-pooled/kernel.o" \
  -DTRITON_PREFILL_POOL_BYTES=1048576
cmake --build "$kernel_output/runtime-prefill" \
  --target qwen-triton-experimental-attention-static -j4
```

The result is `libqwen_triton_experimental_attention.a`. It contains both kernel
objects, their adapters, the worker pool and `TemporaryBufferPool`. When linking
a development model, use it instead of the other attention archive; do not also
compile `temporary_buffer_pool.cpp` into that library. Put the attention archive
before the HexKL and SDK C++ support libraries, or place those static libraries
in a linker `--start-group` / `--end-group`. This archive is a build component,
not a standalone model or a newly sealed whole-model source build.

The compiler's default-off `externalKernelAllocator` option redirects generated
malloc/free calls to `hexagon_kernel_malloc(uint64_t)` and
`hexagon_kernel_free(void*)` **before** runtime bitcode is linked. Runtime-owned
allocations therefore retain their original allocator. Do not replace allocator
symbols across the final object with `objcopy`.

`TRITON_PREFILL_POOL_BYTES` defaults to zero; the maximum is 1 MiB per logical
worker. Four independent pools retain explicitly released DDR buffers between
calls. TLS binds a pool to the current job because physical threads may change.
HMX-owner allocations outside that job use ordinary malloc; foreign releases
and quota pressure fall back safely, and live results are detached at job exit.
The limit covers cached/owned buffers, not total DSP memory or pool metadata.

CMake checks the adjacent `manifest.json`, object hash, precision profile and
allocator option. LWP objects are rejected by this parallel adapter. Runtime
validation admits only canonical causal masks with zero visible entries and
negative infinity or FP32 minimum for future entries; padding or additive masks
return an error. Q/K/V offsets and strides are validated before execution.

### Optional preparation and transpose fusion

Add `--prepare-hmx-matmul` when building experimental prefill to keep QK's
private transpose until HMX packing and prepare both matmul inputs on the
calling HVX worker. HMX execution remains on its owner thread; different
workers can prepare their own tiles while that owner is busy. Arithmetic stays
in Triton/MLIR. This is synchronous dispatch with overlap across workers, not
llama.cpp's full within-head DMA/compute pipeline.

```bash
python -m models.qwen3.kernels.compile experimental-prefill-fp16 \
  --libtriton "$kernel_compiler" \
  --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" \
  --external-allocator --reuse-hmx-workspace --prepare-hmx-matmul \
  --output-dir "$kernel_output/prefill-prepared"
```

Use the resulting object with the experimental adapter above and explicitly
select its worker-pool budget. This option starts from the Triton source:
enabling it on an intermediate that already materialized/vectorized the
transpose cannot recover that fusion. It is off by default and does not change
the model builder's accepted FP32 selection.

The compiler admits contiguous static FP16 inputs with complete 32×32 tiles
and at most 128 KiB of preparation storage per matmul. Unsupported shapes keep
the original decomposition. Transpose fusion additionally requires a private
output with no other observer or intervening input mutation. A 2 KiB alignment
keeps scatter regions within a page; explicit scatter completion precedes HMX
dispatch. Four runtime leases retain at most 512 KiB of VTCM. Extra callers
wait, nested leases are rejected, and allocation and final cleanup execute on
the owner thread. This VTCM reservation is separate from the DDR worker pools.

The maintained source/CMake route passed 16 forward calls and four complete KV
snapshots against the same experimental precision policy. On QCS9075 at
1459.2 MHz, a warm ABBA comparison (four samples per version, q32, past992,
28 layers, graph and DDR worker pools enabled in both) reduced full forward
from 508.603 to 461.990 ms and the 28 attention calls from 168.244 to 122.781 ms.
Compilation, loading, seed setup and output comparisons are excluded. All12
forwards including warmups matched the experimental baseline. Preparation
without transpose fusion was slower (508.070 → 514.655 ms), so that intermediate
is not a recommended optimization.

These checks establish integration and unchanged experimental results, not
independent FP32 equivalence or continuous-generation acceptance. The fixed
long-KV case still has a maximum logits difference of 1.238635 against the
independent FP32 reference. An earlier 448.720 ms prototype measurement uses
different generated helpers and is not the result of this maintained build.

## Build the optional strict SiLU adapter

The 32-lane compatibility exponential also requires [HEX-PRED-003](../../../docs/qwen3/LLVM_ISSUES.md),
including when FMA scheduling is disabled. Rebuild the selected compiler library
with that fix before building this kernel or the quantized SiLU repair kernel.

The compiler must include the parallel subvector change in
`VectorizationPass.cpp`: mixed FP32/predicate computations must accept a
32-element parallel tile. This does not relax reduction ordering. Without
that change this kernel compiles but remains scalar.

```bash
python -m models.qwen3.kernels.compile strict-silu \
  --libtriton "$kernel_compiler" \
  --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$hexagon_opt" \
  --output-dir "$kernel_output/silu"

cmake -G Ninja -S models/qwen3/native/runtime/attention \
  -B "$kernel_output/silu-runtime" \
  -DLLAMA_SOURCE="$llama_source" \
  -DHEXAGON_SDK="$hexagon_sdk" \
  -DHEXAGON_TOOLS="$hexagon_tools" \
  -DTRITON_SILU_OBJECT="$kernel_output/silu/kernel.o"
cmake --build "$kernel_output/silu-runtime" \
  --target qwen-triton-silu-static -j4
```

With a compiler that includes `ScheduleFMAChains.h`, add
`--schedule-fma-chains` to the `strict-silu` compile command to enable bounded
four-lane scheduling of strict FP32 FMA chains. The compiler option is
`scheduleFMAChains=True`; it defaults to false. This preserves the Triton
formula and per-element operation order. It changes neither division nor the
attention precision policy. The kernel manifest records the selection.
The transform admits only connected 32-lane FP32 chains on the v73 IEEE,
128-byte HVX target. It skips observable intermediate vectors, fast math,
strictfp functions and denormal flushing. Added private buffers are bounded
to 1536 bytes/function; this is not a bound on the entire generated stack frame.
The model build's default SiLU selection is unchanged.

The strict SiLU build uses a 4096-byte stack-promotion budget for bounded
integer temporaries. Division by a positive normal denominator uses integer
mantissas and restoring division, followed by round-to-nearest, ties-to-even.
It preserves the FP32 result, including signed zero. A block containing an
unsupported exponent, subnormal quotient, nonfinite input, or nonpositive
denominator uses the original strict division. The compatibility exponential
is unchanged. The manifest records the allocation-admission budget; it is not
a bound on the complete generated stack frame.

Link `libqwen_triton_silu.a` instead of `ordered_silu.o` or
`libqwen_ordered_silu.a`, after the objects that reference it. The archive
provides the existing `_mlir_ciface_qwen_ordered_silu_v1` ABI and worker pool.
The original tensor validation, cache maintenance and concurrency guard are
shared with the scalar implementation. Input has contiguous columns and a
row stride of at least 3072; output is contiguous. All arithmetic remains in
Triton. There is no new runtime environment switch; the default model build
continues to use scalar SiLU.

Development checks on QCS9075 found bit-identical output for random normal,
random finite-bit-pattern, strided and special-value fixtures. At 1171.2 MHz,
32×3072 standalone SiLU decreased from 4.245 to 1.741 ms. In the existing
**experimental FP16-attention model**, 28 SiLU calls decreased from 90.929 to
38.527 ms and full forward from 601.952 to 558.374 ms (warm ABBA medians,
q32, past 992, effective KV 1024, 1459.2 MHz). Compilation, input seeding and
output comparisons are outside these timers. This does not establish accepted
FP32-model equivalence or a continuous-generation release. One initial RPC
failure remains unexplained; the identical library subsequently passed ABBA,
and the archive above passed a separate full-forward check. Keep this option
experimental until that failure and full-model acceptance are resolved.

## Experimental HMX integration and measurement

`attention.py` is an object-level experiment, not a model release. Its object
requires HexKL micro, the compiler's owner-thread HMX dispatcher and a caller
that launches eight programs across the HVX workers. Do not select it as a
precision-equivalent replacement for strict FP32 attention.

Before the first runtime allocation, a caller may explicitly select:

```cpp
hexagon_runtime_configure_vtcm_pool_dsp(1024 * 1024);
hexagon_runtime_configure_hmx_performance_dsp(1);
```

Check both return values. The performance API defaults to platform policy and
freezes on runtime initialization. The opt-in vote requests maximum core/bus
performance on the runtime's own HMX power client and is released with that
client. It is separate from numerical policy and may increase power use.
The 1 MiB VTCM reservation was used alongside the existing 3 MiB INT8 DMA
reservation on QCS9075; it is not a universal device capacity assumption.

On the measured QCS9075, adding an HMX client without its own performance vote
changed the reported DSP clock from 1459.2 MHz to 921.6 MHz. Matching the core/bus
vote restored 1459.2 MHz and reduced the experimental q32/KV1024 model forward
from about 1.17 s to 0.78 s, with identical outputs before/after the vote.
Those outputs still differ from the accepted FP32 model. The strict C control
with identical graph/INT8 objects took about 2.31 s at 1459.2 MHz.

Always report query length, effective KV length, KV dtype/layout, actual DSP
clock, warm/cold status and timing boundary. Native llama.cpp FlashAttention
uses persistent FP16 KV; comparing its operator time with a path that converts
FP32 KV during attention is not an identical-storage comparison. Worker-stage
time sums also are not wall-clock durations. A bitwise single-operator result,
a full-model numerical comparison and continuous generation are separate checks.

## Lightweight profiling

Add `--lwp --lwp-loop-depth 1` for a diagnostic object and per-build site map.
Use `python -m hexagon_runtime.lwp` to validate and summarize exported records.
The upstream buffer is not thread-safe: serialize execution or instrument only
one program instance. See [LWP setup, limits and measured evidence](../../../docs/qwen3/LWP.md).

## Bounded exponential in experimental attention

`attention.py::_softmax_exp` accepts FP32 inputs and retains the experimental
polynomial, two-part ln(2) reduction, `x < -80` cutoff and canonical-NaN policy
for positive/NaN arguments. It does not replace the strict FP32 exponential.

The implementation classifies the sign and magnitude bits once. In-range
values enter the unchanged `x * log2(e) + 0.5` expression, whose result is in
`[-115, 0.5]`. Adding the FP32 bias `12582912` rounds to a nearby integer;
subtracting that bias recovers the integer as FP32, while subtracting its bit
pattern recovers the integer representation. Subtracting one when the rounded
value exceeds the original gives the same floor as the former conversion-based
implementation. All these integer values are exactly representable. Keep
per-operation FP32 rounding; reassociation would invalidate this method.

This removes repeated numeric float/integer conversions and repeated floating
classification from the HVX code. It does not change polynomial coefficients,
KV storage, HMX math, reduction order or the number of workers. Signed zeros,
negative infinity, and positive/negative NaNs keep the prior output policy.
Performance and board validation are recorded in the LWP document linked above;
equivalence to the previous experimental policy is not acceptance against the
independent FP32 model.

## Strict short-KV Triton prefill

`strict_prefill.py` contains four ordered FP32 stages: packing, QK, softmax and
PV. Compile the `strict-prefill-pack`, `strict-prefill-qk`,
`strict-prefill-softmax` and `strict-prefill-pv` profiles with the same compiler
arguments shown above. They use the maintained FMA scheduling and bounded loop
rotation automatically; no external LLVM rewrite is needed.

The adapter admits 16–32 query tokens, effective KV length up to 256, and
`query_length * kv_length >= 1024`. Smaller tiles retain the reference path
because packing and dispatch overhead can outweigh the vectorized stages.
Packing checks that
all Q/K/V magnitudes are strictly below2^40; unsupported shapes and nonfinite or
out-of-bound inputs use the existing strict reference. Under that bound, future
causal scores round to exactly-FLT_MAX and their probabilities are zero. The
kernels skip complete future QK tiles and the zero-probability PV suffix while
retaining the reference FMA order. KV<=256 uses one original PV accumulation
block; its final addition of zero is retained for signed-zero semantics.
Scratch is131104 FP32 elements, plus separate K/V packing buffers sized for256
positions. All newly introduced attention arithmetic is Triton; the C++ adapter
validates descriptors, owns buffers and dispatches workers. After packing,
one four-worker dispatch runs QK, softmax and PV with the same heads assigned
to each worker. Ordinary DSP memory is coherent between scalar and HVX accesses;
worker completion provides synchronization between stages. RPC and DMA cache
maintenance remain at their respective boundaries.

The resident plan can select these objects using `triton_prefill`, a mapping
with `pack`, `qk`, `softmax` and `pv`. Each entry contains `object` (absolute
path), `object_sha256` and `manifest_sha256`, exactly like `triton_decode`.
Require a strict decode object and fused initial/prefill domains. The normal
CMake/build-model path verifies the hashes and ABI bounds. It excludes the
experimental FP16 prefill configuration. Omitting the mapping preserves the
existing strict C++ prefill route.

For strict SiLU, the resident plan also accepts `triton_silu` with the same
three identity fields, provided `parallel_silu` is true. Use the `strict-silu`
profile with `--schedule-fma-chains`. This preserves the original FP32 exponential
and division results; it does not enable the experimental quantizer-specific product.

For controlled benchmarks, `benchmark_max_performance: true` explicitly requests
maximum core/bus performance before the timed model interval and releases the
vote at unload. It defaults to false, leaving power control to the caller.
This is not a guaranteed clock: record the observed frequency and platform
limits in comparisons. Kernel objects/build receipts alone do not establish
whole-model numerical or performance acceptance.
