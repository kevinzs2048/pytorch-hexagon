# Building Triton attention and SiLU

This optional path replaces q1 attention arithmetic with
[`strict_decode.py`](../../models/qwen3/kernels/strict_decode.py).
The model graph, W8A8 policy and FP32 KV storage stay the same. Without the optional selections below, q2–32 prefill
continues to use the existing ordered FP32 implementation. The public builder
also supports explicit approximate HMX prefill and FP32 SiLU profiles.

The kernel packs one 8-token K tile immediately before QK, then packs V in
32-feature tiles immediately before PV. Each output retains its original FMA
and softmax reduction order. Four workers use 592 KiB of temporary storage in
total. This temporary packing workspace is separate from the persistent KV cache.

## 1. Compile the kernel

Use the compiler and LLVM source build described in [USAGE](../../USAGE.md).
The toolchain must include the predicate truncation correction documented as
HEX-PRED-003 in [LLVM issues](LLVM_ISSUES.md); an older installed compiler is
not interchangeable with the compiler used in the performance measurements.
Supply your actual tool paths and use a new output directory:

```bash
"$PH_PYTHON" -m models.qwen3.kernels.compile strict-decode \
  --libtriton "$PH_BUILD/compiler/profiles/dynamic-kv/libtriton.so" \
  --triton-shared-opt /absolute/path/to/triton-shared-opt \
  --linalg-hexagon-opt "$PH_BUILD/compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt" \
  --backend-source "$PH_REPO/third_party/hexagon-mlir" \
  --schedule-fma-chains \
  --output-dir "$PH_RUN/strict-decode"
```

The directory contains `kernel.o`, its source/tool hashes in `manifest.json`,
and the intermediate IR and stage logs. A successful compilation does not
constitute numerical acceptance on the board. Do not pass an LWP-instrumented
object to the normal four-worker runtime; that format requires serialized
profiling and the build rejects it.

## 2. Select it in the existing captured-session plan

Start with the optimized resident plan from USAGE. Its `decode` domain must
already select `attention="fp32-ordered-workers-v1"`. Add the following root
entry, using the hashes of the files produced above:

```json
"triton_decode": {
  "object": "/absolute/path/to/strict-decode/kernel.o",
  "object_sha256": "SHA256_OF_KERNEL_OBJECT",
  "manifest_sha256": "SHA256_OF_KERNEL_MANIFEST"
}
```

Obtain the two hashes with `sha256sum "$PH_RUN/strict-decode/kernel.o"
"$PH_RUN/strict-decode/manifest.json"`. Run the same
`models.qwen3.build_model --resident-spec ... --llama-source ...` command in
USAGE; add `--compile-resident` only when the graph domains require compilation.
No graph recompilation is needed solely to replace this ABI-compatible kernel.

The measured q1 graph also uses the existing batched ARM RoPE preparation
(`"batch_arm_rope": true` at the plan root) and these options inside its
`domains.decode` entry:

```json
"compiler_options": {
  "enableRowMinMax": true,
  "scheduleDequantLoops": true,
  "scheduleReductionLoops": true
}
```

Unlike replacing the attention object, changing these graph settings requires
recompilation. `enableRowMinMax` overrides the plan's `row_minmax` default only
for this domain. Scheduling options default to false. They retain the existing
FP32 reduction policy; no fast-math setting is enabled. The session manifest
records the effective options for every domain.

The builder checks both hashes before linking and again at completion. CMake
checks the object receipt, profile, four-worker launch grid and scratch bound;
it builds the adapter with the recorded allocation size. The emitted
`resident-plan.json` preserves the selection when graph domains are compiled,
and `session.json` records the selected kernel identity. Numerical qualification
remains a separate board step, as for the existing model builder.

## 3. Validate the linked session

Use the numerical and generation validation steps in USAGE with the new session.
Compare logits, updated KV and generated tokens against the same W8A8 reference.
Measure warm execution separately from compilation, loading and equality checks.
Keep the previous accepted session for rollback. Removing `triton_decode` from
the input plan restores the ordered C++ attention build on the next link.

## 4. Optional fast prefill and SiLU

These are explicit numerical-policy changes, not bitwise replacements. HMX
prefill uses FP16 matrix operands/results with FP32 online softmax state and
FP32 persistent KV. Decode retains the strict FP32 attention kernel. Fast SiLU
uses a bounded FP32 exponential polynomial and three reciprocal refinements;
exceptional vectors use the strict implementation. Generated tokens can change
because W8A8 activation quantization amplifies small differences.

Compile the two objects with the same toolchain used above:

```bash
"$PH_PYTHON" -m models.qwen3.kernels.compile experimental-prefill-fp16 \
  --libtriton "$PH_BUILD/compiler/profiles/dynamic-kv/libtriton.so" \
  --triton-shared-opt /absolute/path/to/triton-shared-opt \
  --linalg-hexagon-opt "$PH_BUILD/compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt" \
  --backend-source "$PH_REPO/third_party/hexagon-mlir" \
  --reuse-hmx-workspace --prepare-hmx-matmul --external-allocator \
  --output-dir "$PH_RUN/hmx-prefill"

"$PH_PYTHON" -m models.qwen3.kernels.compile experimental-silu-fp32 \
  --libtriton "$PH_BUILD/compiler/profiles/dynamic-kv/libtriton.so" \
  --triton-shared-opt /absolute/path/to/triton-shared-opt \
  --linalg-hexagon-opt "$PH_BUILD/compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt" \
  --backend-source "$PH_REPO/third_party/hexagon-mlir" \
  --schedule-fma-chains --output-dir "$PH_RUN/fast-silu"
```

Starting from a plan with strict decode, prepacked INT8 HMX projections and
`initial`/`prefill` fused attention already enabled:

- Replace `triton_prefill`, if present, with `triton_hmx_prefill` using the
  object/manifest hash format from section 2. Point it at `hmx-prefill/kernel.o`.
- Set `parallel_silu` to `true` and `triton_silu` to the same three-field object
  record, pointing at `fast-silu/kernel.o`. SiLU can also be selected independently.
- Run the normal source build after updating native sources, then the public
  `models.qwen3.build_model --resident-spec ... --llama-source ...` command.
  The builder selects the shared HMX owner for both projections and attention;
  do not combine independently cached HMX leases by hand.
- Run `hexagon_runtime qwen3 generate --resident-execution compiled
  --resident-session /absolute/path/to/new-session ...` as in USAGE.

The session records both precision policies. A successful link deliberately
leaves `model_quality_accepted=false`; it does not turn a compilation result
into a model-quality claim. For this path, compare probability distributions
and fixed-text loss against the original unquantized model as well as the W8A8
reference. Keep task-quality checks separate from timing. Strict argmax or KV
byte equality is not an acceptance criterion for a changed precision policy.

Fast SiLU admits zero and `2^-64 <= abs(x) <= 16`. Rejected vectors are repaired
before returning. Input/output overlap is rejected before writing, because the
repair pass needs the original input. The four-worker ABI, padded row strides,
and FP32 output shape are unchanged. The measured ordinary-domain error check
is `abs(error) / max(1, abs(strict_output)) <= 2e-6`; it is a device test bound,
not a universal proof of end-to-end model quality.

For rollback, run the previous session directory. To rebuild with strict
kernels, restore the previous `triton_prefill` selection and strict `triton_silu`
object hashes in the plan. Keep the fast and strict session directories separate.

## 5. SiLU aligned with the pinned llama.cpp Hexagon build

Use `llama-silu-fp32` to select the maintained Triton translation of llama.cpp
`f818065d753700341b5355ce36ca676ecfb5d139` SiLU. This is distinct from the older
`experimental-silu-fp32` polynomial. It follows the SDK-generated QFloat
reciprocal refinement, the ±87 sigmoid guards, and uses no strict-reference
rounding repair. It does not change the model's weight/activation quantization
format or claim whole-model equivalence with llama.cpp.

```bash
"$PH_PYTHON" -m models.qwen3.kernels.compile llama-silu-fp32 \
  --libtriton "$PH_BUILD/compiler/profiles/dynamic-kv/libtriton.so" \
  --triton-shared-opt /absolute/path/to/triton-shared-opt \
  --linalg-hexagon-opt "$PH_BUILD/compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt" \
  --backend-source "$PH_REPO/third_party/hexagon-mlir" \
  --schedule-fma-chains --output-dir "$PH_RUN/llama-silu"
```

The Hexagon tool must include packed HVX lowering in `--lower-libdevice`.
The command enables QFloat for this object only. Supply its `kernel.o`, object
SHA256 and `manifest.json` SHA256 in the resident plan's existing `triton_silu`
field, with `parallel_silu=true`. The model builder records the separate
numerical policy. Existing strict and previous fast artifacts remain usable.

If the compiler source/build identity changes, rebuild the compiler and
recompile the model objects before using the normal source-build flow; the
builder intentionally rejects mixing unidentified compiler products. The
2026-11-09 device experiment instead reused the previously sealed forward
objects and relinked only the SiLU archive, with explicit derivation receipts.
It is not evidence of a fresh full source rebuild.

See [alignment evidence](reports/2026-11-09-llama-silu/REPORT.zh-CN.md) and
[the compiler finding](LLVM_ISSUES.md#qfloat-conversion-folding-differs-between-sdk-and-open-llvm).
