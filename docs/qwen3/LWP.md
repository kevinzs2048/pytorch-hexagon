# Hexagon-MLIR Lightweight Profiling (LWP)

LWP instruments generated functions and loops with Hexagon cycle-counter reads.
Use it to locate time inside an operator. An instrumented kernel is a diagnostic
artifact; keep a matching uninstrumented control for performance reporting.

## Build and analyze

Use the regular kernel build command in [the kernel guide](../../models/qwen3/kernels/README.md),
adding `--lwp --lwp-loop-depth 1`. The selected compiler must include the LWP site
preservation fixes in this checkout. No LLVM source change is required.

```bash
python -m models.qwen3.kernels.compile experimental-prefill-fp16 \
  --libtriton "$compiler_library" \
  --triton-shared-opt "$triton_shared_opt" \
  --linalg-hexagon-opt "$linalg_hexagon_opt" \
  --output-dir "$kernel_output/lwp" \
  --reuse-hmx-workspace --lwp --lwp-loop-depth 1
```

The output directory contains `kernel.o`, `kernel.mlir`, `compile.log`,
`lwp-map.txt` and build manifests. The map associates each site ID with lines in
that build's `kernel.mlir`; do not reuse a map from another compilation.
`compile.log` includes the instrumented IR. Depth zero records outer loops;
greater depth adds eligible nested sibling loops and can increase overhead.

The object requires the backend's `bin/runtime/profiler/lwp_handler.S` and its
`lwp_counter`, `lwp_buffer`, `lwp_buffer_size`, and `lwp_buffer_count` storage.
The upstream example storage/exporter is `test/utils/dsp/include/prof_utils.cc`.
The normal model launcher does **not** automatically provide these symbols or
collect records. Use a diagnostic adapter, reset counters before each capture,
and export the buffer after execution, outside the operator timer.

The upstream record format is:

```json
{"entries": [{"id": 1, "cyc": 100}, {"id": 1, "cyc": 200}]}
```

Analyze an exported capture with the maintained entry:

```bash
python -m hexagon_runtime.lwp capture.json \
  --map "$kernel_output/lwp/lwp-map.txt" --output capture.csv
```

The analyzer rejects duplicate map IDs, decreasing timestamps, interleaved
sites and unclosed regions. It emits inclusive and exclusive cycles, call
counts, parent IDs and IR locations. Inclusive child/parent durations overlap;
use exclusive cycles for a non-overlapping breakdown of recorded regions.
Uninstrumented work remains in the enclosing region's exclusive time.

## Timer units

Check the assembly handler used in each artifact before converting records to
milliseconds. The upstream handler reads `C15:14` (processor cycles). Our
whole-model diagnostic handler reads `C31:30` (the 19.2 MHz global timer) so its
intervals can be reconciled with `HAP_perf_get_time_us`. These units differ;
a processor-cycle capture must not be divided by 19.2 MHz. Keep raw counters
and the measured clock in the result. Serial operator diagnostics do not give
the throughput of a four-worker launcher.

## Concurrency and capacity

The upstream handler has shared counters and a shared buffer, no thread IDs,
and no synchronization. Do not enable it for several concurrent program
instances. The supported diagnostic approaches used here are:

1. Run the complete grid serially and compare against a serial control.
2. Keep four workers, but instrument **only one program instance**. Link the
   instrumented attention functions under distinct names while sharing the
   existing HMX dispatcher/runtime. Other instances execute the uninstrumented
   object. The selected caller blocks while its instrumented HMX callback runs,
   so its records remain serial. This measures one program in concurrent
   execution; it is not a simultaneous trace of all threads.

The upstream handler caps each ID at 200 records. A long-KV serial capture in
our experiment needed 512 records for one ID. The diagnostic handler raises the
cap to 20,000 with a separate instruction packet for the extended immediate;
it retains the upstream record format. Buffer size remains 600,000 words.
Reject captures unless `sum(lwp_counter) == lwp_buffer_count / 3`, capacities
were not exceeded, and all regions close. Pairing alone cannot detect dropped
complete start/end pairs.

The build entry serializes MLIR compilation for LWP because the upstream pass
mutates module-level symbols and global IDs. It also locks access to the shared
`/tmp/lwp_infodump.txt`, copies the map into the output, then restores its prior
contents. Other build entry points must honor the same lock and serialization.

## Verified QCS9075 evidence, 2026-11-06

Evidence directory on this development machine:

`/home/kevin/hexagon/pytorch-hexagon/.local/lwp-attention-20261106`

- `summary.json`: paired timings and output hashes.
- `lwp-result.json`: full serial traces; 4,512 records for q32/KV1024.
- `sampled-result.json`: four-worker runs with only KV-head program 0 traced;
  564 records for q32/KV1024. All eight programs still compute the full output.
- `sampled-q32-kv1024.csv`: validated cycle breakdown for one warm sample.
- `maintained/kernel.mlir`, `maintained/lwp-map.txt`: source mapping.
- `build_probe.py`, `build_sampled.py`, `link-*.json`, `run_attention.py`:
  experiment-specific adapter/link/board recipes; not production model code.

The existing board probe can be repeated with a fresh label:

```bash
ssh radxa@192.168.31.135 \
  'timeout 90 /mnt/nvme/qwen3-work/arm-pytorch-r1362/venv/bin/python /mnt/nvme/qwen3-work/lwp-attention-20261106/run_attention.py sampled-b sampled'
```

Its `result.json` is written to
`/mnt/nvme/qwen3-work/lwp-attention-20261106/sampled-b/`.
The runner checks record counts/capacity and saves output hashes. Analyze the
`lwp_entries` of each sample separately; never concatenate reset captures.

For q32, effective KV=1024, warm median full attention was 5.678 ms without
instrumentation and 5.653 ms with selective instrumentation. This difference
is noise, not a speedup. Full serial instrumentation measured 12.894 ms versus
12.514 ms for its serial control (about 3% higher).
All 64 outputs across four configurations and the short/tail/long/reset cases
had identical per-input hashes. These are experimental FP16 attention results,
not a new FP32 model acceptance or end-to-end performance claim.

## Optimization decisions grounded in the paper

The [Hexagon-MLIR paper](https://arxiv.org/html/2602.19762v1) describes operator
fusion (§4.3), TCM tiling (§4.4), DMA double buffering (§4.6), and layout
propagation to reduce packing/unpacking (§4.8). The companion
[ablation study](https://arxiv.org/html/2602.20204v1) separates vectorization,
threading, and double-buffering effects.

In the measured attention artifact, `fusion=True`, but a separate 128×128 FP16
K transpose remains before HMX packing. Generic VTCM tiling, double buffering,
and HexagonMem-copy-to-DMA are disabled; HMX's own VTCM workspace is separate.
Therefore the next justified experiment is to eliminate that intermediate
layout conversion and measure its effect independently. The paper does not
establish that blindly enabling all passes is legal or faster for this IR.

In the four-worker selected-program trace, K transpose measured a median
800 us over its eight KV blocks, versus about 302 us combined for K/V loading
and FP16 conversion. Those numbers describe one program, not the complete
layer. They establish a concrete hotspot; PMU evidence is still needed to
separate bandwidth, cache and execution-resource contention.

### First layout experiment: standalone gain did not survive model integration

The 2026-11-06 follow-up retained the RHS transpose as a view and generated HVX
scatter directly into HMX layout, following the §4.8 hypothesis above. An initial
version scalarized its loads and regressed; exposing the physical contiguous
axis fixed that. Same-input, uninstrumented q32/KV1024 attention improved from
5.737 to 5.175 ms, with identical output bytes. However the 28-layer forward
measured 613.174 versus 625.442 ms (four warm samples per version); cumulative
attention was almost unchanged at 167.823 versus 167.197 ms.

Therefore this candidate was **not promoted**. Its edits were rolled back and a
reviewed patch, numerical evidence, LWP mapping and recipes retained in:

`/home/kevin/hexagon/pytorch-hexagon/.local/layout-fusion-20261106/REPORT.md`

The separate transpose site disappeared, but the selected program's enclosing
KV-loop residual increased; that region includes waits and uninstrumented work.
It does not isolate HMX queue time. The next attention experiment must separate
packing and owner-thread wait/execution, then verify the full-model result.
The same full-model trace also confirms ~91 ms in scalar SiLU; paper §4.7's
vector-math approach is another concrete target, subject to numerical checks.

### SiLU: vectorization admission fixed, scalar math remains

Following paper §4.7 (math lowering) and §5.1 (vectorization), the next
experiment found a concrete compiler restriction: intermediate `i1` predicates
made the nominal tile width 128, so a complete 32-lane FP32 parallel tile was
rejected. The compiler now admits compatible parallel subvectors; reduction
loops retain their previous restriction. `models/qwen3/kernels/silu.py` reuses
the compatible exponent from strict decode and keeps strict division.

A four-worker probe instruments only program 0. All 20 captures passed record
validation and had bit-identical outputs. For one warm 32×3072 sample, the full
operator took 1747 us; the selected program's compute loop took 1670.702 us,
its enclosing function 1698.359 us. These nested timings must not be added or
multiplied by four. LWP does not separate the fused exponent and division.
Disassembly independently shows scalarized strict FMA/division in otherwise
vectorized code. This points to math lowering as a remaining candidate, not
proof that a particular arithmetic instruction accounts for all elapsed time.

Uninstrumented standalone SiLU improved 4245→1741 us. In the same experimental
FP16-attention full model, cumulative SiLU improved 90.929→38.527 ms and forward
601.952→558.374 ms. The maintained archive passed a separate forward check;
scalar default object bytes are unchanged. One initial RPC failure remains
unexplained, so this is an opt-in candidate, not a new accepted model release.
See the kernel guide for its build commands. Raw data and the focused patch:

`/home/kevin/hexagon/pytorch-hexagon/.local/silu-vector-20261106/REPORT.md`

### Attention: separate owner execution from caller waits

The next capture isolates the HMX owner rather than instrumenting one complete
attention program. Keep depth-2 LWP sites only in the two generated
`__hexagon_matmul*.hmx_impl` functions. All callbacks execute on one owner, so
four HVX workers can remain active without concurrent writers to the LWP buffer.
The per-build map is still required. Verify that all exported IDs belong to
those functions, counts are complete, nesting is valid and outputs match.

A separate diagnostic wrapper records submission, callback start/end and return
for every dispatch in disjoint slots. It redirects actual LLVM call sites;
linker `--wrap` alone did not intercept same-object calls in the earlier probe.
Its two site IDs identify M32/M64 specializations, not QK/PV. Non-overlapping
owner intervals and expected call counts are checked before analysis.

At q32/KV1024, the matched uninstrumented attention took 5722 us; the dispatch
trace took 5723 us. Across 128 callbacks, owner execution totaled 3073 us.
Between owner intervals, 1622 us had no submitted task pending and 395 us had
pending work. These exclude time before the first and after the last callback.
Caller wait sums overlap other work and must not be added to wall time.

The separate owner-only LWP capture measured 5781 us wall time. Its cumulative
regions show about 514 us LHS packing, 1383 us RHS packing plus multiplication,
and 729 us accumulator deinterleave/output stores. These are instrumented
regions, not pure HMX instruction times. A full-tile output-mask candidate
removed redundant masks but gave no standalone gain and only a marginal model
change; its source edits were reverted. Repeated RHS packing across M tiles is
the next target, informed by paper §4.4 and llama.cpp's reused tile layouts.

Raw records, per-build map, analysis, exact recipes and rejected patch:

`/home/kevin/hexagon/pytorch-hexagon/.local/hmx-dispatch-profile-20261106/REPORT.md`

### Revisit: RHS reuse reduced owner work but did not improve the model

The follow-up hoisted RHS packing across M tiles under a 256 KiB workspace
budget. For 64×128×128, RHS tile packs decreased from 32 to 16 and workspace
increased from 52 to 96 KiB. Owner-only LWP fell from about 3.29 to 2.82 ms in
separate instrumented captures. However standalone attention was 5.779 versus
5.916 ms and complete forward 555.336 versus 568.847 ms. The candidate was
reverted. A small worker-count comparison found four workers faster than two
or three for this workload.

Combining RHS reuse with the earlier K-transpose fusion improved standalone
attention to 5.331 ms with identical outputs, but its full-model ABBA run failed
with a DSP/RPC error. That combination is also reverted. The same error code
occurred in a three-worker control without RHS reuse; a subsequent unchanged
retry passed, which does not explain or resolve the failure. Disk/shared-memory
capacity checks and static stack-frame inspection did not identify its cause.

These measurements reject the assumption that reducing owner execution alone
will reduce complete-forward latency. Next steps are DSP exception capture and
measuring overlap/critical-path behavior before adding further transformations.
All candidate edits, raw results, failure logs and rollback hashes are retained:

`/home/kevin/hexagon/pytorch-hexagon/.local/hmx-rhs-reuse-20261106/REPORT.md`

### DSP exceptions: consult the service log before interpreting timings

Three intermittent failures have exception traces in `journalctl -u cdsprpcd`.
Exact linked binaries place their attention frames beside `free()` calls. This
narrows the investigation to deallocation/heap integrity; it does not establish
a particular memory bug. An instrumented allocator passed 48 standalone calls
and, after fixing its own bookkeeping, 13 complete forwards, all matching the
experimental baseline payloads. Allocation instrumentation can mask the original
failure, which remains unresolved. These runs are diagnostic, not speed evidence.

The investigation also records two failures caused by diagnostic limitations;
they must not be presented as additional original-model failures. Raw evidence,
bounded recipes, coverage limits and the paper-guided next step are in:

`/home/kevin/hexagon/pytorch-hexagon/.local/dsp-exception-20261106/REPORT.md`

### Caller/owner timeline: K transpose dominates no-pending gaps

A follow-up replaces selected compiler LWP sites with thread-safe records tagged
by thread, using the same19.2 MHz qtimer as dispatch timestamps. Four callers
and the HMX owner are captured together. Per-thread nesting,8 program roots,
expected callback counts, owner serialization and output hashes all validate.
This avoids concurrent writes to the stock LWP buffer.

For q32/KV1024, control/timeline are5.628/6.044 ms (+7.4% instrumentation impact).
Between callbacks,1.667 ms has no HMX task pending;1.419 ms overlaps K transpose
on the first subsequently submitting worker. This is temporal attribution, not
pure instruction time or a guaranteed removable amount. All16 captures match
control outputs. Local larger-tile and shuffle rewrites provided no accepted
speedup; a32-tile experiment exceeded the worker stack and was rejected.

The next target is layout preparation placement/overlap, following paper§4.4–4.6,
§4.8,§5.4 and llama.cpp's interleave/HMX pipeline. Avoid repeatedly tuning only
transpose tile size. Detailed measurements, limits and rejected recipes:

`/home/kevin/hexagon/pytorch-hexagon/.local/attention-timeline-20261106/REPORT.md`

### Caller preparation plus fused K layout: first model-level gain

A generated-IR prototype moves full input packing to independent caller VTCM
leases and fuses K transpose into HMX layout writes. The sole HMX owner performs
matrix arithmetic and accumulator output work; submission remains synchronous.
Allocation/resizing is also owner-mediated because the existing buffer manager
is not concurrent and initial resource acquisition binds HMX to its caller.

At q32/effective KV1024, standalone attention falls5.783→4.485 ms. Full28-layer
ABBA measures cumulative attention169.096→144.216 ms and complete forward
567.870→543.282 ms. All12 forwards including warmups match the experimental
logits/new-KV hash. This retains experimental FP16 attention, is not FP32-model
acceptance, and has not been integrated into the maintained compiler/default path.
Preparation-only without K-layout fusion regressed, confirming pass interaction.

Same-clock preparation/owner traces validate overlap and about1.27 ms owner work.
Caller LWP-site traces show K transpose gone; remaining no-pending gaps involve
conversion, softmax, preparation and unclassified loop work. Also, about275.7 ms
of complete forward lies between the currently instrumented model operators;
further attribution is needed before calling that framework overhead.

Exact conditions, numerical limits, lease design, integration work and raw data:

`/home/kevin/hexagon/pytorch-hexagon/.local/hmx-preparation-20261106/REPORT.md`

### Complete-model follow-up: allocation and quantization

The fused-attention model's serial graph LWP capture now checks the exported
invocation total, not just balanced pairs. Its first capture hit the old200-entry
cap; the corrected capture retains30,028 records with a maximum2,702/site and
no dropped entries. External callee times are subtracted on the same global
clock, so model-root residuals do not double-count attention/matmul/SiLU.

On q32 with seeded past992, loop-level diagnostics identify about48.2 ms in
minimum/maximum reductions and33.2 ms in RoPE sin/cos. Separate wrappers limited
to graph-object references measure1,413 malloc calls at about78.1 ms, requesting
100.2 MB total; frees begin after model computation. This is temporary-buffer
allocation evidence, not proof of KV-copy or framework/RPC overhead. Helper
timings overlap loop timings and must not be added together.

Evidence and replay commands are in
`.local/hmx-preparation-20261106/REPORT.md`, `model-lwp-v2-result.json`,
`model-lwp-analysis.json` and `model-helpers-summary.json`. These remain diagnostic
artifacts using experimental FP16 attention; output hashes match that same-policy
control, not an accepted FP32 release.

A subsequent generated-IR experiment accumulates min/max order keys over a full
1024-element row before horizontal reduction.112 sites replace3,584 static
horizontal reductions with112. A matched-codegen model ABBA measures548.890 to
532.545 ms (2.98%), with all12 forwards matching the experimental output hash.
Only7.366 ms is the reduction in the surrounding non-callee intervals; unchanged
attention timing also varies, so the full16.345 ms must not be attributed to
min/max. This is a prototype awaiting compiler integration, not a release or a
standalone-operator benchmark. See `row-minmax-summary.json` in the same evidence
directory.

The next isolated experiment redirects only the original model graph's malloc/
free references to a bounded cache. It reuses buffers only after explicit graph
frees and detaches still-live results at forward completion. Matched ABBA improves
warm full forward536.188 to461.579 ms (13.91%); non-callee intervals274.026 to
205.468 ms. All12 forwards match the experimental control.1,356 of1,413 allocation
requests reuse storage, with92.2 MB retained; the first forward still allocates.

A query32/2/17/32/2 sequence also matches all control outputs. It exposes metadata
slot exhaustion across sizes with safe fallback, so the diagnostic cache is not
the final general/session allocator. The same evidence directory contains
`graph-cache-summary.json`, `model-graph-cache-shapes-result.json` and the replay
recipes. Runtime ownership integration, bounded mixed-shape allocation policy,
accepted-precision regression and intermittent-exception diagnosis remain open.

### Current compiler row min/max experiment

With graph/worker pools and maintained prepared attention enabled, the latest
whole-model LWP still measured about50.4 ms in minimum/maximum reductions.
The new compiler option `enableRowMinMax=True` (MLIR pass spelling
`enable-row-minmax=true`) accumulates full32-lane vectors before one horizontal
reduction per row. Dynamic token/outer dimensions are allowed; reduction widths
must be static multiples of32, at least64. IEEE FP32 min/max and arbitrary seed
are preserved; coupled argmin/argmax, fused bodies and tails fall back. The
option defaults off and is available through the compiler driver’s `--option`.

This follows llama.cpp's vector-accumulate-then-reduce scheduling in
`ggml/src/ggml-hexagon/htp/hvx-reduce.h`, implemented in MLIR rather than a C
operator. Matched compiler builds differ only in this flag. On QCS9075,
q32/past992, full28-layer warm forward falls459.568→414.927 ms (9.71%);
INT8, attention and SiLU call times remain approximately unchanged. The changed
surrounding intervals decrease214.858→168.268 ms. These are uninstrumented
performance runs, excluding compilation/loading/seed transfers and comparisons.

All12 profiling forwards match the same experimental logits/new-KV output;
a separate normal resident entry matches16 forwards and4 full KV snapshots.
Host execution of the actual MLIR rewrite/LLVM lowering checks572 cases for
finite values, signed zero, infinities, subnormals, NaNs and non-default seeds.
Experimental FP16 prefill is still not accepted FP32-equivalent; independent
FP32 logits maxabs1.238635 remains. Continuous-generation acceptance is separate.

Raw measurements and matched build commands are in
`.local/hmx-preparation-20261106/maintained-row-summary.json`,
`maintained-row-profile-v1-result.json`, `maintained-row-entry-v1-result.json`
and `REPORT.md`. Earlier prototype timings above describe different artifacts.

## LLVM code-generation findings

The strict SiLU FMA-chain investigation, reduced reproducer, alternate compiler
checks and numerical/performance limits are recorded in [LLVM findings](LLVM_ISSUES.md).
The scheduling candidate is diagnostic and has not been integrated into model
builds. LWP loop time alone does not establish per-instruction cycle attribution.

The parallel sampled-attention capture initially linked its extracted function
against a slow compatibility copy implementation. This diagnostic defect is
fixed; see [the attribution correction](LLVM_ISSUES.md#excluded-attribution-sampled-attention-copy-dispatch).
Use `.local/attention-timeline-20261106/fixed-analysis.json`, not the original
`analysis.json`. The corrected trace covers KV-head program 0 only, using the
19.2 MHz global timer while four workers run. Its warm median elapsed time is
2.072 ms within a 4.530 ms instrumented operator invocation. These are selected
task latency and whole-operator latency, respectively; do not add or multiply
them to estimate total parallel work.

Within that selected program, warm median exclusive intervals include softmax
exponential processing (362 us), K/V loading and FP32-to-FP16 conversion
(161/142 us), and the two HMX result rearrangement sites (123/137 us).
Instrumentation and scheduling affect these intervals; they prioritize further
experiments and are not a per-instruction cost model or a matched llama.cpp
stage comparison. All eight corrected captures retain the prior experimental
outputs and pass record balance/capacity checks.

A separate integer-exponent reuse trial was bit-identical on 32 calls, but its
q32 median changed only 4.512 to 4.482 ms with overlapping samples. The source
change was reverted because this run did not establish a useful performance
gain. Evidence is retained in the same directory's `exponent-reuse-summary.json`.

### Follow-up experiments and a local tiling correction

The following ABBA tests use the same two saved cases, four workers and warm
1,459.2 MHz samples. Each row has its own paired baseline; compilation, loading,
RPC and correctness checks are outside the operator timer. Times below are for
q32/KV1024. None establishes a new deployed-model speedup.

| Experiment | Baseline | Candidate | Decision |
|---|---:|---:|---|
| Reshape only the softmax exponential into 32-lane groups | 4.508 ms | 19.187 ms | Reject; exposed scalar fallback |
| Same reshape with the tiling correction below | 4.528 ms | 4.720 ms | Vectorization restored, but reject kernel reshape |
| Allocate temporary buffers at 128-byte alignment | 4.525 ms | 4.489 ms | Overlapping samples; revert the global alignment change |
| Retain HMX outputs in VTCM and unpack on submitting workers | 4.512 ms | 5.063 ms | Reject scheduling/storage change |
| Declare proven full output transfers in bounds | 4.527 ms | 4.524 ms | No useful observed gain; revert |

All 160 calls across these paired experiments retain the experimental output
hashes. This is not a claim of FP32-model equivalence. The caller-unpack LWP
trace also preserves all eight outputs and has balanced, uncapped records.
Its selected task rises from about 2.072 to 2.216 ms; the two rearrangement
regions total about 350 us versus about 260 us in the original trace. The loop
nesting and instrumentation sites differ, so this supports rejecting the change
but does not isolate a particular hardware contention mechanism.

One narrowly scoped compiler fix remains: `HexagonTiling` now matches the local
vectorizer's existing admission of complete 32/64-lane parallel subvectors.
Previously, an i1 predicate selected a nominal 128-element tile; a smaller
innermost dimension caused tiling to skip the whole multidimensional expression,
and subsequent vectorization failed. Tiling the outer dimensions restores vector
FP32 arithmetic without altering per-element operations or reduction order.
This closes an inconsistency in our local compiler extensions, not a newly
proven upstream LLVM bug. Tests cover static/dynamic outer shapes and exclusion
of partial, smaller and reduction cases. Recompiling the original attention
with this correction alone produces the original object byte for byte.

The Triton attention source, HMX scheduling/output copy implementation and
buffer-alignment policy are restored to their pre-experiment versions. The
accepted compiler profile is unchanged. Source-build/deployment qualification
remains separate. Evidence and rejected variants are grouped under
`.local/attention-timeline-20261106/REPORT.md` and `experiment-audit.json`.

## RoPE result reuse, 2026-11-06

The earlier approximately 33 ms sin/cos attribution exposed repeated evaluation
of the same four-lane SLEEF group. An optional application policy now preserves
the same grouping and algorithm while returning all four lanes per call. It
uses destination-buffer helpers, without shared memoization or a new math
approximation. Enable with `batch_arm_rope: true` in a newly rebuilt resident
plan; the default remains false. See the [native runtime notes](../../models/qwen3/native/README.md).

Dynamic q32 standalone sin+cos falls from 16.632 to 4.185 ms. The matched
28-layer q32/past992 forward falls from 414.711 to 388.099 ms (6.42%), with
identical experimental logits/new-KV payloads. All 48 standalone calls and
12 full forwards pass; the normal resident comparison passes 16 forwards and
4 complete KV snapshots. This model test replaces the prefill graph; initial
and decode graphs retain their prior versions. The existing experimental FP16
attention and independent FP32 logits difference remain unchanged.

Both full builds use the same source compiler/options and runtime object.
Outer-callee profile intervals are measured without LWP instrumentation; the
26.699 ms reduction in other graph work is not an exclusive sin/cos measurement.
Detailed evidence: `.local/rope-batch-20261106/REPORT.md`, `model-summary.json`,
`operator-summary.json`, `entry-result.json`. This does not qualify a complete
new four-domain source build or continuous generation, and no default compiler
or accepted deployment was replaced.

## Paired HMX FP16 output stores, 2026-11-06

The prepared-matmul pass now combines adjacent 32-column FP16 accumulator
outputs into complete 64-element rows before storing them. The original single
32-column write lowered to several predicated short-vector stores. In the
inspected q32 QK helper, the change removes all 256 static predicated store
sites; the disassembly shrinks from 1,982 to 330 lines with no unknown
instructions. These are static properties, not cycle measurements.

The approach follows the dual-tile row reconstruction in llama.cpp's
`ggml/src/ggml-hexagon/htp/hmx-flash-attn-ops.c` at pinned commit
`f818065d753700341b5355ce36ca676ecfb5d139`. The implementation remains MLIR vector
loads/shuffles/stores. It neither embeds the C operator nor changes floating-point
operations, K accumulation order, worker assignment or VTCM size. Two 2 KiB
accumulator reads fit the existing 4 KiB output reservation. Only admitted static
full-tile helpers with FP16 output and N divisible by 64 use the pairing. FP32
output and odd counts of 32-column tiles retain the previous schedule.

Same-input ABBA, four workers at 1,459.2 MHz:

| Measurement | Before | After |
|---|---:|---:|
| q2/KV2 single attention | 364.5 us | 320.0 us |
| q32/KV1024 single attention | 4.533 ms | 4.014 ms |
| 28-layer q32/past992 forward | 388.421 ms | 376.330 ms |
| Attention calls within that forward | 123.517 ms | 111.454 ms |

The operator time decreases 11.45%; the measured full forward decreases 3.11%.
Both model variants use the same RoPE-batched graph and runtime, replacing only
the attention kernel archive member. All 32 operator comparison calls, 24 boundary
calls, eight sampled calls and 12 model forwards preserve the corresponding
experimental outputs. The normal resident entry also matches 16 forwards and
four full KV snapshots, including q1, q2, q17, q32 and request reset. Boundary
checks cover q3, q16, q17 and KV tails around the 128-column blocking boundary.
The final source kernel is byte-identical to the tested kernel object.

### Remaining cost and comparison limits

A corrected parallel LWP trace samples KV-head task 0 (two Q heads) while all
four workers run. Its warm median is 1.757 ms; the instrumented full operator is
4.046 ms. Exclusive intervals below are medians across three warm captures.
They are selected-task intervals, **not** an additive decomposition of the
uninstrumented 4.014 ms full operator:

| Selected-task region | Time |
|---|---:|
| Softmax exponential expression | 347 us |
| K/V loads and FP32-to-FP16 conversion | 304 us |
| HMX input packing and transpose | 218 us |
| FP32 online output accumulation | 145 us |
| Maximum/sum reductions | 112 us |
| HMX output rearrangement | 35 us |
| HMX multiply inner loops | 19 us |
| Remaining exclusive intervals | 573 us |

The original output rearrangement measured about 260 us in the corresponding
control trace. The multiply inner loops exclude packing, accumulator clearing
and readback, allocation, dispatch and waiting. Remaining intervals are not yet
fully separated into those costs in that capture. The follow-up below separates
the direct helper calls. Individual medians need not sum exactly to
the median root time. All captures pass balance, capacity and output checks.

A fresh invocation of the installed llama.cpp backend measures 1.919 ms warm
median on the same saved FP32 QKV source, token-major FP16 KV, four workers and
about 1,460 MHz. The observed ratio is now 2.09x, but this is still **not** an
equal-precision stage comparison: its FP16 KV conversion occurs outside the DSP
operator timer, while ours occurs inside; its attention arithmetic also differs.
No new native stage breakdown was measured. The llama.cpp source consumes S
and produces P in VTCM tile layouts, while our generated helpers still materialize
row-major intermediate tensors and repack them. This is a concrete next target;
the trace does not establish that the entire residual gap is an LLVM defect.

Evidence: `.local/attention-pair-output-20261106/REPORT.md`, `summary.json`,
`model-summary.json`, `profile-summary.json`, `native-comparison.json` and
`validation.json`. LWP uses 19.2 MHz C31 ticks; the full-model wrapper records
`HAP_perf_get_time_us()` directly. The accepted FP32 compiler/deployment is
unchanged. This remains experimental FP16 attention, with the prior independent
FP32-model logits difference, and is not continuous-generation acceptance.

### Direct helper calls and small-buffer experiment, 2026-11-06

A follow-up capture wraps direct allocation, release, copy, zero-fill and HMX
runtime calls in the same sampled task. All eight input/output hash pairs match
the preceding paired-output capture. Records remain balanced and uncapped.
The warm selected task measures 1.776 ms and the instrumented whole operator
4.054 ms. Added markers affect code layout and timing; these measurements are
not an exact subtraction from the earlier 573 us residual.

| Direct helper calls in the selected task | Calls | Exclusive time |
|---|---:|---:|
| Temporary allocation hook | 86 | 117.8 us |
| Temporary release hook | 86 | 29.5 us |
| LLVM memcpy | 26 | 89.1 us |
| Backend memrefCopy | 2 | 30.1 us |
| Zero-fill | 11 | 10.8 us |
| Prepared workspace acquisition | 16 | 7.8 us |
| Prepared workspace release | 16 | 7.8 us |
| Synchronous HMX dispatch/wait, excluding nested sampled work | 16 | 74.8 us |

Allocation hooks include temporary-pool reuse, so 86 calls do not mean 86 new
heap allocations. Dispatch includes 174.4 us when nested sampled HMX work is
included; do not add that inclusive number to the nested regions. These are
single-task intervals, not a whole-operator breakdown or native llama.cpp stage
comparison. Evidence: `.local/attention-pair-output-20261106/helper-profile/`
(`result.json`, `analysis.json`, `summary.json`, `validation.json`).

The existing `stackBufferBudget=4096` compiler option was then tested with the
same paired attention source. It removes six static allocation and release
call sites, but grows the main function frame from 8,832 to 10,752 bytes.
Same-input ABBA at 1,459.2 MHz gives q32/KV1024 **4.033 vs 4.053 ms**, with
overlapping sample ranges; q2/KV2 gives 318 vs 312 us. All 32 outputs match.
There is no demonstrated benefit for the target long attention, so this option
is not enabled in the maintained kernel build. Evidence and exact commands:
`.local/attention-pair-output-20261106/small-stack/`,
`build_small_stack.py` and `small-stack-link-commands.json`.

The next substantial target remains producer/consumer data layout: avoid
writing row-major S/P intermediates only to copy and repack them for HMX.
This requires preserving masking, online-softmax state and numerical behavior
through the Triton/MLIR lowering. The current evidence does not establish an
upstream LLVM bug as the cause of the remaining attention gap.

### Input-layout follow-up: two rejected changes, 2026-11-06

Two measured experiments narrow the next layout work. Neither is retained.

1. **Pair input tiles.** Read adjacent 32-column FP16 blocks with full HVX row
   vectors and interleave into two HMX tiles, following llama.cpp's FP16 Q-pack
   pattern. This reduces static memory instructions but q32/KV1024 ABBA is
   3.990 vs 3.964 ms (0.65%, overlapping samples); q2 is 313.5 vs 315 us.
   All 32 outputs match. Compiler build and the extended preparation check pass.
2. **Fuse P conversion into packing.** Sink the private same-block FP32-to-FP16
   probability conversion into the outlined PV helper, then convert during
   VTCM packing. Keep Q's loop-invariant conversion outside the KV loop.
   q32/KV1024 ABBA is 4.037 vs 3.996 ms (1.03%, overlapping samples); q2 is
   319 vs 325 us. All 32 uninstrumented and 16 sampled outputs match.

The second experiment has a fresh single-task LWP comparison. Baseline P cast
is only 13.0 us and P packing is 39.3 us; the fused region is 41.0 us. The
approximately 11.3 us local saving does not reduce the selected-task latency
(1.777 vs 1.779 ms). Online accumulator time changes from 142.6 to 157.1 us;
the trace does not establish why that unchanged region moves. Instrumented
full-operator times cannot establish throughput benefit over the overlapping
uninstrumented measurements. Avoid treating fewer intermediate tensors or
static instructions as proof of speedup.

The larger measured targets remain K/V load+conversion (307–320 us) and
softmax exponential computation (347–350 us). Direct K/V-load-to-HMX-layout
fusion must cross the current masked tile-load/helper boundary and preserve
runtime row masks, strides and FP16 rounding. Further P-cast-only optimization
has little headroom. Complete S/softmax/P layout integration is still unfinished.

Both experiments restore their source and test changes byte for byte. Evidence
is grouped under `.local/attention-pair-input-20261106/REPORT.md` and
`.local/attention-cast-pack-20261106/REPORT.md`; the latter includes mapped LWP
results in `profile-comparison.json`. No accepted compiler or deployment is
replaced, and no upstream LLVM defect is established by these experiments.

## Direct K/V loads into HMX packing, 2026-11-06

The prepared-HMX pass now recognizes the private masked FP32 tile-load and
FP16-conversion loops feeding an outlined matmul. Instead of materializing a
row-major FP16 K/V tile and reading it again for packing, the helper reads the
original strided FP32 source and converts while writing the HMX workspace.
The transformation proves a single writer/consumer, preserves positive-zero
padding and rejects intervening writes/frees that may alias the source.
Unknown calls block fusion. A helper proven from its body to write only its
own allocations may be crossed; this permits V fusion across QK.

A runtime row-count check selects unmasked packing for complete KV blocks and
masked packing for the final partial block. This distinction matters: the
initial all-masked K-only variant regressed long attention 3.984 to 4.096 ms;
the full-block path improved it 4.008 to 3.831 ms. Both K and V are fused in
the retained version. HMX math order, FP16 rounding, FP32 KV storage, workspace
quota and worker scheduling remain unchanged. The feature remains within the
existing optional `prepareHMXMatmul` path; no new deployment default is enabled.

Same-input ABBA, four workers at 1,459.2 MHz:

| Measurement | Prior paired-output baseline | Direct K/V packing |
|---|---:|---:|
| q2/KV2 attention | 321.5 us | 244.0 us |
| q32/KV1024 attention | 4.010 ms | 3.743 ms |
| 28-layer q32/past992 forward | 375.431 ms | 364.088 ms |
| Attention calls within that forward | 110.468 ms | 100.801 ms |

The long operator saves 6.66%; full forward saves 3.02%. The model comparison
replaces only the attention archive member in the same RoPE-batched graph.
Validation covers 32 final operator calls, 24 boundary calls, 12 full forwards
including warmups, and 16 normal resident forwards plus four full KV snapshots
across mixed query lengths and request reset. These retain the existing
experimental FP16 outputs. The final source-built object is byte-identical to
the tested candidate. This is not FP32-model or continuous-generation acceptance;
the independent FP32 logits max-absolute difference remains 1.238635.

The new single-task LWP warm median is 1.619 ms; the whole instrumented operator
is 3.825 ms. Against the prior paired-output capture, selected-task K
load/convert/pack falls from 252.5 to 196.7 us, V from 225.7 to 181.3 us.
Softmax exponential work remains 350.0 us. These are selected-task exclusive
intervals with instrumentation overhead, not a decomposition of the uninstrumented
3.743 ms wall time. Some unchanged regions also shift with code placement.

The new full/tail branches exposed a profiling compilation defect: MLIR merged
different exit markers and created a variable site ID. A per-site call attribute
now prevents that merge; see [LWP-CFG-002](LLVM_ISSUES.md#lwp-cfg-002-branch-site-ids-merged-before-llvm-optimization).
The uninstrumented kernel is unchanged by this fix. Eight corrected captures
preserve outputs and pass record-balance/capacity checks.

A fresh installed llama.cpp invocation measures 1.879 ms (about 1,460 MHz),
an observed ratio of 1.99x. Its FP16 KV conversion is outside the DSP timer,
ours remains inside, and arithmetic differs. Do not present this as an
equal-precision speed ratio or an LLVM-only performance gap.

Evidence and exact build/link commands:
`.local/attention-kv-load-pack-20261106/REPORT.md`, `summary-kv.json`,
`model-summary.json`, `profile-summary.json`, `native-comparison.json`, and
`validation.json`. The accepted compiler/deployment is unchanged. The next
large remaining measured region is the softmax exponential expression.

## Bounded softmax exponent conversion and classification (2026-11-06)

The 350 us selected-task exponential interval above motivated disassembly of
that exact LWP site. Repeated FP32/integer conversions and floating-point
classification produce long vector instruction sequences and temporary
stores/loads. The native reference's `htp/hvx-floor.h` handles conversion with
bit operations; its complete exponential has different arithmetic. Rather than
adopt those coefficients, the Triton helper now uses its existing bounded domain
to remove numeric conversions and classify FP32 input bits once.

`models/qwen3/kernels/attention.py::_softmax_exp` retains the polynomial,
two-part ln(2) reduction, exceptional-input policy and FP32 arithmetic order.
The bias-and-correction derivation and its no-reassociation requirement are in
the [kernel README](../../models/qwen3/kernels/README.md#bounded-exponential-in-experimental-attention).
This change is in the application-owned Triton kernel; no new LLVM or compiler
source change is involved.

Same-input warm ABBA, four workers, 1459.2 MHz:

| Measurement | Direct-K/V baseline | Bounded conversion + classification |
|---|---:|---:|
| Attention q2/KV2 | 246.5 us | 214.5 us |
| Attention q32/KV1024 | 3719.0 us | 3321.5 us |
| 28-layer q32/past992 forward | 364.893 ms | 351.557 ms |
| Attention calls within that forward | 101.673 ms | 88.291 ms |

Long attention time falls 10.69%; full forward falls 3.65%. Only the attention
archive member changes in the model comparison. Compilation, loading, input
seeding and correctness comparisons remain outside the timers. These are
32-token chunk-forward measurements, not complete-prompt or generation latency.

The bias-only ablation improves long attention 3731.0 -> 3614.5 us and brings
the selected exponential interval to 257.9 us. Adding bit classification brings
that interval to 142.1 us. Independent ablation baselines were remeasured;
percentage improvements must not be added.

The final sampled task median is 1376.1 us, with whole instrumented attention
3330 us. Selected exclusive intervals are K load/convert/pack 194.1 us,
V load/convert/pack 191.3 us, exponential 142.1 us, online accumulator 110.5 us,
and max/sum 113.7 us. Only one KV-head task is instrumented; these numbers are
not an additive full-operator breakdown. Code layout and instrumentation also
affect timings. The marked exponential interval's static instruction count
falls 1030 -> 443, vector memory instructions 178 -> 65, with eight input/output
vector accesses in either case. This is codegen evidence, not per-instruction
cycle attribution.

Validation includes 4,015,867 host FP32 inputs around floor/cutoff boundaries
and arbitrary bits; 48 actual generated exponential calls on DSP with zero bit
mismatches across old/new code (including tested NaNs); 32 attention ABBA calls;
24 boundary calls; 12 full forwards including warmups; 16 normal resident
forwards and four full KV snapshots; eight valid LWP captures. The final source
build after the FP32 dtype guard is byte-identical to the tested kernel.

Independent FP32 logits max-absolute difference remains 1.238635. This preserves
the experimental FP16 policy; it does not qualify FP32 equivalence, continuous
generation, or resolution of prior intermittent DSP exceptions. The accepted
compiler/deployment remains unchanged. No commit/push or external issue was made.

Evidence: `.local/attention-softmax-exp-20261106/REPORT.md`,
`classified-summary.json`, `model-summary.json`, `profile-summary.json`,
`range-check.json`, `exp-candidate-result.json`, and `validation.json`.
K/V data preparation now exceeds the exponential region in this sampled task;
revisit preparation/conversion and its scheduling before further exponent-only
tuning. A fresh native per-phase comparison remains outstanding.


## Row-first direct K/V preparation (2026-11-06)

The optional prepared-HMX pass now traverses direct FP32 K/V input by physical
row pairs, sharing row address calculations across column tiles, following the
pinned llama.cpp `htp/hmx-utils.h` traversal. V conversion uses the IEEE HVX
converter's native interleaved result, eliminating redundant shuffles. FP16
input fallback, dynamic strides, tail padding and scatter completion remain.
Arithmetic stays in generated MLIR; no native attention arithmetic is substituted.

Warm ABBA on QCS9075: q32/KV1024 attention 3346 -> 3075 us, q2/KV2 206.5 ->
207.0 us. Conversion-only ablation 3327 -> 3237.5 us; row-first traversal on top
of conversion 3217.5 -> 3095.5 us. Each pair has its own control. Full 28-layer
q32/past992 forward 354.275 -> 346.863 ms, with attention 88.662 -> 83.019 ms.
Selected-task LWP V preparation falls from 191.25 to 110.63 us; current K is
184.53 us and exponential 134.01 us. These are sampled-task intervals, not an
additive full-operator breakdown. No fresh native per-stage result is claimed.

Two preparation IR checks, 96 standalone calls across three comparisons,
24 boundary calls, 12 complete model forwards including checked warmups,
16 normal resident forwards, four complete KV snapshots and eight LWP captures
pass. Final-source kernel equals the tested object. Model equality is against
the previous experimental FP16 policy; independent FP32-model logits error
remains 1.238635. Accepted deployment is unchanged. Evidence:
`.local/attention-pack-convert-20261106/REPORT.md` and `validation.json`.

Timing scope: historical 62.056 s covered FOUR q32 forwards for a complete
128-token prompt (`.local/prefill-20261101/summary.json`). The recent 351.557 /
346.863 ms values cover ONE q32 forward with past992. Input context and attention
precision differ; do not quote their ratio as an end-to-end speedup.


## Complete-prompt timing and initial-domain integration (2026-11-06)

Replaying the exact historical128-token input through the normal resident ABI
found model-body1.340s and native-interface-to-first-argmax1.412s, rather than
351ms for the whole prompt. The351ms figure is one q32/past992 forward. The
old62.056s covers all four q32 chunks and uses a different precision/runtime;
do not report their ratio as a controlled speedup.

This revisit exposed an integration gap: the first q32 domain still used an
older compiler output and took492ms; continuation chunks took280–285ms. Updating
that first domain to the current compiler with existing row-minmax and batched
RoPE optimizations gives a controlled warm ABBA comparison:

| Metric | Previous initial graph | Updated initial graph |
|---|---:|---:|
|128-token complete prefill model body |1339.599ms |1116.378ms |
|Normal resident API to first argmax |1409.133ms |1186.879ms |
|First32-token chunk |492.072ms |281.368ms |

The two variants use the same experimental attention policy.12 requests,
132 complete forward payloads match; the eight generated tokens also match the
historical record.24 extra boundary/state forwards and8 full KV snapshots match.
Three constant objects are unchanged. Initial-domain profiling places the gain
in other graph work (344.965->130.916ms in two short diagnostic captures), with
INT8, attention and SiLU essentially unchanged. This is a combined compiler/
configuration update, not isolated attribution to one pass.

The normal first-token timer includes packet transfer, output copies, cache
growth and argmax, but excludes compilation, loading, tokenization, Transformers
logic and all validation hashing. Thus it is not complete Transformers latency.
The assembled candidate is measured and retained, but full build-recipe/default
release integration remains pending; independent FP32-model acceptance is also
unchanged. Evidence: `.local/initial-domain-20261106/REPORT.md`,
`prompt-summary.json`, `profile-comparison.json`, and `validation.json`.

## Strict SiLU scheduling in the complete model (2026-11-06)

The four-lane FMA-chain scheduling experiment described in
[HEX-FP-001](LLVM_ISSUES.md) now has full-model measurements. Only the strict SiLU
kernel changes; the adapter, worker pool, graph, weights and attention arithmetic
are unchanged. This remains an LLVM IR prototype, not a maintained compiler pass.

| Warm ABBA median | Control | Scheduled SiLU |
|---|---:|---:|
|One q32 forward, past 992 |348.131 ms |342.188 ms |
|28 SiLU calls inside that forward |39.351 ms |32.712 ms |
|Complete 128-token prefill model computation |1114.513 ms |1087.876 ms |
|Normal resident interface to first argmax |1184.485 ms |1163.307 ms |

The long-KV comparison reports the same 1459.2 MHz clock on both sides. In a
separate profile of the complete 128-token prompt, SiLU sums to156.081 ->130.765ms;
remaining graph work is546.877 ->546.228ms, INT8 matrix operations341.471 ->
341.502ms and attention70.698 ->70.171ms. This locates the observed improvement
and shows why SiLU alone does not resolve model latency. These profile totals
use two warm requests per variant; normal timings above use four.

All132 normal forward payloads match, all twelve eight-token sequences match
the historical sequence, and32 diagnostic prompt payloads match the normal run.
Long-KV output hashes are unchanged, including the existing experimental
FP32-reference logits difference. Compilation/loading/tokenization are excluded;
validation hashing is outside normal timing. The diagnostic wall timer includes
profile queries and must not be reported as production first-token latency.
Evidence: `.local/silu-model-20261106/REPORT.md`, raw results and link commands.
No accepted deployment or default was replaced.

The schedule is now available through the optional compiler setting
`scheduleFMAChains=True` and the maintained strict-SiLU build flag
`--schedule-fma-chains`. This integrates the earlier benefit rather than adding
it again: current warm model measurements are1116.089 ->1090.667ms for the full
128-token prompt and344.953 ->340.996ms for one q32/past992 forward. Default-off
kernel bytes match the old baseline; enabled bytes match the generic scheduling
kernel tested on DSP.21 structural compiler checks,18 host tests,32 standalone
calls,20 edge calls and132 normal model payload comparisons pass. The model
remains an experimental assembly, not a qualified source release.

Fresh same-input SiLU measurements are1158.5us strict Triton versus87us native
llama.cpp at q32; native approximation differs by up to2.2888e-5 and its estimated
clock differs slightly. This still-large observed gap is not a compiler-only
ratio. See `.local/silu-chain-compiler-20261106/REPORT.md`, raw results and
`docs/qwen3/LLVM_ISSUES.md` for implementation and measurement scope.
