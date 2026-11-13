# LLVM Hexagon findings

This log separates correctness defects from performance limitations. A finding
is not an upstream bug until it has been reproduced on an identified upstream
revision, or the effect of local patches has been isolated. Reports should retain
both successful and unsuccessful experiments. No issue has been filed for the
finding below.

## Recording and reporting policy

Record a suspected LLVM issue when it is found, before attempting a workaround.
Each entry must include the smallest reproducer available, exact compiler
revision and local patch status, target features, command, expected and actual
behavior, numerical or timing evidence, and the current workaround. Preserve
the failing artifact so that subsequent fixes can be checked against it.

Use separate statuses for suspected defects, reproduced correctness bugs,
performance opportunities and rejected attributions. Notify the project owner
of confirmed impact and prepare an upstream report with a standalone reproducer;
record the issue URL only after a report has actually been submitted. Do not
attribute a runtime, linking or instrumentation defect to LLVM.

## Excluded attribution: sampled Attention copy dispatch

**Status, 2026-11-06:** diagnostic harness linking defect, fixed and retested;
not an LLVM bug or a production attention speedup.

Extracting the instrumented attention function left an external `memrefCopy`
reference. The control object's optimized implementation was localized, so the
instrumented function resolved to a weak compatibility implementation that
copies one element at a time. Numerical equality and balanced LWP records did
not detect this performance distortion.

Both variants now resolve to the same optimized implementation through the
strong symbol `hexagon_sample_shared_memrefCopy`. Eight device calls preserve
the saved input/output hashes, with balanced, uncapped LWP records. For
q32/KV1024, the selected program's outer exclusive interval falls from about
445 to 81 microseconds. Discard the earlier interval as a production bottleneck
estimate. This does not change the uninstrumented attention benchmark.

Evidence: `.local/attention-timeline-20261106/fixed-link-commands.json`,
`fixed-result.json`, `fixed-analysis.json` and `fixed-profile-summary.json`.
The original contaminated captures remain in `result.json` and `analysis.json`
for audit; use only the corrected files for further attribution.

## LWP-CFG-002: branch-site IDs merged before LLVM optimization

**Status, 2026-11-06:** reproduced local MLIR instrumentation/lowering defect;
not evidence of an upstream LLVM optimization bug. Normal inference is unaffected.

Direct K/V packing adds full-block and masked-tail branches. With LWP enabled,
the final MLIR canonicalization merges different exit markers into one call
whose site ID is a block argument. Exported LLVM IR then has a PHI ID. Hexagon
instruction selection only accepts an immediate ID and aborts with
`Cannot select: intrinsic %llvm.hexagon.instrprof.custom`.

The nonconstant ID is already present in the LLVM dialect **before** LLVM IR
export/optimization. The existing LLVM `nomerge` call attribute is applied too
late to prevent this MLIR transformation. A simple LLVM-only branch reproducer
did not reproduce the merge; retain that negative evidence instead of blaming
LLVM SimplifyCFG. The target's immediate-only selection pattern explains the
crash after invalid instrumentation reaches it.

Evidence: `.local/attention-kv-load-pack-20261106/profile-failure/pre-llvm.mlir`,
`pre-llvm-command.json`, `sampled.ll`, `sampled.bc`, and `build.log` in that directory. Four outlined matmul helpers contain block-argument IDs.
The command uses `--mlir-disable-threading`, matching the profiling Python
context; omitting it also exposes the existing shared-handler declaration race.
LLVM-only probe results are in `lwp-repro/`. The affected local llc is the LLVM
22 build identified in HEX-FP-001. No external issue has been submitted.

The local fix attaches the constant site ID as `hexagon.lwp.site` when creating
each MLIR call, so calls for different sites cannot merge before export. The
existing LLVM-stage `nomerge` protection remains. The new branch regression
passes through final MLIR canonicalization, and the full attention LWP object
now compiles with no nonconstant IDs. Eight device captures have balanced,
uncapped records and unchanged outputs. Existing external-function, loop and
routine-substitution instrumentation checks pass. This changes profiling only;
the uninstrumented attention object remains byte-identical after the fix.

## HEX-OPT-003: optimizer runs without a Hexagon target machine

**Status, 2026-11-08:** reproduced local integration/performance problem;
not yet an upstream LLVM cost-model defect. No external issue filed.

`LLVMIRTranslation.cpp::translateLLVMToLLVMIR` calls `runLLVMOptimizer`
with a null `TargetMachine`. The O3 optimizer therefore does not consult
Hexagon's target cost model, although instruction selection later uses Hexagon.
This matters when improving pointer representation in `LowerTPtr`: ordinary
byte GEPs expose contiguous loads, and SLP combines 16 independent strict scalar
FMA accumulators into a `<16 x float>` FMA. v73 legalization expands it into
scalar arithmetic plus repeated vector stack transfers and inserts; the actual
assembly also contains padded zero-lane computations.

Same-input QCS9075 probe, q32/KV128, four workers: the original Triton helper
is about 9.37 ms; a diagnostic GEP rewrite after LLVM optimization is 8.13 ms.
Preserving pointers before O3 instead produces 39.05 ms. Disabling SLP in an
isolated diagnostic compiler reduces that to 8.46 ms. Seven q/KV cases pass
byte-for-byte comparison with the C++ strict-FP32 control in both source-level
variants. These are standalone attention measurements, not model speedups.
The C++ control is about 8.3 ms for this case and is still the accepted path.

The identified local LLVM 22 build's standalone target-aware cost analysis
estimates scalar FMA cost 1, v16 FMA 174 and v32 FMA 350. Thus the evidence does
not justify blaming the Hexagon TTI implementation for reporting FMA as cheap.
Supplying the actual target machine to O3, with SLP still enabled, restores
scalar helper FMAs and gives 8.74 ms in the same case (seven numerical cases
pass). Target-aware O3 without the GEP change gives 8.91 ms. Neither exceeds
the accepted C++ control; these fixes are not yet model performance gains. Globally disabling SLP
is diagnostic only and is not a maintained fix.

Evidence under `.local/prefill-8h-20261108` and the active bounded work directory
`/dev/shm/qwen-prefill-8h-20261108`: `compiler-gep.ll`, `compiler-gep.s`,
`compiler-gep-board.log`, `no-slp-board.log`, `fma-cost.ll`, `fma-cost.log`,
and isolated compiler build commands. Exact source snapshots and incremental
patches are retained; the experiment has not replaced the deployed compiler.

## HEX-FP-001: strict FP32 FMA-chain scalarization and register pressure

**Status, 2026-11-06:** no new numerical miscompilation established. The initial
LLVM IR scheduling experiment now has an optional maintained compiler
transformation, kernel-build entry and full-model measurements. It remains off
by default and is not an accepted model deployment.

### Workload and compiler

Qwen3 SiLU is written in Triton. Its compatibility exponential uses eight strict
FP32 FMAs, followed by scaling and strict division. MLIR already produces
32-lane vector arithmetic. The current Hexagon backend scalarizes the FMA and
division portions. Vector syntax alone therefore does not establish that the
result executes entirely on HVX.

- Device: QCS9075, Hexagon v73, four worker threads.
- LLVM: `22.0.0git`; checkout HEAD
  `064f02dac0c81c19350a74415b3245f42fed09dc`, **with local modifications**.
- `llc` SHA256:
  `8af647bb1cad5dd0f43542bee752d13003c67bd30b062f2f906f744450372803`.
- Function target features:
  `+hvxv73,+hvx-length128b,-hvx-qfloat,+hvx-ieee-fp`.
- Code generation: `-O3 -filetype=obj -relocation-model=pic -code-model=small`.
- Recompiling the original LLVM IR with this `llc` reproduces the existing
  SiLU object byte for byte: SHA256
  `8c73c6a5031858bda8a50c855e2c3613b22f940928261d064c4d90443ce3aa03`.

### Evidence and bounded conclusions

A reduced 32-lane, eight-FMA chain generates 256 scalar FMA instruction sites
and an 896-byte frame. The complete SiLU baseline has a 4,608-byte frame.
These are static code properties, not measured per-instruction cycle costs.
The same reduced example also compiles without warnings using the installed
Ubuntu LLVM 18.1.3 and Qualcomm SDK LLVM 19.0.0: both generate 256 scalar FMA
instruction sites and an 896-byte frame for the tested function. This establishes
that the scalarization is not unique to our locally modified LLVM 22 build.
It does not establish a regression in current upstream LLVM. Only reduced
code generation was compared across these compilers; their full SiLU runtimes
were not benchmarked.

The backend's HVX legality setup admits vector add/subtract/multiply but does
not admit strict vector FMA in this configuration. That alone does not prove a
bug: target arithmetic capabilities and IEEE requirements must be considered.

The experiment evaluates each lane's original FMA chain together in a loop,
instead of materializing each full-vector intermediate. It preserves the
original operations, constants and per-lane dependency order. Other vector
operations and the strict division remain unchanged. Grouping four lanes per
loop iteration further improves the measured result.

ABBA on the same saved inputs, four samples per session with the first excluded;
both variants report 1,459.2 MHz. Times exclude compilation, loading, input
preparation, RPC and output checking. Both use the maintained worker adapter.

| LLVM IR transformation | q32 baseline | q32 candidate | Decision |
|---|---:|---:|---|
| FMA chain in a scalar loop | 1.3845 ms | 1.2290 ms | Positive evidence |
| Include scaling and division in that loop | 1.3870 ms | 1.5285 ms | Reject |
| Full chain, four-lane loop unroll | 1.3420 ms | 1.3740 ms | Reject |
| FMA portion only, four-lane loop unroll | 1.3880 ms | 1.1405 ms | Best diagnostic result, 17.83% less time |

The last variant also improves q1 from 50 to 42.5 microseconds. All saved-input
comparisons are bit-identical. A separate 20-call check covers q1, q32, strided
input, random finite bit patterns, zeros, infinities, NaNs and exponent
thresholds: zero bit mismatches. This is bounded device validation, not proof
over all FP32 inputs.

A subsequent full-model ABBA run replaces only that kernel object. The shared
worker-pool objects and the SiLU adapter are byte-identical. For one q32 forward
with 992 cached tokens, all 28 SiLU calls sum to 39.351 -> 32.712 ms, while total
forward time is 348.131 -> 342.188 ms. INT8/attention/remaining graph times stay
essentially unchanged. The exact historical 128-token prompt, through the normal
resident ABI, improves from 1114.513 -> 1087.876 ms of model computation; all 132
logits/new-KV payloads and all twelve eight-token sequences match the control.
This uses the same experimental attention precision on both sides; it does not
establish independent FP32-model equivalence. Evidence:
`.local/silu-model-20261106/REPORT.md` and its raw results/link commands.

The result does **not** explain the entire gap to llama.cpp. Its SiLU uses
different math; this experiment retains our strict compatibility arithmetic.
Do not label the whole observed gap a compiler defect or report this candidate
as the currently deployed model.

### Reproduction material

Local evidence is retained under
`/home/kevin/hexagon/pytorch-hexagon/.local/silu-vector-20261106/`:

- `scalar-chain/minimal.ll`, `minimal.s`, `codegen.json`: reduced code-generation
  reproducer, compiler fingerprint and exact invocation.
- `scalar-chain/minimal-portable.ll`, `alternate-compilers.json`,
  `ubuntu18-minimal.s`, `sdk19-minimal.s`: equivalent older-LLVM constant syntax,
  separate compiler fingerprints, warning-free invocations and assembly.
- `vector32/kernel.ll`: original full SiLU LLVM IR.
- `scalar-chain-poly4/kernel.ll`, `commands.json`: candidate and build commands.
- Each `scalar-chain*/result.json` and `summary.json`: complete paired runs,
  including regressions; `scalar-chain-poly4/edges-result.json`: edge checks.
- `try_scalar_chain.py`, `try_scalar_chain_full.py`, `try_chain_unroll.py`,
  `try_chain_poly4.py`: diagnostic transformations, not production build steps.

The reduced IR is standalone: run the recorded `llc` invocation in
`scalar-chain/codegen.json`. A DSP runtime is required only for timings.

### Maintained compiler implementation

`ScheduleFMAChains.h` in Hexagon-MLIR recognizes connected, single-result
32-lane FP32 chains with 4–16 FMAs, at most24 operations and two nonconstant
vector inputs. It preserves every operation and per-lane dependency, schedules
four lanes together, and leaves division unchanged. It applies only to v73
IEEE128-byte HVX, rejects fast math/strictfp/denormal-flushing environments and
observable intermediate results, and limits private buffers to1536 bytes per
function.21 structural tests include dependent chains and the stack bound.

The compiler option `scheduleFMAChains` defaults to false. The maintained
`strict-silu` kernel command exposes `--schedule-fma-chains`. With it disabled,
the resulting kernel object is byte-identical to the previous baseline; enabled,
it is byte-identical to the generic transformation measured on DSP.32 standalone
calls and20 random/edge/strided calls retain exact outputs. q32 median is
1346 ->1158.5us and q1 is50 ->43.5us, both sides at1459.2MHz. The complete prompt
test gives1116.089 ->1090.667ms of model computation, with132 identical forward
payloads and twelve identical eight-token sequences. This integrates the prior
benefit; it is not an additional cumulative speedup over the hand-edited prototype.
Evidence: `.local/silu-chain-compiler-20261106/REPORT.md`.

### Remaining work before an upstream claim or model release

1. Ubuntu 18 and SDK 19 independently reproduce the scalarization. For an
   upstream regression claim, still check a clean current upstream revision and
   determine whether strict FMA scalarization is expected for this target. Ask
   the Hexagon maintainers about preferred lowering and scheduling for this
   pattern.
2. Retain the measured distinction between unavoidable scalar arithmetic and
   avoidable register pressure. Test any compiler change with the same numerical
   contract; do not silently enable approximate arithmetic.
3. The compiler transformation and optional kernel build are now integrated;
   continue independent model-precision and full source-release qualification
   before changing a default deployment. No SiLU-specific LLVM-text rewriting
   was added to the maintained build, and no upstream issue has been submitted.

## HEX-MEM-001: alias information and alignment in conversion loops

Status: measured performance gap, not a demonstrated LLVM correctness bug or
upstream regression. Experiments use the same local LLVM22 toolchain as
HEX-FP-001. No issue has been submitted upstream.

LWP identifies113 initial-domain regions containing vector I32-to-F32 conversion
and scaling: K/V/gate/up outputs in28 layers plus the last-token vocabulary
output. Their combined time is about34ms within a roughly266ms initial32-token
forward. The loops are already HVX-vectorized. Unrolling alone does not improve
the model, and unchecked `noalias` assumptions would be unsafe.

A reduced gate loop copied from model LLVM measures420.5/405.0/425.0/238.5us for
base/unroll4/noalias/noalias+unroll4, at1459.2MHz. LLVM LoopVersioning supplies
runtime overlap checks and an original-loop fallback:404.0 ->239.5us on disjoint
buffers, with identical ARM FP32 output. Deliberate shifted overlap matches the
original DSP loop and retains its performance. The arithmetic is unchanged.

Memory layout determines whether the scheduling benefit survives. With unchanged
kernel code,128-byte-aligned buffers measure400.5 ->241.0us; offsetting them by64
bytes measures408.5 ->382.5us. Flush/invalidate-before-timing gives the same trend.
In the real graph, some allocations only guarantee64-byte alignment, and all311
tensor symbols in the first constant DSO are at address64 modulo128.

The profitable prototype combines DSP-local function extraction, explicit weight
address arguments, checked loop versioning/unrolling, and128-byte temporary
storage. Final LWP conversion/scaling total is34.010 ->24.511ms. Aligning constant
DSO1 additionally yields24.195ms, mainly improving vocabulary output; the extra
whole-model benefit is small/noisy. The complete128-token prompt measures
1097.134 ->1082.515ms, with88 complete forward payloads and eight generation
sequences equal to the prior maintained model. Only the initial domain receives
compiler scheduling changes; no broader-domain or independent FP32 acceptance
is implied. Experimental FP16 attention remains unchanged.

The runtime's128-byte allocation fix is maintained and its SDK machine code
matches the tested prototype. The compiler transformation remains diagnostic.
Before upstreaming, add structural admission and overlap-fallback tests, and
verify normal source lowering with the intended buffer alignment. The existing
`OneShotBufferizePassOptions.bufferAlignment` also reaches constant globals via
Arith bufferization; avoid a production ELF-rewriting workaround. Preserve the
numerical contract and report the remaining up-branch cost separately.

Evidence: `.local/graph-regions-20261106/REPORT.md`,
`experiment-validation.json`, reduced LLVM cases and transformation sources in
`experiment-sources.tar.gz`. The private archive also records rejected variants;
it is reproduction material, not a proposed upstream patch.

### Up-branch scheduling and checked alignment follow-up

The exact up loop (three ordered FP32 multiplies) measures490.5/444/410/337us
for original/checked-unroll2/4/8 at1459.2MHz, with128-byte-aligned buffers.
Generated hot loops show no vector spills; do not attribute the gap to spilling.
A second prototype checks buffer base alignment at runtime and only raises load
or store alignment when ScalarEvolution proves the four-byte element offset is
a multiple of32. Overlap or alignment failure retains the original loop.
FP32 weight loads with weak alignment are unchanged. In a separate paired run,
original -> checked-aligned-unroll8 measures475.5 ->291.5us; with only weights
offset by64 bytes,479 ->289.5us. All480 reduced-kernel calls across the two runs
match ARM FP32 for disjoint buffers or the original DSP loop for shifted overlap.

LWP in the complete initial32-token forward measures the28 up regions at
12.239 ->8.638ms and all113 regions at24.470 ->19.339ms, relative to the prior
outlined/unroll4 prototype. The complete128-token prompt measures1080.602
->1074.536ms; normal resident time to first argmax1154.572 ->1150.862ms.
Only two warm observations per variant were collected for these full-prompt
medians: report this small gain as diagnostic, not a robust general speedup.
All88 forward payloads and eight eight-token sequences match the saved previous
model. Math, constants, other graph domains and the FP16-attention limitation
remain unchanged. No compiler integration or new release acceptance is implied.

Splitting the already outlined helpers into separate objects takes106.9s for
the main object once, then about2s to regenerate helpers. It has no meaningful
standalone model-speed benefit; its purpose is quicker controlled iteration.
Evidence and reproduction: `.local/up-schedule-20261106/REPORT.md`.


### Maintained dequant scheduling and continuation-domain validation

The optional `scheduleDequantLoops=False` compiler hook now implements structural
admission, function extraction, runtime overlap/alignment checks and unrolling.
It admits v73 IEEE 128-byte HVX loops containing I32-to-F32 conversion followed
by two or three ordered multiplies; no model names or fixed model dimensions
are matched. Analysis runs on the extracted helper, not the giant model function.
The application default remains disabled. No accepted deployment was replaced.

33 structural checks and eight focused host tests pass. All90 DSP micro calls
match ARM FP32 for disjoint buffers or the original DSP loop for shifted overlap.
New source builds cover initial and continuation q2-32 domains. Continuation
with the option disabled reproduces all five prior objects byte-for-byte;
enabled compilation leaves its four constant objects unchanged. Initial source
compilation used the first reviewed implementation; continuation used a later
review adding conservative admission exclusions and analysis-cache refresh
on rollback. They are not claimed to use identical compiler source revisions.

For the complete128-token prompt, four warm observations per variant give
1089.357 ->1033.456ms of model computation and1159.202 ->1102.649ms to first
argmax through the normal resident API. All198 forward payloads and18 eight-token
sequences match the previous experimental model. LWP on q32/past32 attributes
33.209 ->19.421ms to113 conversion/scaling regions (K3.844 ->2.088, V3.901 ->2.114,
gate12.135 ->6.268, up12.549 ->8.613, vocabulary0.780 ->0.337ms).
These region medians are diagnostic, not an exact additive decomposition.
Compilation/loading/checking are outside timers. FP16 attention remains
experimental; this is not independent FP32 acceptance or Transformers end-to-end.
Evidence: `.local/dequant-compiler-20261106/REPORT.md` and `experiment.tar.gz`.

## HEX-SE-001: ScalarEvolution analysis stack exhaustion on the full graph

Status: locally reproduced analysis failure, not yet a minimized or clean
upstream LLVM regression. No upstream issue has been submitted.

A verifier-valid model LLVM module triggers SIGSEGV with an8MiB stack when
LoopAccessInfo analyzes the inner loop under `.preheader12359` in the original
large function. A standalone C++ reproducer uses only LLVM read-only analysis
APIs. It prints constant trip count96, then crashes in about1.4s. The captured
backtrace has roughly23,470 frames, repeatedly visiting ScalarEvolution range,
no-wrap, exit-limit and implied-condition analysis. Standard `opt` access-info
printing instead timed out after30s; that is not a second crash reproduction.

LLVM revision is064f02dac0c81c19350a74415b3245f42fed09dc with local modifications.
Replacing only ScalarEvolution.cpp by the unmodified HEAD source, recompiling
that object and linking it before the local LLVM archives still reproduces
SIGSEGV in1.418s. This excludes the local constant-disposition-cache edit as a
necessary cause, but remaining libraries are local: it is not a clean full
upstream build. Preserve this qualification in an external report.

The compiler integration outlines the small region first, then queries loop
access and trip-count information locally; the model transform completes in
about0.5s without raising stack limits. Next upstream preparation requires a
smaller IR and a clean-toolchain check. The self-contained reproducer, commands,
source identity and compressed backtrace are private reproduction artifacts in
`.local/dequant-compiler-20261106/experiment.tar.gz`.


## HEX-MEM-002: repeated one-vector copies in strict SiLU

Status: measured code-generation overhead with an existing compiler-option
workaround, not an LLVM correctness bug or demonstrated upstream regression.

The Triton SiLU loop processes 32 FP32 values at a time. With the previous
`stackBufferBudget=0`, a 128-byte temporary survives as a heap allocation and
an external `memcpy` call survives inside the loop: 3,072 calls for 32 x 3,072
values across four workers. This is DSP-local copying, not an ARM/DSP RPC.

A split-operator diagnostic uses the same saved inputs, actual DSP-generated
exponential denominators, unchanged strict arithmetic and four workers. Warm
q32 timings are full SiLU 1,164us, exponent/denominator 770us, division 656.5us,
and copy-only 169.5us. Each separate stage pays its own memory/cache/dispatch
costs; these are not additive fractions of the full operator. Composition of
the exponent and division stages reproduces the original output bytes.

Qtimer markers inside the existing adapter place approximately 37us in the
three cache-maintenance calls and 1,132us in workers for full SiLU. Copy-only
spends approximately 33us in cache maintenance and 133us in workers. Removing
cache synchronization blindly would not address the main cost. All standalone
strict comparisons report 1459.2MHz. This diagnostic adds timing only; its
adapter is not part of the maintained runtime.

Two private compiler prototypes were evaluated: bounded strict-division
scheduling and LLVM `memcpy.inline`. The simpler existing stack promotion is
better: the temporary costs 128 + 63 = 191 bytes including alignment padding.
A budget of 128 therefore has no effect; 256 admits it, and LLVM eliminates
both the allocation and copy. Increasing the budget to 4096 produces the same
object. Generated `silu_forward` stack frame falls from 4096 to 1664 bytes;
this is a static frame size, not a measurement of total worker stack use.

A paired comparison gives q1 44 ->35.5us and q32 1151.5 ->1009.5us, with exact
outputs. The existing compiler plus the maintained strict-silu CLI reproduces
the tested object byte-for-byte. The final change only sets the existing
256-byte budget for that optional kernel build and records it in the manifest.
The trial inline-copy compiler hook, option and tests were removed. No new
compiler transformation, public flag, runtime environment variable or native
math implementation remains from this follow-up.

In the complete experimental model, the same 128-token prompt measures
1031.751 ->1018.100ms of model computation and 1106.084 ->1090.350ms through the
normal resident interface to first argmax (four warm observations per variant).
All 132 logits/new-KV payloads and twelve eight-token sequences match the saved
previous model. Separate initial q32 operator profiling measures 28 SiLU calls
at 32.826 ->29.376ms, INT8 at 85.307 ->85.182ms and attention at 15.212 ->15.270ms.
The whole initial block is 251.277 ->247.423ms. Independent medians need not add.
The initial-block attention timing must not be compared with KV1024 timings.
Twenty random/finite-bit/special/strided calls also retain exact output bytes.
Seven existing host checks pass. No independent FP32-attention acceptance or
accepted deployment replacement is implied; the FP16 attention gap remains.

The native llama.cpp SiLU implementation uses fast approximate sigmoid followed
by multiplication, with VTCM/DMA staging (`htp/act-ops.c`, `hvx-sigmoid.h`). It does
not evaluate our compatibility exponential followed by strict division. Its
previous same-input q32 result was 87us at an estimated 1472.8-1486.8MHz, with
max absolute difference 2.288818e-5 to our strict output. The remaining roughly
11.6x ratio is neither equal-math nor an isolated compiler-only comparison.
Evidence and reproduction: `.local/silu-copy-20261107/REPORT.md` and its private
`experiment.tar.gz`; the maintained incremental patch contains only the build
configuration and documentation.


## HEX-PRED-001: native predicate bitcasts do not pack logical bits

**Status, 2026-11-07:** reproduced DSP correctness defect, with a narrow local
source fix and isolated-compiler validation. Not yet submitted upstream. The
active compiler and accepted deployment have not been replaced. This finding
does not establish the cause of the outstanding FP16-attention precision gap.

A reduced LLVM-only function compares four integer lanes, shuffles the boolean
results, bitcasts `<4 x i1>` to `i4`, and zero-extends the result. For all 16
logical combinations, the correct integer is the four logical bits packed into
bits0–3. On QCS9075, the current locally modified LLVM22 compiler returns only
0 or15 (the first logical bit replicated): 14/16 masks are wrong, and a test
against15 takes the wrong branch for7/16 inputs. Turning off Hexagon bit-copy
generation does not fix this reproducer.

The same LLVM IR was compiled and run with the installed Ubuntu LLVM18.1.3 and
Qualcomm SDK LLVM19.0.0. Both expose the native predicate's duplicated bits
instead: masks1,2,4,8 become3,12,48,192, and mask15 becomes255. Their 15/16 wrong
masks produce2/16 wrong all-true decisions. These cross-version device results
establish the defect is not confined to our local LLVM22 modifications; they do
not establish its status on a fresh current-upstream checkout.

The practical trigger was a private Triton SiLU division experiment. Its
normal-result certificate should reject a denominator of +Inf. The denominator
capture confirms +Inf, but the failed-lane counter was zero. For input-104,
strict SiLU returned negative zero while the experiment returned approximately
-3.0562853e-37. A 98,304-element grid over[-104,104] had7,221 differences. Replacing
just the packed predicate check with equivalent scalar checks removed all grid
differences. This was not a model precision tolerance or an approximate-division
error.

Native Hexagon predicates replicate each logical lane four times for`v2i1` or
twice for`v4i1`. The fix combines these two bitcasts before generic legalization:
read the native predicate register, select one bit from each replicated group,
and pack those bits into the result. Changes are in
`llvm/lib/Target/Hexagon/HexagonISelLowering.cpp`, with
`llvm/test/CodeGen/Hexagon/native-predicate-bitcast-pack.ll` in the LLVM checkout.
Other bitcast widths retain their existing handling. No runtime flag is added.

Validation:

- The isolated compiler passes all16 four-lane logical masks, and all four
  two-lane masks (repeated across16 probe cases), including the all-true test.
- Nine code-generation commands pass: new scalar/HVX target cases and seven
  existing native/HVX predicate-bitcast cases. The new check fails on the
  previously active compiler, making it a regression test rather than only a
  source-pattern check.
- Recompiling the private SiLU experiment with the isolated compiler makes all
  72 paired DSP calls match the strict reference bytes, across saved inputs,
  wide values, finite random bit patterns, stride, NaN/Inf and signed zero.
- The experiment is nevertheless rejected for performance: q32 exact-residual
  division SiLU2829.5us versus strict1036.5us in the same run. This compiler fix
  is not presented as a production SiLU speedup.

Compiler HEAD and existing`llc` hash remain those recorded in HEX-FP-001. The
isolated executable links one rebuilt translation unit before the existing
local LLVM libraries; it is not a clean full-upstream build. Exact versions,
hashes, commands, raw device results, source delta and minimized IR are in
`.local/silu-quant-guard-20261107/experiment.tar.gz` and its report. Only the
LLVM source/test and this issue log are maintained changes from this iteration.


## HEX-RED-001: FP32 min/max reductions become scalar library calls

**Status, 2026-11-07:** measured performance limitation in the local LLVM22
build; not a confirmed upstream regression or a correctness defect. No issue
has been filed. The compiler is the existing modified toolchain at HEAD
`064f02dac0c81c19350a74415b3245f42fed09dc`; active llc SHA256 is
`8af647bb1cad5dd0f43542bee752d13003c67bd30b062f2f906f744450372803`.

The standalone `.local/silu-selective-20261107/reduce-minimal.ll` loads one
128-byte FP32 vector and calls `llvm.vector.reduce.fmax.v32f32` or `fmin`.
At `-O3 -hexagon-insert=false`, with v73 IEEE HVX enabled and no fast-math,
each reduction generates 31 `fmaxf` or `fminf` call sites. Expected opportunity:
a vector reduction preserving the intrinsic's NaN, signed-zero and denormal
semantics. The existing opt-in `outlineVectorMinMax` handles `fmaximum` /
`fminimum`; it does not handle this distinct `fmax` / `fmin` family.

Reproduce from the application root:

```bash
LD_LIBRARY_PATH=/home/kevin/hexagon/sysroot_compat \
  .local/llvm-toolchain-20261030/build/bin/llc -O3 \
  -hexagon-insert=false -filetype=asm \
  .local/silu-selective-20261107/reduce-minimal.ll \
  -o /tmp/hexagon-reduce-minimal.s
```

In a private selective SiLU×up kernel, per-block candidate and quantization
boundary scans exposed this cost. Four workers ran while LWP recorded only
worker0 (eight of32 rows). Its candidate scan was430.56us and boundary scan
714.03us, inclusive; nested intervals must not be added. This handler reads
C15:14 processor cycles, converted using the measured1459.2MHz clock.
Replacing the reductions with a vector-tree IR experiment changed the q32
candidate from about1.60ms to0.83ms. This is evidence for this kernel, not an
attribution of the Attention gap to the compiler.

The final private candidate expresses finite-value order keys in Triton,
avoiding these float reductions without patching generated IR. After refining
the error intervals, paired q32 SiLU×up time was1015.5us strict versus613.0us
candidate. All96 calls preserved row scale and INT8 codes; arbitrary FP32
intermediates differ, and boundary-heavy inputs are slower. Mathematical bound
review, actual generated-instruction review and full-model integration remain
pending. No production/default backend or compiler binary was replaced.

Evidence: `.local/silu-selective-20261107/REPORT.zh-CN.md`, `bounded-result.json`,
`lwp-result.json`, `lwp-summary.json`, `reduce-minimal-counts.json`,
`error-bound.json` and `experiment.tar.gz`. Experiments use the previously
isolated predicate-pack llc fix; the minimal reduction reproducer above uses
the unchanged active compiler. Neither result establishes current-upstream
behavior. Keep `maxnum` versus `maximum` semantics separate in any general fix.

## MLIR-VEC-001: multi-result vector writeback and incomplete transfer checks

**Status, 2026-11-07:** local Hexagon-MLIR performance and correctness finding;
not an LLVM backend defect. An isolated compiler fix and regression tests are
available; no default compiler/deployment was replaced, and no issue was filed.
Hexagon-MLIR HEAD is `f13109dd425e3002999f488968542d040825bdca`, with local
modifications. Exact binary identities and the incremental source patch are in
`.local/swiglu-pair-20261107/identities.json` and `vector-writeback.patch`.

`EraseVectorToTensorWritebackPass` admitted only single-result loops. A Triton
SwiGLU experiment with two reduction accumulators retained two tensor writebacks
to the same empty tensor. Bufferization then generated malloc/free inside each
32-element iteration: 3072 pairs for q32/width3072. The revised pass retains all
admitted accumulators as vector loop arguments/results. It requires full static,
unmasked, zero-offset, identity transfers and single uses; mixed result loops
are conservatively left alone. Shape equality alone was unsafe: the previous
single-result implementation incorrectly transformed masked/nonzero-offset
reads. New negative cases cover those existing correctness gaps.

The existing `erase-vector-to-tensor-writeback.mlir` and new
`erase-vector-writeback-multiple.mlir` pass their FileCheck commands with the
isolated `linalg-hexagon-opt`. The new positive multi-result case and negative
single-result cases distinguish the old compiler. Rebuilding the original
32-lane guarded kernel with the new compiler produces an identical object
(`0854965e74732689e083d55975a706fd9635ec3215f8b1a2db3352e995d3702c`).
Full-model source rebuilding with this compiler remains unverified.

Removing heap writebacks reduces the integer-guard experiment's common-input
q32 first-layer kernel from about4.03ms to0.497ms; the original guarded kernel
in the latter pairing is0.574ms. This is **not an accepted model improvement**.
Including actual ordered up dequantization reverses the benefit: 28 per-layer
warm medians sum to20.418ms original versus21.272ms candidate. Complete128-token
model computation is951.187 versus954.254ms. Therefore the application guard/max
rewrites are not retained. The compiler fix is reviewed independently.

Worker0 LWP (C31:30,19.2MHz, balanced uncapped traces) localizes raw regression
to the first pass:401.17→422.81us in layer0. In contrast, the common-input first
pass improves346.88→275.00us. Static raw first-loop packet count124→66 and folded
spill/reload annotations10/32→0/14 do not establish actual DSP cycles; increased
spill volume is not supported as an explanation. Precise scheduling/memory
causality remains unresolved. Evidence, commands, IR and timings are archived
in `.local/swiglu-pair-20261107/experiment.tar.xz`; see its Chinese report for
scope and rejected alternatives.

## HEX-PRED-002: 64-lane integer-to-predicate truncation cannot be selected

**Status, 2026-11-07:** reproducible compilation failure in four identified
installed compilers, blocking a wider-vector SwiGLU experiment. Not established
as a regression in current upstream LLVM; no upstream issue or fix submitted.
Both original and integer-guard kernels fail when their Triton BLOCK is changed
from32 to64. Neither64-lane kernel was executed on DSP.

The standalone reproducer `.local/swiglu-pair-20261107/predicate64.ll` loads
`<64 x i32>`, truncates to `<64 x i1>`, selects between two `<64 x float>` values,
and stores the result. It has no external library or model dependencies.
Target: hexagonv73, `+hvxv73,+hvx-length128b,-hvx-qfloat,+hvx-ieee-fp`.
Expected behavior: legalize the two-HVX-vector operation and compile. Actual:
`LLVM ERROR: Cannot select: v64i1 = truncate`, process abort. The same32-lane
control compiles successfully.

```bash
ulimit -c 0
LD_LIBRARY_PATH=/home/kevin/hexagon/sysroot_compat \
  .local/llvm-toolchain-20261030/build/bin/llc -O3 \
  -hexagon-insert=false -filetype=obj \
  .local/swiglu-pair-20261107/predicate64.ll -o /dev/shm/predicate64.o
```

The32-success/64-failure pair reproduces on the active modified LLVM22 build,
the isolated HEX-PRED-001-fixed build, Ubuntu LLVM18.1.3 and SDK LLVM19.0.0.
The local LLVM22 checkout HEAD remains
`064f02dac0c81c19350a74415b3245f42fed09dc`; it contains local changes.
`predicate-width-results.json` records all eight commands, exit statuses and
compiler SHA256 hashes. Logs and generated full kernel IR are in the experiment
archive. Existing32-lane behavior/default compiler are unchanged. Splitting into
two legal32-lane vectors is a possible next experiment, not yet a tested fix.
This compilation failure does not explain the measured32-lane runtime gap.

### Follow-up: two separate32-lane vectors, 2026-11-07

An application-level scheduling change now exposes two independent32-lane
products before storing either, without creating a64-lane predicate. It builds
with the isolated MLIR-VEC-001 compiler and existing HEX-PRED-001-fixed llc.
The compiler limitation itself is still unfixed. The maintained experimental
Triton kernel rebuild produces the exact tested object, SHA256
`94f213aa0eea419e2eb1ca22057d0717d0eb59480f916cc2bb690cd6d5d33362`.

Combining this scheduling with integer domain/max accumulators changes the
previous rejection: actual raw-input28-layer medians sum20.457→14.274ms, and
the128-token model pairing is950.317→926.274ms. All132 forward payload hashes
and12 eight-token sequences match the prior experimental baseline. Five
quantization-floor cases preserve FP32, scale and INT8 results. Four-way
interleaving was slower than two-way (14.562 versus14.295ms in its own pairing)
and is not retained. This is application-plus-compiler scheduling evidence,
not a claim that fixing64-lane selection alone accelerates a model.

Worker0 LWP first-pass exclusive time falls398.28→181.33us in layer0 while
boundary repair remains93.05→92.58us. Full-model testing still uses private
generated graphs and the unaccepted FP16-attention baseline; no accepted
deployment/default compiler changed. Full data, sources and rebuild commands:
`.local/swiglu-interleave-20261107/REPORT.zh-CN.md` and its experiment archive.

### Control-flow limitation exposed by block repair

The next application experiment lets only rejected32-element blocks take the
strict path. Carrying the vector peak across its repair `scf.if` prevents the
current writeback pattern from matching; the first loop again contains a192-byte
malloc/free per64 elements. The raw28-layer sum regresses14.379→58.203ms.
Moving only the reduced scalar peak across the branch, and loading repaired
values after the inner conditional store, eliminates all malloc/free calls in
that kernel without another compiler patch. The refined pair is14.314→12.072ms;
the128-token model pair is927.786→918.022ms, with unchanged full-model payloads.
This is a limitation of the admitted MLIR pattern, not a new LLVM correctness
bug. Branch-dependent vector writeback remains an optimization opportunity.
Sources, failed and successful generated IR, and commands are preserved in
`.local/swiglu-block-repair-20261107/experiment.tar.xz`.

## HEX-PRED-003: HVX truncation tests every byte instead of each integer lane

**Status, 2026-11-07:** wrong-code reproduced on QCS9075 with the local LLVM22
build and Qualcomm SDK LLVM19.0.0. A two-pattern fix passes the device reproducer,
64/128-byte CodeGen checks, and the original strict-model generation replay.
No upstream issue or patch has been submitted. Read-only inspection of upstream
LLVM HEAD `5148cbe83ef52f61307acb55656a6e12e5fcfbf2` also finds the same i16/i32
patterns in [HexagonPatternsHVX.td](https://github.com/llvm/llvm-project/blob/5148cbe83ef52f61307acb55656a6e12e5fcfbf2/llvm/lib/Target/Hexagon/HexagonPatternsHVX.td#L763).
This is source evidence only; that upstream revision was not built or run.
This is separate from the
64-lane i32 compilation failure in HEX-PRED-002: successful compilation of a
32-lane control did not establish correct execution.

`trunc <32 x i32> to <32 x i1>` must take bit zero of each **word**. The existing
`HexagonPatternsHVX.td` rule emits `q = vand(v, 0x01010101)`, testing bit zero of
every **byte**. Its four predicate bits per word can therefore disagree.
The analogous halfword rule has the same defect; the byte rule is correct.
A later `vmux` mixes bytes from two different operands instead of selecting a
whole logical lane. Extending the malformed predicate can likewise return a
value other than zero/one or zero/minus-one.

A reduced test loads integer conditions and two arbitrary vectors, truncates
the conditions to i1, selects the two vectors, and stores the result. There is
no PyTorch, quantization, model state, approximation or runtime math dependency.
Negative and upper-byte marker inputs expose the error; testing only 0/1 or
all-byte-replicated conditions can miss it. On each original compiler, the
32-bit fixed/random fixtures have 24/30 wrong lanes and the 16-bit fixtures
32/23; both 8-bit controls pass. With the fix, all six fixtures pass.

### Model trigger and diagnosis

Expanding strict decode softmax from eight to32 elements exposed this pattern
in nearest-even integer rounding. A captured finite input is FP32 bits
`0xc136fda9` (approximately -11.436929); multiplication by log2(e) yields exactly
-16.5. The compatibility exponential should choose integer -16, but vector
code chooses -272: the correction is -256 instead of zero. The resulting
exponential overflows; normalization produces NaN in one attention head.

The first strict-model discrepancy occurred at occupied position76, q1, in a
Chinese/code generation replay; the first differing new KV is layer4 (one-based).
Capturing layer3's Q/K/V reproduces the NaN in standalone attention. A32-element
version **without** FMA scheduling also fails, excluding that scheduling pass.
The eight-element control passes all262 recorded requests and full KV snapshots.
The fixed32-element model passes the same262 records, previously independently
checked against ARM W8A8. This is a replay equivalence check, not a new public
source-build receipt or acceptance of experimental FP16 prefill.

### Fix and validation

For i16/i32, mask each full lane with integer1, then compare against integer1
using `vcmp.eq` at that lane width. This generates identical predicate bits for
every byte of a logical lane. Keep the i8 rule unchanged. The source change is
in `llvm/lib/Target/Hexagon/HexagonPatternsHVX.td`; the regression test is
`llvm/test/CodeGen/Hexagon/hvx-truncate-predicate-lanes.ll`. Both64/128-byte HVX
widths and the unchanged byte path are checked. The four existing bitwise and
vector-comparison tests for those widths also pass.

The optional strict-decode32 profile requires this LLVM fix in the compiler
library actually used to build the kernel. Rebuilding only Triton Python code
or passing `--schedule-fma-chains` does not repair an older LLVM library.
Default eight-element strict decode is unchanged. The 32-element compatibility
exponential is also shared by strict SiLU and quantized SiLU repair; rebuild
those objects with the corrected LLVM as well. Fixing only the attention
object does not repair already-compiled SiLU objects.

Evidence: `.local/predicate-trunc-20261107/`; scratch:
`/dev/shm/qwen-predicate-trunc-20261107/`; original model input and replay:
`/dev/shm/qwen-strict-current-20261107/`. The issue package records tool hashes,
standalone LLVM IR, commands, expected/actual device values, and an incremental
patch excluding unrelated existing LLVM changes.

## PMU-API-001: iTrace IDs are not raw hardware event numbers

**Status, 2026-11-08:** diagnostic harness error, not a compiler or DSP defect.
The SDK `itrace_dsp_events_pmu.h` explicitly defines logical iTrace IDs, not
hardware PMU values. Removing their `0x8000` prefix does not convert them to
values accepted by `qurt_pmu_set`. The private `hvx-workers` experiment used
that incorrect mapping for some events. Discard its named HVX activity/stall/
concurrency interpretations; its elapsed times and numerical checks remain valid.

Correct raw V73 values come from the scalar programmer manual 80-N2040-53
Rev. AB, Table 9-1, and HVX manual 80-N2040-54 Rev. AB, Table 4-1:

- `3`: committed packets; `0x50`: committed FP instructions (not FLOPs).
- `0x300`–`0x303`: cycles committing one through four packets.
- `256`: HVX active; `273`: HVX packets; `275`: VFIFO-full stall cycles.
- `277`–`280`: cycles with one through four HVX contexts running.

The corrected runs satisfy `committed_packets = sum(n * cycles_n_packets)`
to within 5.2e-7 relative error. The context counters now return nonzero values.
For q32/KV1024, the strict C++ control takes about61.45ms, with152.49 million
committed FP instructions; the staged Triton4 experiment takes63.75ms with
140.90 million. Fewer instructions alone did not produce a faster kernel.
These are single-operator measurements, not full-model or equal-precision
llama.cpp comparisons. Do not derive a hard scalar throughput ceiling from
instruction slot counts: V73 has multiple clusters and SMT.

Evidence: `.local/prefill-8h-20261108/`, working set `pmu-verified/`.

## HEX-MEM-004: scalar lowering of ordinary copies in the model graph

**Status, 2026-11-08:** measured performance opportunity, not a correctness bug.
The graph and its generic `memrefCopy` helper make many small copies. For the
fixed128token prompt, the ordinary-copy histogram includes89,344 copies of256B
and43,008 of512B. Looking only for direct graph copies misses the helper calls.

Two separate changes retain byte-copy semantics:

1. LLVM `HexagonTargetLowering::getOptimalMemOpType` can choose native HVX words
   for full vector-sized constant memory operations, with scalar tails.
2. The opt-in Hexagon-MLIR `outlineHVXMemCpy` lowering uses LLVM's existing
   `expandMemCpyAsLoop` construction for dynamic/large copies in v73/128B
   functions. It accepts unaligned byte ranges, retains bounds-safe tails, and
   excludes volatile, inline-only and nonzero-address-space copies. Small
   constant copies retain their ordinary instruction-selection path. The
   option defaults to false and adds no external runtime symbol.

The dynamic-copy-only whole-model ABBA experiment gives1103.79 ->1067.31ms
for128token prefill, with identical logits, appended KV and eight generated
tokens. This uses the accepted arithmetic IR; it is not evidence that all
model graphs have been rebuilt through the public source pipeline. Eighteen
admission/idempotence cases and500 board byte/canary cases pass. The separate
LLVM constant-copy change passes HVX128/HVX64/scalar code-generation checks
and existing aligned/inline memory-operation regressions. Combined-model
qualification is tracked separately; do not add isolated speedups together.

## HEX-PIPE-002: target-aware unrolling hides dequantization regions

**Status, 2026-11-08:** Hexagon-MLIR pass-order integration issue; not an LLVM
correctness bug. A fresh model source build exposed a difference that composing
previously optimized graph objects had hidden. With the real Hexagon target
machine, O3 partially unrolls the ordered INT32-to-FP32 scaling loops before the
post-O3 `scheduleDequantLoops` matcher. The matcher requires one conversion and
one store, so it finds only one region instead of113 in each prefill graph.
The resulting loops retain weak alignment and lack the guarded no-alias path.

The opt-in transformation now also runs at the module optimizer-early extension
point, after scalar simplification and before late vectorization/unrolling.
LLVM retains its actual target costs; no global unrolling switch is changed.
The existing runtime alias/alignment checks and fallback remain. Fresh initial
and continuation source builds both recover113 outlined regions. This is
structural evidence; whole-model performance and numerical qualification are
recorded separately. The33 existing admission/overlap/idempotence checks pass.

## BUILD-HMX-001: SDK HMX bitcode requires the matching SDK linker

The HexKL micro archive contains SDK bitcode using `llvm.hexagon.M8.*`
intrinsics. Source-built upstream LLD can accept that archive yet emit unresolved
calls to those intrinsic names; the DSP loader then rejects the model. This is
a toolchain compatibility gap, not a supported-instruction miscompile.

The resident builder selects the SDK `hexagon-link` only for model links that
consume the optional INT8 HMX archive. Constant libraries keep the source-built
linker and their checked segment splitting. The SDK linker does not accept GNU
`-z separate-loadable-segments`, so that flag is omitted for the HMX model link.
The builder records the SDK linker hash and rejects any unresolved `llvm.*`
symbol after linking, before publishing a session receipt.

## HEX-PIPE-003: runtime copies bypass the opt-in copy lowering

**Status, 2026-11-08:** integration omission in our new `outlineHVXMemCpy`
optimization, not an upstream LLVM miscompile. The Python backend links runtime
bitcode after the model LLVM optimizer. Running the new copy transformation
only inside that optimizer leaves the three dynamic `memcpy` sites in
`memrefCopy` untouched. A reduced strided-copy input confirms the difference
through the actual Python translation entry: option off retains those three
intrinsics; option on now lowers all three after runtime linking.

`LinkRuntimeModules.cpp` reapplies the existing idempotent transformation after
linking, gated by the same default-off option. Function target restrictions and
bounds-safe vector/byte-tail construction are unchanged. No numerical kernel
or runtime copy implementation is replaced.

Private same-graph ABBA, fixed128token prompt: model prefill1143.157 ->1104.698ms;
first-token packet wall1217.555 ->1177.180ms. All logits/KV payloads and eight
generated tokens match. Separate official LWP measurements localize the change:
664 `memrefCopy` calls, inclusive time50.893 ->13.804ms. LWP uses19.2MHz global
timer ticks; its queries are excluded from the normal ABBA measurement.
Fresh public source build/qualification is recorded separately in
`.local/prefill-8h-20261108`; these numbers alone do not establish deployment.

## HEX-PRED-004: storing a 32-lane predicate fails instruction selection

**Status, 2026-11-08:** compilation failure reproduced with the local source-built
LLVM22, system LLVM18 and Qualcomm SDK LLVM19. No wrong output or upstream issue
submission is claimed. A private exact-division Triton experiment caused
bufferization to materialize a logical predicate across a branch. The independent
LLVM reproducer needs only a vector load, comparison and predicate store:

```llvm
target triple = "hexagon"
define void @store_mask(ptr %input, ptr %output) #0 {
  %values = load <32 x i32>, ptr %input, align 128
  %predicate = icmp eq <32 x i32> %values, zeroinitializer
  store <32 x i1> %predicate, ptr %output, align 4
  ret void
}
attributes #0 = { "target-cpu"="hexagonv73"
                  "target-features"="+hvxv73,+hvx-length128b" }
```

`llc -O3 -mcpu=hexagonv73 -filetype=obj store.ll` fails with
`Cannot select ... store (s32) ... v32i1 = setcc`. SDK
`hexagon-clang -O3 -mv73 -mhvx -c store.ll` fails at the same operation.
The logical vector occupies four bytes in memory; its native HVX predicate
representation cannot simply be stored as an ordinary word without packing.
This is distinct from HEX-PRED-001's wrong native small-predicate bitcast and
HEX-PRED-002's64-lane integer-to-predicate truncation.

The private Triton source avoids carrying predicates across its branch by
retaining integer masks instead. That workaround compiles and passes device
numerical checks, but is not a compiler fix. Raw source, compiler commands,
logs and exit statuses are retained in the `predicate-store-reproducer`
subdirectory of the20261108 iteration evidence.


## HEX-PRED-005: replicated HVX predicate bits are packed as logical lanes

**Status, 2026-11-08:** wrong output reproduced on QCS9075 with the local
LLVM22 toolchain and the independent system package, Ubuntu LLVM18.1.3. A proposed isolated fix passes device checks; it has not
replaced the qualified compiler or been submitted upstream.

A minimal reproducer loads `<32 x i32>`, compares each element with zero, then
bitcasts the resulting `<32 x i1>` to `i32`. A word predicate uses four physical
HVX predicate positions per logical lane; a halfword predicate uses two.
`HexagonTargetLowering::compressHvxPred` packed those repeated physical bits
instead of selecting one bit per logical lane. This corrupts the packed mask
and can also corrupt `llvm.vector.reduce.and/or` results.

On 128-byte HVX, isolated original-HEAD `HexagonISelLoweringHVX.cpp` linked
against the existing local LLVM libraries produced:

| Logical lanes | Device cases | Wrong packed masks | Wrong AND reductions | Wrong OR reductions |
|---|---:|---:|---:|---:|
| 128 byte lanes (control) | 514 | 0 | 0 | 0 |
| 64 halfword lanes | 386 | 384 | 32 | 32 |
| 32 word lanes | 322 | 320 | 26 | 24 |

The original translation unit is from LLVM revision
`064f02dac0c81c19350a74415b3245f42fed09dc`. This is an isolated translation-unit
comparison, **not a clean full upstream LLVM build**. SDK LLVM19 passed the
32-word and 64-halfword cases tested; the issue is not attributed to that SDK.
The local source-built compiler also reproduced the 32-word failure. The
independent `/usr/bin/llc-18` build reproduced all three rows of the table
exactly, without linking any locally rebuilt LLVM objects.

The proposed fix selects one physical position per logical lane, merges groups
of eight logical lanes, and collects the packed bytes. The final isolated
binary passes all three device suites above and seven focused CodeGen checks
(one new test plus six existing RUN lines). Both 64- and 128-byte configurations
compile; the QCS9075 environment rejected 64-byte HVX locking, so no 64-byte
hardware validation is claimed. No model speedup is attributed to this fix.

Reproducer, command records, final test results and the incremental patch are
retained under `.local/prefill-5h-20261108`; the patch is
`patches/llvm-logical-predicate-pack.patch`. The iteration report identifies the
persistent evidence location. The active LLVM working tree and dependency
recipe remain unchanged by this proposed fix.

## HEX-ASYNC-006: coroutine allocation uses a 64-bit C ABI on Hexagon

**Status, 2026-11-08:** reproduced in unmodified LLVM/MLIR conversion output.
An isolated draft fix passes 32-bit, 64-bit and default-layout IR checks and
the three existing AsyncToLLVM conversion test files. A Hexagon probe with the
correct 32-bit allocator ABI, task counts and reference releases passes device
checks. Not active in the accepted inference model and not filed upstream.
This is separate from the Hexagon-MLIR async task-count defect.

The unmodified `mlir/lib/Conversion/AsyncToLLVM/AsyncToLLVM.cpp` at local LLVM
revision `064f02dac0c81c19350a74415b3245f42fed09dc` hardcodes i64 for coroutine
size/alignment and `aligned_alloc` arguments. A minimal `async.coro.begin` in a
module with `llvm.data_layout = "e-p:32:32"` still lowers to
`llvm.func @aligned_alloc(i64, i64)`. Hexagon's C `size_t` is 32-bit: the linked
SDK helper takes `(i32, i32)`. In the dynamic-threading probe, LLVM IR contains a
call `aligned_alloc(i64 8, i64 56)` alongside an i32/i32 helper definition.
This is an ABI mismatch, not evidence of an attention arithmetic error.

Reproducer and generated IR are retained in this iteration's
`.local/prefill-4h-20261108` evidence under `async-trip-count/coro32.mlir` and
`coro32-original.mlir`. The command is `mlir-opt coro32.mlir
-convert-async-to-llvm`. Expected: allocator argument widths follow the target
pointer/size_t width; the default 64-bit target remains unchanged.

The draft reads the legacy `llvm.data_layout` module attribute; DLTI integration
has not been generalized or validated. The current Hexagon-MLIR pipeline also
leaves module data-layout assignment commented out, so correcting AsyncToLLVM
alone does not correct an input module without a target layout. The device
probe explicitly supplies a 32-bit module layout. The earlier prototype result
without this attribute is superseded and is not evidence for the ABI fix.

Evidence includes the actual generated i32/i32 allocator call, matching SDK
helper, zero live async objects after completed calls, and bitwise-correct
outputs for dynamic row counts 0, 1, 16, 31, 32, 33 and 64. On the relevant
32x1024 shape, 32 repeated calls take a median 1716.5 us serial versus 2792.5 us
through the corrected generic async prototype. This correctness repair does
not establish a performance benefit. Draft patch:
`.local/prefill-4h-20261108/patches/llvm-async-allocation-size-draft.patch`.

The performance model continues to use the previously qualified synchronous
worker ABI. Do not enable the experimental structured async route merely because
it emits a parallel loop; allocator ABI, task counts and token/group lifetime
must all be validated first. No active LLVM source or dependency lock was changed
for this investigation.

## QFloat conversion folding differs between SDK and open LLVM

**Status, 2026-11-09: reproduced code-generation/precision-policy difference;
not a confirmed upstream correctness bug. No issue submitted.**

Reference: llama.cpp `f818065d753700341b5355ce36ca676ecfb5d139`,
`ggml/src/ggml-hexagon/htp/hvx-sigmoid.h` and `hvx-inverse.h`;
QuIC Hexagon Clang 19.0.02, v73. The comparison compiler is local LLVM
`064f02dac0c81c19350a74415b3245f42fed09dc` with the existing project patches,
not an unmodified upstream release.

The first SiLU divergence occurs inside the second reciprocal refinement.
The source expresses a QFloat-to-FP32 conversion of both operands before an
SF×SF multiplication. SDK-generated code combines these conversions and
executes QFloat×QFloat using the original intermediate values. The local open
LLVM preserves the conversions. Both inputs *after* FP32 conversion agree,
but the SDK multiplication retains the earlier QFloat values; comparing only
the converted input dumps therefore initially obscured the cause.

Compiling the **same SDK-emitted LLVM IR** with the two code generators
reproduces the difference without Triton, model weights, quantization, KV or
RPC framework changes. In 128 sampled lanes, the first multiplication inside
the second refinement differs in 32 QFloat results; final SiLU differs in 17
lanes. This establishes a compiler-path difference, not which numerical
policy should be considered normative.

For the requested llama.cpp alignment, `models/qwen3/kernels/llama_silu.py`
explicitly retains the QFloat denominator and reciprocal in the last two
refinements. It does not rely on an implicit SDK optimization and does not
change the general LLVM floating-point policy. Device comparisons against the
pinned SDK reference then agree bitwise on the tested normal/boundary/special
fixtures. The compiler adds packed HVX primitive lowering, not a C SiLU call.

Standalone evidence, outside the repository:
`/home/kevin/mnt/qwen-builds/llama-math-alignment-20261109/`:
`llama-stages.c`, `llama-stages.ll`, `llama-stages2.o`,
`llama-stages-llvm.o`, `native-stages-disasm.txt`, `stages-disasm.txt`,
`qfloat-call-comparison.json`, `sdk-dual-comparison.json`.
The LLVM IR has no model dependency. The direct compiler comparison command is:

```bash
llc -O3 -filetype=obj -relocation-model=pic llama-stages.ll -o llama-stages-llvm.o
```

The SDK reference object is built from the same diagnostic source with
`hexagon-clang -O3 -mv73 -mhvx -fPIC`; headers come from the pinned llama.cpp
checkout. Board captures remain under
`/mnt/nvme/qwen3-work/llama-math-alignment-20261109/stages-llvm-comparison/`.
Do not describe this finding as an LLVM bug without establishing the intended
QFloat conversion semantics and the SDK's permitted folding policy.

### 2026-11-09: packed integer conversion split HVX scheduling (fixed locally)

This was an omission in our packed primitive lowering, not an established LLVM
correctness bug. `llama_silu.py` used `$0.sf = $1.w` for the 32-lane signed integer
to FP32 conversion. Other QFloat primitives were mapped to recognized LLVM
intrinsics, but this conversion remained inline assembly. Hexagon's default
`hexagon-sched-inline-asm=false` treats inline assembly as a scheduling and
packetization boundary (`HexagonInstrInfo.cpp`).

Map the exact pure, standard-constraint, 32-lane primitive to
`llvm.hexagon.V6.vconv.sf.w.128B`; leave other assembly contracts unchanged.
The machine conversion and arithmetic semantics stay the same. Four independent
Triton vectors then expose enough scheduling freedom to improve SiLU:

- Common four-worker QCS9075 adapter, q32 × 3072 FP32: old object 227.5 us,
  intrinsic-only 146 us; maintained intrinsic plus four-vector loop 95.5 us.
- The final candidate's paired pinned llama.cpp math reference is 111 us.
  This is an operator comparison, not complete llama.cpp backend throughput.
- Qualcomm V73 simulator, identical per-worker workload and function ranges:
  `HVX_REG_ORDER` drops from 36,096 to zero reported stalls; core committed packets
  from 81,062 to 55,792. These are simulated counters, not board hardware PMU.
- Alias/output-alignment experiments stayed near 229–230 us. Compiling the
  original inline-assembly IR with the SDK also stayed near 227 us. Neither was
  retained as a production change.

The IR lowering regression checks the recognized conversion and preserves an
unknown-assembly fallback. Board normal, strided, domain-boundary and special-bit
fixtures match the native math reference bitwise. Full evidence and model
validation are recorded in `reports/2026-11-09-silu-codegen/REPORT.zh-CN.md` and
`/home/kevin/mnt/qwen-builds/silu-codegen-20261109/`. No global LLVM scheduling
switch or new numerical approximation was introduced.


## HEX-MEM-005: FP32 lane extraction spills an entire HVX vector

**Status, 2026-11-10:** reproduced performance limitation on the identified local
LLVM toolchain, not a numerical correctness defect. No upstream issue filed;
local LLVM modifications have not been isolated from pristine upstream.

A bitcast from `<32 x i32>` to `<32 x float>` followed by `extractelement` of
lane zero emits an aligned 128-byte stack store and scalar load. Calling
`llvm.hexagon.V6.extractw.128B` on the original raw bits produces `vextract`
without the stack round trip and preserves the exact FP32 bit representation.
The extracted index uses a **byte offset**, not a lane index.

Minimal two-function reproducer, assembly and compiler identity:
`/home/kevin/mnt/qwen-builds/decode-attention-20261110/extract/` contains
`repro.ll`, `repro.s`, and `repro-identity.json`. Command:

```sh
/home/kevin/mnt/qwen-builds/llvm-prefill-20261108/bin/llc \
  -mtriple=hexagon -mcpu=hexagonv73 \
  -mattr=+hvxv73,+hvx-length128b,+hvx-qfloat repro.ll -o repro.s
```

The matching q1 Attention packet capture identifies its hottest four core
stall packets immediately after vector-store/scalar-load lane extraction.
These are simulated cache/coherency stalls, not board PMU counters. See
`maintained-packet-summary.json` and `maintained.dis` in the parent directory.
The controlled candidate changes only the single-lane constant-gather lowering
in Hexagon-MLIR to the explicit intrinsic; board A/B and identical-output checks
are recorded in `extract/`. The ordinary multi-lane gather remains a shuffle.
Do not treat a source-level reduction rewrite as equivalent to removing this
backend spill. Further board/model validation determines the actual benefit.


HEX-MEM-005 controlled result (same date): the vextract-only candidate preserved
all saved outputs, but board q1/KV1024 was1431.5us versus1438us control, and
KV960 was1341us for both. This is not a demonstrated useful board speedup.
Simulation reduced core stalls54977→34209 and cache stalls25879→4335; the
simulation improvement did not transfer to the four-worker board run. The
special-case intrinsic lowering was removed from maintained code. Reproducer
and candidate source remain in the evidence directory for compiler discussion.
