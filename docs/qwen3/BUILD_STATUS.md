# Compiler source and build status

The application and compiler remain separate repositories. The application pins
the compiler at `third_party/hexagon-mlir`; compiler passes, DSP kernels and LLVM
patches belong to that fork.

The supported build entry is now `python -m qwen3.build`. See
[source-build instructions](SOURCE_BUILD.md) for configuration, captures, build
outputs and the DSP comparison command. `python -m qwen3.tools build` is an alias
for the same implementation.

| Component | Current source-build route |
| --- | --- |
| Triton + Hexagon-MLIR plugin | Pinned Triton Git export and the selected compiler checkout; CMake/Ninja |
| q1 and q2–32 compiler configurations | Explicit `qwen-compiler-q1` / `qwen-compiler-q2` source targets; no copied historical compiler objects |
| Numerical policy tools | Four source-built optimizers in `build/policy` |
| DSP helpers | Explicit production targets in `build/runtime`, including INT8, RoPE, attention guards and copy helpers |
| PyTorch → torch-mlir | Actual ARM AOT captures, locally verified immutable weights and a checked frontend cache |
| Model objects, callers and constants | Generated and linked in `build/models/q1` and `build/models/q2` |
| LLVM/MLIR/LLD | Complete pinned source patch and separate source-build recipe; the application verifies the resulting toolchain receipt |
| Qualcomm SDK / Tools / HexKL | Explicit external dependencies |

The pinned toolchain remains SDK 6.5.0.0 and Hexagon Tools 19.0.02. A newer public
SDK, LLVM or framework release is a separate configuration and is not silently
substituted.

## Reproducible LLVM dependency

Compiler `23cb318` packages the complete 29-file LLVM source patch, pinned
upstream revision and patched tree, CMake recipe, and build/identity verifier.
The application requires a matching receipt for actual libraries, generated
headers, compiler/linker executables and resolved host shared libraries. A Git
revision alone is no longer accepted as dependency verification.

The existing qualified installation remains untouched. Its explicit
`qualified_prebuilt_snapshot` receipt records the accepted binaries without
claiming historical source-to-binary provenance. A separate checkout has been
prepared from the pinned base, patched, and fully built into a new directory;
its receipt is labelled `source_build`. The source tree exactly matches the
qualified checkout, including its local changes. This complete compatibility
snapshot is separate from the four selected upstream correctness fixes; it is
not a claim that all historical LLVM changes are needed or ready for upstream.

The rebuilt LLVM passes 1,381 Hexagon code-generation tests, with nine expected
failures and two unsupported tests, and the Hexagon LLD preemptible-thunk test.
The backend built against it passes 611 configured host tests and nine IR
checks without failures or skips. All 14 DSP helper products are byte-identical
to the preceding build; the eight host compiler/policy binaries were rebuilt.
All 46 execution-source files remain unchanged.

Both complete model domains were compiled from the retained trusted ARM captures
using the rebuilt compiler and linker. For q1 and q2, the model object, generated
caller, all three constant libraries and final DSP library match the preceding
qualified delivery byte-for-byte. The final model SHA256 values remain
`c2d7f0176b803a9aee45177cfff96adc7b7cfc2a25ea7f4f829084a6693331ee` (q1) and
`fc6a73f8205f4b75770aa78216cd3f643b29c243f3d36b24fa974178290b249a` (q2).
The preceding 336-forward acceptance therefore covers these exact binaries;
it is not counted as a newly executed full regression.
A fresh QCS9075 process using the current execution sources and both rebuilt
libraries additionally passed 16 complete forwards: logits, all 56 KV tensors,
inputs, positions, tokens and stopping matched. Its fresh AOT namespace recorded
two misses and no hits. No CPU model forward fallback was used.

Evidence is retained in `.local/llvm-toolchain-20261030/`, including the LLVM
build/identity receipt, LLVM and host test results, `binary-comparison.json`,
compressed build evidence, and `delivery/models/artifacts.json`. The canonical
local toolchain configuration now selects the source-built dependency. The
original installation, original configuration and qualified model delivery
remain available for rollback. Large intermediate model/weight copies are
removed after verifying the retained delivery; the LLVM and backend build
products remain available for incremental work.

## Official LLVM GenInsert policy

The preceding qualified branch pins compiler `839af5a`. Each model compilation runs in an
isolated worker configured with LLVM's official `-hexagon-insert=false` switch.
The model builder no longer needs the private `hexagon-no-gen-insert` function
attribute. The compiler rejects late or conflicting configuration; the legacy
function policy remains available for comparison. Other LLVM dependencies have
not been removed, and this is not qualification of an unpatched LLVM release.

The paired sources pass 598 configured host tests and nine IR checks without
failures or skips, including five tests of the actual compiler extension and
its LLVM machine-pass pipeline. Both full model objects, generated callers and
constant libraries match the preceding qualified build byte-for-byte. The q1
library differs only in non-loaded build-path comments. The q2 library also
differs in 143 instruction bytes inside the unchanged SDK-compiled multi-token
INT8 helper. Four repetitions of an identical Hexagon Clang 19.0.02 command
produced different instruction bytes; this is an observed reproducibility issue,
not evidence of a functional compiler failure.

With the new q1 library and the original q2 library, EOS and 32-token generation
passed 42 complete output comparisons. The full 336-forward suite using both
new libraries also passed: cold/warm processes, chunk tails, EOS, 32-token and
64-token generation, 1024-position prefill, decode to the capacity boundary and
request reset. Complete logits/KV packet hashes, inputs, positions, tokens and
stopping matched the references. Cold execution had two AOT misses; warm
execution had two hits and no misses. No CPU model forward fallback was used.
The qualified baseline below remains available.

Evidence is under `.local/submission-review-20261028/process-codegen/`:
`host-verification/result.json`, `candidate-regression.json`,
`binary-comparison.json`, `sdk-repro/result.json`, and
`delivery/models/artifacts.json`. Large model intermediates and the frontend
tensor cache have been removed after checking the retained delivery and compressed
build evidence. The compiler build directory was subsequently retargeted to the
source-built LLVM dependency described above; the preceding model delivery and
qualification receipts remain intact.

## Previously reviewed compiler and cleaned application

The previously qualified branch pins compiler `b4008a5`, consisting of the 12 inference-support
commits through `de8c2c2` and one optional diagnostic-source commit. Tests needing
Qwen-specific fixtures now read them from the application; the five retained C
diagnostic sources stay in the compiler fork. Old recipe and experimental
reference consumers have been retired. No missing compiler helper is hidden by
a skipped test.

The paired sources pass 588 configured host tests and nine IR checks, with no
failures or skips. The new q1 and q2 models were built from the trusted captures
and the reviewed compiler. Adding the optional diagnostic files leaves all 22
compiler/policy/production-runtime products byte-identical; they are not linked
into either model. Both rebuilt model libraries passed the complete 336-forward
QCS9075 suite: cold and warm processes (16 each), chunk tails (6), EOS (9),
32-token generation (33), 1024-position prefill (33), decode from a 993-token
prompt through the capacity boundary and reset (95), and two 64-token requests
(128). Whole logits/KV packet hashes, all input checksums, positions, request
boundaries, tokens and stopping matched the accepted references. No CPU model
forward fallback was used. The first process had two AOT misses; the second had
two hits and no misses. Existing accepted assets remain the independent reference.

The 588 host tests and nine IR checks also passed in an independent offline
Git-bundle clone, without failures or skips. Board deployment preceded the final
documentation and two test-fixture fixes; all 46 runtime manifest entries and
the build/production sources are identical to the qualified source snapshot.
The fixture fixes preserve 84 cases: memref-copy uses the maintained reference
implementation, and SiLU guards use a small independent caller fixture.

Local qualification evidence is under
`.local/submission-review-20261028/`: `candidate-regression.json` (336 forwards,
SHA256 `c1ea5e877853761caf5b855a0e7049b4df996ddac85df45ed840e803f446b7c0`),
`candidate-host-fixed-verification/`, `offline-host-verification/`, and
`delivery/models/artifacts.json`. Large build intermediates are not delivery
requirements; the retained model libraries, primary objects, callers and receipts
are in `delivery/models/`. The constants are byte-identical to the original asset
bundle. This qualification makes no new model, platform or performance claim.

Current application/runtime Python: 65 files, 8,229 physical lines. Maintainer
Python: 14 files, 2,222 lines. The previous tree had 261 Python files and 32,716
lines in these two groups. Counts include comments and blank lines, exclude tests
and JSON, and do not assert that every retained line executes during inference.
The execution manifest remains 46 files, including five compiler Python files.

## Previously qualified compiler (`ce317a3`)

The existing execution bundle remains the numerical reference. Its `doctor`
output still reports `clean_compiler_rebuild_verified: false`, because that bundle
contains the previously accepted binaries. A new source-build result does not
rewrite that historical bundle's provenance.

The single-command source build completed in a clean offline Git-bundle clone.
Both actual ARM graphs took a real torch-mlir cache miss. Their prepared IR and
all eight model/constant objects matched the preceding development builds
byte-for-byte. The final model libraries, generated callers and source-built DSP
helpers then passed the following QCS9075 comparisons:

| Check | Complete model forwards | Result |
| --- | ---: | --- |
| Fresh process and empty candidate AOT cache | 16 | Identical; two AOT misses |
| Second process reusing the candidate cache | 16 | Identical; two AOT hits |
| q32 prefill, q1/q2 tails and request reset | 6 | Identical |
| Native early EOS under the original token budget | 9 | Identical |
| 32-token continuous generation | 33 | Identical |
| 1024-position prefill and request reset | 33 | Identical |
| Decode from a 993-token prompt through the capacity boundary, then reset | 95 | Identical |
| Two 64-token generation requests | 128 | Identical |
| **Total** | **336** | **Complete input/output agreement** |

Every comparison checks whole packet hashes, covering logits and all 56 KV
tensors, positions, request boundaries, generated tokens, stopping and all three
input checksums. The 64 saved input/output packets from the first two runs were
also independently read and rehashed. Long state tests retain complete hashes
rather than raw KV packets. No CPU model forward fallback was used.

The configured host verification passed **699 tests and nine IR checks**, with
zero failures or skips, in both the development checkout and the offline clone.
An additional toolchain audit matched 9,002 selected LLVM source/install,
SDK/Tools, HexKL, compiler, linker and Python files to the existing dependency
lock. This audit does not constitute an LLVM source rebuild.

After committing the qualification and removing development build copies, old
board deployments and obsolete AOT namespaces, verification was repeated:
699 tests and nine IR checks passed, both current-source graph/weight restores
validated the original frontend cache artifacts, and a new board process passed
16 more complete forwards with two AOT hits. The total for the final source-built
models is **352 forwards**. No additional model code was compiled during this
post-cleanup repeat.

Exact source revisions, model hashes and receipt hashes are recorded in
[the source-build qualification](../../qwen3/validation/unified-source-build.json).
The compiler/model build ran at application `68ff472`; the cleaned runtime was
executed at `6701f22`. The retained build functions were checked for AST equality
across that cleanup. This qualifies the existing Qwen3-0.6B scope, not another
model, hardware platform or performance result.

## Earlier cleanup and recovery record

Old recipe consumers and experiment supervisors are now retired. Use application
commit `c7b3f8f` and its original artifacts to inspect their historical execution.
The supported workflow and retained diagnostics are described in
[SOURCE_BUILD.md](SOURCE_BUILD.md#what-was-replaced). The counts and generated-data
cleanup below describe the earlier qualification, before this retirement.
The removed reference-wrapper recipe, its three exclusive templates and the
redundant RoPE build launcher are recoverable from application revision `9a5ce27`.
Nine disconnected runtime modules and unused functions were removed separately;
their identities and validation are in
[the cleanup record](../../qwen3/validation/orphan-cleanup.json).

During that earlier cleanup, the explicit runtime dependency list decreased from 49 to 46 files. Including
the new build entry, application/runtime/maintainer Python decreased from 268
files and 33,174 physical lines to 261 files and 32,700 lines. These counts include
comments and blank lines, and exclude tests and reference data. The added state
JSON files preserve existing test inputs and output hashes; they are not runtime
implementation or a new prompt corpus.

Generated-data cleanup removed 22.65 GB of host file contents and freed 0.55 GB
of allocated board storage. One canonical build, original captures, verification
receipts and selected raw baseline/candidate packets remain. The 120 retained
development records/logs are in a verified 1.2 MB archive. Local locations:

- `.local/unified-build/final/`: compiler/model products and build receipts.
- `.local/unified-build/inputs/`: actual captured build inputs.
- `.local/unified-build/validation/`: board/restore receipts and development evidence.
- `.local/unified-build/application-source.bundle` and `compiler-source.bundle`:
  complete offline source histories.
- `.local/unified-build/toolchain.json`: this machine's explicit configuration.

The canonical build keeps useful IR, objects and frontend cache data for further
compiler work. Its `work/` development predecessor and duplicate source-transfer
archives were removed. Cleanup inventories are retained locally and identified
by hash in the qualification record.

The source rebuild exposed a strict scalar FP32 `sqrt`/division rewrite that
bypassed the floating-point permission check. Compiler commit `277c108` fixes it
and adds strict/partially-fast regression cases. Commit `ce317a3` adds the explicit
source targets and records the restored accepted pass-source versions. These
changes are separate rollback points.

For recovery, use a separate clone of the application bundle, check out
`9a5ce27372d495b2857e433bd5da02242f1b107b`, then initialize its pinned compiler
submodule (`aad5959e00e95ec5ffdec6595260969c6e08174c`). The original accepted model
and weight files remain available. The offline Git bundles retain the complete
source history, including removed implementations.
