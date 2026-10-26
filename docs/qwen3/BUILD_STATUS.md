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
| LLVM/MLIR and Qualcomm SDK | Explicit prebuilt toolchain dependencies; LLVM is not rebuilt by this command |

The pinned toolchain remains SDK 6.5.0.0 and Hexagon Tools 19.0.02. A newer public
SDK, LLVM or framework release is a separate configuration and is not silently
substituted.

## Validation boundaries

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

## Retained historical code

Old recipe consumers and diagnostic tools remain available for inspecting prior
evidence and targeted numerical diagnosis. Some require their historical source
revision and artifacts. They are outside the new source-build dependency path.
The removed reference-wrapper recipe, its three exclusive templates and the
redundant RoPE build launcher are recoverable from application revision `9a5ce27`.
Nine disconnected runtime modules and unused functions were removed separately;
their identities and validation are in
[the cleanup record](../../qwen3/validation/orphan-cleanup.json).

The explicit runtime dependency list decreased from 49 to 46 files. Including
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
