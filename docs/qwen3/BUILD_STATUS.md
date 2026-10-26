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

Source-built compiler configurations and helpers have passed the controlled
16-forward comparison with the existing prepared IR, including the reduced
callers. Both freshly captured ARM graphs have also passed a real torch-mlir
lowering cache miss and reproduced the prepared IR byte-for-byte. The final
single-command build and its independent DSP qualification are recorded
separately; compilation is never treated as numerical acceptance.

The unified host verification runs the whole retained test suite and explicit IR
FileChecks against the newly built tools, rejecting skipped or failed tests. The
development pass completed 694 tests and nine IR checks without failures/skips.

## Retained historical code

Old recipe consumers and diagnostic tools remain available for inspecting prior
evidence. They are not part of the new source-build dependency path. The removed
reference-wrapper recipe, its three exclusive templates and the redundant RoPE
build launcher are recoverable from application revision `9a5ce27`.

The source rebuild exposed a strict scalar FP32 `sqrt`/division rewrite that
bypassed the floating-point permission check. Compiler commit `277c108` fixes it
and adds strict/partially-fast regression cases. Commit `ce317a3` adds the explicit
source targets and records the restored accepted pass-source versions. These
changes are separate rollback points.
