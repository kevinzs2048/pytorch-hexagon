# Compiler source and build status

The application and the compiler have separate repositories. This application
pins the compiler under `third_party/hexagon-mlir`; compiler sources, DSP kernels
and LLVM patches are owned by that fork. No historical experiment worktree is
required by the portable execution entry.

| Component | Source in this repository | Status |
| --- | --- | --- |
| Hexagon-MLIR | `qcom_hexagon_backend/` (in the compiler repository) | Complete retained development source, including previously uncommitted scan/scatter lowering and codegen changes |
| Numerical policy optimizers | `tools/qwen3/policy/` (in the compiler repository) | Accepted source imported; `tools/qwen3/build.py` builds the policy stages against a configured LLVM installation |
| DSP helper kernels | `tools/qwen3/runtime/` (in the compiler repository) | Retained source, including quantized linear/attention/copy helpers |
| Captured graph compilation | `qwen3/runtime/arm/compile_qwen_symbolic_carrier.py`, `tools/qwen3/native_compile_session_worker.py` | Source collected; original build recipes still need clean-build consolidation |
| LLVM/MLIR | `dependencies/sources.lock.json`, `dependencies/patches/llvm-local.patch` (in the compiler repository) | Base `064f02d`, complete current local diff and new regression files preserved |
| torch-mlir | lock and importer patch in the compiler repository | Pinned Python release `20260908.870`; upgrades need separate validation |
| Transformers | prefill-position patch under `dependencies/`; exact files in execution bundle | Native generation semantics retained; not a replacement generation implementation |

The accepted model libraries were built using patched `libtriton.so` SHA256
`4043395bddd404bcb8370f98fe068a3ddf2a481f10e1e11847a355d965f034ce`,
Hexagon SDK 6.5.0.0 and Tools 19.0.02. Public main's newer SDK/framework versions
are a different configuration and are not silently substituted.

Remaining clean-build work:

1. Recreate the external compiler build from the pinned source and local patches;
   the historical standalone lowering binary/build directory is missing.
2. Replace the historical multi-stage recipe's external intermediate paths with
   one generated build directory; rebuild model constants and both graph domains.
3. Run the compiler tests and same-input full-output/KV comparisons on the newly
   built libraries before promoting a new source-built release.

`python -m qwen3 doctor` validates execution assets; it explicitly reports
`clean_compiler_rebuild_verified: false`. The portable execution bundle does not
require a compiler host during inference. This preserves the existing compiled
route and makes its deployment reproducible without claiming a new build result.
