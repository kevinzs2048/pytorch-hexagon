# Application/compiler repository boundary

The combined-tree organization at `1db3dae` was incorrect: a single reproduction
entry does not require owning compiler and application source in one repository.
This split supersedes that layout.

| Owner | Maintained code |
| --- | --- |
| `qwen3-hexagon-runtime` application | PyTorch model capture/adaptation, native generation, logical KV lifecycle, deployment, Python orchestration, application tests and reports |
| Hexagon-MLIR compiler fork | Dialects/passes, scatter/bufferization/codegen, generic tensor ABI, DSP numerical kernels, LLVM/torch-mlir patches and compiler tests |

The compiler lives in its own Git repository and is consumed as the
`third_party/hexagon-mlir` submodule. The application locks commit
`e593bd292fa4136ef142187c7c4337cfc73ad1e5` and hashes all 751 ordinary tracked
compiler files. There are no application-owned copies of `qcom_hexagon_backend/`,
compiler policy C++ or DSP kernel sources. Application build orchestration resolves
those files through `qwen3/compiler_dependency.py`.

The dependency commit is local, not pushed. The offline delivery includes separate
application and compiler Git bundles; the root README gives the standard submodule
initialization commands. Do not claim the GitHub remote already contains this commit.

## Dead-code cleanup

The reusable audit module imported three historical device-launch scripts only to
obtain a context hash and caller path. Those imports pulled in absolute host/board
paths and unrelated experimental execution code. The audit now imports the context
hash from its maintained definition and requires `--caller` / `--caller-sha256`.
These obsolete launchers have been removed from the application:

- `run_qwen_dynamic_debug59_q5_device.py`
- `run_qwen_dynamic_layer23_q32_device.py`
- `run_qwen_dynamic_native_generate_dsp.py`

Their original bytes remain in compiler-history commit `1db3dae`. The audit's
cache/packet/lifecycle routines remain, with existing tests. Other internal
maintenance tools have not been declared dead solely because the public command
does not invoke them; some are needed for compiler recipe reconstruction or
historical evidence review. Further cleanup must trace those uses explicitly.

## Validation scope

The maintained host command includes the audit/lifecycle tests affected by this
cleanup. The fresh board regression uses the independent application source,
separately pinned compiler source, unchanged accepted assets, system Python and
an unset PYTHONPATH. All 169 host tests passed. All 16 board forwards matched every reference input/
output hash (logits and full KV), position, token and checksum. The completed
record is `qwen3/validation/repository-split.json`.

This is a source-ownership/import-path refactor. The two compiled model libraries,
weights, numerical policies and generated packet values are not intentionally
changed. A clean full compiler/model rebuild is still unqualified; packaging and
repository separation do not close that existing build gap.
