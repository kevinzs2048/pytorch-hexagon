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
`aad5959e00e95ec5ffdec6595260969c6e08174c` and hashes all 753 ordinary tracked
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
an unset PYTHONPATH. At the initial split, all 169 host tests passed. All 16 board forwards matched every reference input/
output hash (logits and full KV), position, token and checksum. The completed
record is `qwen3/validation/repository-split.json`.

This is a source-ownership/import-path refactor. The two compiled model libraries,
weights, numerical policies and generated packet values are not intentionally
changed. A clean full compiler/model rebuild is still unqualified; packaging and
repository separation do not close that existing build gap.

## Follow-up cleanup — 2026-10-27

Eight execution helpers moved from `tools/qwen3` to `qwen3/runtime/support`;
seven tests moved into `test/python/qwen3`. Shared checks live in
`qwen3/runtime_checks.py`, so inference no longer imports historical regression,
performance or release launchers. Existing internal callers reuse those checks.
The runtime cache uses `qwen3/runtime-sources.json`; unrelated tool/test edits no
longer change its identity. A regression checks that local imports remain covered.

The missing QK MLIR fixture and pure SiLU IR generator were recovered from frozen
runtime revision `feb45fd` and placed in the compiler repository. The old SiLU
inspection launcher was not restored merely to supply its fixture function.

Removed code is recoverable from application commit `5e3f49c`:

- `hexagon_native1024_closed_probe.validate_closed_probe`: no callers in the
  retained source; it depended on an absent historical device auditor. The used
  `validate_closed_record` remains.
- `tools/qwen3/test_runtime_dependencies.py`: orphan tests for an absent historical
  `runtime_dependencies` module with no retained production callers. Current
  bundle integrity, missing/corrupt assets and runtime guards remain tested.

The FastRPC tracing tool retains its kernel/module checks but accepts a supplied
absolute output directory instead of requiring a dated directory on one board.
The process-lifecycle tests use disposable clean Git fixtures, so ordinary local
edits no longer prevent those tests from reaching the code they exercise.

Historical numerical/compiler stages are not removed based only on naming or lack
of imports from the warm inference path. The source-build consolidation gap in
BUILD_STATUS.md remains open. Compiler algorithm and DSP kernel code are unchanged
by this cleanup; the compiler commit adds only test fixtures.

Cleanup validation: 531 host tests passed in both the development tree and an
offline clone; 161 compiler/SDK-dependent tests skipped. The final runtime source
passed 16 forwards in a fresh process/cache and 16 more in a second process using
the same cache. Every input/output hash (logits and full KV), position, token and
checksum matched the included reference. AOT counts were two misses on the first
run and two hits on the second. See `qwen3/validation/code-cleanup.json` for source
identities and evidence hashes. This does not qualify a clean compiler rebuild.

## Orphan implementation cleanup

After the source-build path was established, repository-wide imports, identifier
uses and subprocess/filename references were checked again. The old standalone
static-model reuse chain, its archive/relink helpers, the unused q1 scalar bridge,
an unused tutorial utility copy and disconnected capacity-128 corpus modules
were removed. The still-used case dataclass moved unchanged into its only caller.
Unused helper functions were removed from otherwise retained modules.

This is separate from numerical/compiler changes. Every remaining top-level
function in the edited modules was checked against its pre-cleanup AST; the
shared file-hash function also has the same AST as the removed duplicate. The
runtime manifest shrank from 49 files to 46. The complete host suite passed 696
tests with no skips, plus nine IR checks. Final board regression is recorded in
the source-build qualification, not inferred from this static review.

The removal list and original file identities are in
`qwen3/validation/orphan-cleanup.json`; recover the old code from application
commit `68ff472` if historical investigation needs it. Active native generation,
KV lifecycle, numerical policy passes and DSP helper implementations remain.
