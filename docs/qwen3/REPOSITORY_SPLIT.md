# Application/compiler repository boundary

The combined-tree organization at `1db3dae` was incorrect: a single reproduction
entry does not require owning compiler and application source in one repository.
This split supersedes that layout.

| Owner | Maintained code |
| --- | --- |
| `pytorch-hexagon` application | PyTorch model capture/adaptation, native generation, logical KV lifecycle, deployment, Python orchestration, Qwen-specific native compiler/DSP extensions, toolchain reproduction, application tests and reports |
| Hexagon-MLIR compiler fork | General dialect/lowering/codegen support, generic tensor ABI, runtime correctness fixes and compiler tests |

The compiler lives in its own Git repository and is consumed as the
`third_party/hexagon-mlir` submodule, pinned by the current application lock.
The application owns `models/qwen3/native/` and `dependencies/llvm/`; source language does
not determine repository ownership. The historical placement of all numerical
policies and DSP helpers in the compiler fork is superseded by this boundary.

Compiler `f13109d` is published on the fork branch `review/qwen3-integration`.
The application has its own repository and remote. Initialize the compiler
submodule as described in the root README. Within this application, shared
infrastructure lives in `hexagon_runtime/`; Qwen-specific implementation lives
in `models/qwen3/`. Adding a model does not require renaming this repository.

## Historical cleanup record

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
record is `models/qwen3/validation/repository-split.json`.

This is a source-ownership/import-path refactor. The two compiled model libraries,
weights, numerical policies and generated packet values are not intentionally
changed. A clean full compiler/model rebuild is still unqualified; packaging and
repository separation do not close that existing build gap.

## Follow-up cleanup — 2026-10-27

Eight execution helpers moved from `tools/qwen3` to `models/qwen3/runtime/support`;
seven tests moved into `test/python/qwen3`. Shared checks live in
`models/qwen3/runtime_checks.py`, so inference no longer imports historical regression,
performance or release launchers. Existing internal callers reuse those checks.
The runtime cache uses `models/qwen3/runtime-sources.json`; unrelated tool/test edits no
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

Historical numerical/compiler stages were not removed based only on naming or
lack of imports from the warm inference path. At that initial cleanup step, the
source-build consolidation gap remained open and the compiler change added only
test fixtures. See [build status](BUILD_STATUS.md) for the subsequent source-build
work and its separate compiler changes.

Cleanup validation: 531 host tests passed in both the development tree and an
offline clone; 161 compiler/SDK-dependent tests skipped. The final runtime source
passed 16 forwards in a fresh process/cache and 16 more in a second process using
the same cache. Every input/output hash (logits and full KV), position, token and
checksum matched the included reference. AOT counts were two misses on the first
run and two hits on the second. See `models/qwen3/validation/code-cleanup.json` for source
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
runtime manifest shrank from 49 files to 46. The complete host suite passed 699
tests with no skips, plus nine IR checks. Final board regression is recorded in
the source-build qualification, not inferred from this static review.

The removal list and original file identities are in
`models/qwen3/validation/orphan-cleanup.json`; recover the old code from application
commit `68ff472` if historical investigation needs it. Active native generation,
KV lifecycle, numerical policy passes and DSP helper implementations remain.
