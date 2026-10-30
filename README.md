# Qwen3 Hexagon Runtime

中文操作入口：[使用指南](USAGE.zh-CN.md)（现有开发板试跑、回归、离线源码交付和重新编译）。

This is the **Qwen3 application and integration repository**. It uses a pinned
fork of Qualcomm Hexagon-MLIR as a compiler/backend dependency. It is not the
official Hexagon-MLIR repository and does not own a copied compiler source tree.

| Repository | Responsibility |
| --- | --- |
| **qwen3-hexagon-runtime** (this repository) | PyTorch/Qwen3 integration, generation and KV lifecycle, Qwen-specific compiler/DSP extensions, toolchain reproduction, deployment and regression |
| **Hexagon-MLIR downstream fork** | General dialect/lowering/codegen support, tensor ABI and compiler/runtime correctness fixes |

The application integrates PyTorch/Transformers with Hexagon DSP: PyTorch
supplies the model forward and parameters, Transformers drives generation on
ARM, and this repository captures, compiles and invokes the DSP forward while
managing inputs and KV state. The current model integration is Qwen3-0.6B.

The compiler dependency is a Git submodule at `third_party/hexagon-mlir`, pinned
by both its gitlink and `dependencies/hexagon-mlir.lock.json`. That directory is
a separate repository checkout, not application-owned compiler code. Our local
review checkout is `/home/kevin/hexagon/hexagon-mlir-review`; its package
names retain upstream compatibility. General backend changes stay in that fork.
Model-specific compiler extensions are built from this application's `native/` directory.

For compiler and model source builds, use `python -m qwen3.build`; see the
[source-build instructions](docs/qwen3/SOURCE_BUILD.md). It produces both runtime
query domains in one output tree without historical compiler/model object files.

## Reproduce execution

The application retains one public command: `python -m qwen3`. Obtain this
repository, the pinned compiler checkout, and the matching execution asset bundle.
The latter contains weights, compiled libraries, ARM/DSP runtime, decomposition
policy and exact ARM Python packages; binaries do not belong in Git.

For the current offline delivery (the new commits have not been pushed):

```bash
git clone -b review/qwen3-native-ownership /path/to/application-source.bundle qwen3-hexagon-runtime
cd qwen3-hexagon-runtime
git config submodule.hexagon-mlir.url /absolute/path/to/compiler-source.bundle
git -c protocol.file.allow=always submodule update --init third_party/hexagon-mlir
```

On the qualified QCS9075 board, with ARM64 Python 3.12.3 and the matching
OS/FastRPC environment:

```bash
python3 -m qwen3 doctor --bundle /path/to/qwen3-qcs9075 --execution
python3 -m qwen3 generate --bundle /path/to/qwen3-qcs9075 \
  --prompt '你好，请介绍一下你自己。' --max-new-tokens 32 --output results/first
python3 -m qwen3 regress --bundle /path/to/qwen3-qcs9075 --output results/regression
```

No private PYTHONPATH or historical report directory is needed. The CLI verifies
the compiler source lock and execution assets before loading the model. The
complete 28-layer Qwen3-0.6B forward runs on DSP; ARM owns Transformers generation,
tokenization and stopping. Scope: batch 1, W8A8 linear layers, FP32 intermediates/KV,
capacity 1024, reusable q1 and q2–32 execution domains. There is no CPU forward fallback.

## Application code

- `qwen3/cli.py`, `qwen3/bundle.py`: public entry, portable assets and checks.
- `qwen3/compiler_dependency.py`: explicit compiler dependency resolution.
- `qwen3/runtime/arm/`: graph capture, Qwen3 adaptation, native generation and KV integration.
- `qwen3/runtime/support/`: AOT cache, byte guards, input packing, timing and constant-loader support.
- `qwen3/runtime_checks.py`: shared generation and full-output validation; no experiment launchers.
- `qwen3/runtime-sources.json`: explicit execution-source dependency set for cache invalidation.
- `qwen3/build.py`, `qwen3/build_model.py`: unified compiler and model source build.
- `native/`: Qwen compiler profiles, numerical policies, DSP helpers and diagnostics.
- `tools/qwen3/`: application build/deployment/regression orchestration (Python).
- `test/`, `qwen3/validation/`: tests and included reference hashes.
- `dependencies/`: compiler/toolchain locks, LLVM reproduction recipe and Transformers patch.
- `docs/qwen3/`: reproduction instructions and engineering reports.

General compiler C/C++/MLIR sources are maintained in the compiler repository.
Qwen numerical policies and DSP helpers are application-owned native extensions.
The full LLVM compatibility recipe is in this application's `dependencies/llvm/`. StaticCache reset/update
coordination belongs to the application; scatter lowering and bufferization
belong to the compiler.

## Validation and current limits

The current native-extension relocation pins compiler `f13109d`. A fresh build
produces all 22 backend/policy/runtime products; 615 configured host tests and
nine IR checks pass. Both numerical tools match their preceding outputs on the
eight moved fixtures. This is build/integration validation; full models were not
recompiled or rerun on DSP for this relocation.

Before the native-extension relocation, compiler `23cb318` and application
`426f7f5` qualified the complete LLVM dependency recipe and product identity checks. An independent LLVM
rebuild passes 1,381 Hexagon tests; the backend passes 611 host tests and nine IR
checks. Both freshly compiled model libraries are byte-identical to the preceding
336-forward-qualified pair. See [build status](docs/qwen3/BUILD_STATUS.md) for the
qualification evidence and the distinction between a rebuilt dependency and
an existing binary snapshot.

The preceding qualified branch pins compiler `839af5a` and uses LLVM's official process-wide
GenInsert switch in isolated model-compilation workers. Its 598 configured host
tests, nine IR checks and the complete 336-forward DSP regression pass. The
model objects are unchanged; an SDK helper rebuild changed q2 instruction bytes,
so both final libraries were fully requalified. See
[build status](docs/qwen3/BUILD_STATUS.md) for evidence and the SDK reproducibility finding.

The previously qualified branch pins compiler `b4008a5`: the 12 inference-support commits through
`de8c2c2` plus a separate optional diagnostic-source commit. The paired application
passes 588 host tests and nine IR checks, also repeated in an offline clone.
Both rebuilt model domains passed 336 complete DSP forwards against the accepted
references: full logits/KV packet hashes, inputs, tokens and stopping all matched.
Coverage includes cold/warm processes, chunk tails, EOS, 64-token generation,
capacity boundaries and request reset. This qualifies the reviewed sources;
the existing execution bundle remains the independent reference.


```bash
python -m qwen3 test
```

The host command collects the maintained application tests, including the numerical
policy tests. Tests requiring an explicit compiler/SDK skip when those tools are
not configured; a skipped test is not numerical qualification. Repository
separation has its historical record in `qwen3/validation/repository-split.json`.

The earlier source build using compiler `ce317a3` passed 699 configured host tests and nine IR checks in
both the development tree and an offline clone. Its two newly built model
libraries passed 336 complete DSP forwards: cold/warm processes, chunk tails,
EOS, long generation, capacity boundaries and request reset. All input/output
hashes, including logits and the full KV cache, matched the original references.
After temporary artifacts were removed, 16 further DSP forwards and the complete
host/IR checks passed again (352 final-source forwards in total).
See [the qualification record](qwen3/validation/unified-source-build.json) and
[build status](docs/qwen3/BUILD_STATUS.md) for exact scope and source identities.
[Reproduction](docs/qwen3/REPRODUCE.md) explains bundle export and regression.

Maintainer scripts use a separate entry that supplies paths to child processes:

```bash
python -m qwen3.build --help
python -m qwen3.tools verify --help
python -m qwen3.tools profile_model_memory --help
```

The public inference entry imports no Python code from `tools/`. The retained tools cover targeted diagnostics and performance analysis.
The source-build route is `python -m qwen3.build`. Historical recipe builders,
object-reuse launchers and experiment supervisors were retired; recover their
original source and artifacts from application commit `c7b3f8f` when needed.
See [retirement scope](docs/qwen3/SOURCE_BUILD.md#what-was-replaced).

The [source-build workflow](docs/qwen3/SOURCE_BUILD.md) builds the compiler profiles,
numerical tools, DSP helpers and both model domains in one output tree. This application
includes a separate LLVM/MLIR/LLD source-build recipe and
checks its source and product receipt. Qualcomm SDK / Tools / HexKL remain
external dependencies. A build and a DSP
numerical qualification have separate receipts; see
[build status](docs/qwen3/BUILD_STATUS.md). Existing execution assets remain intact.

The previous combined tree is preserved at compiler-history commit
`1db3dae9ff09ba17c307c800777c443136050537`. Source origin records and original
copyright/license notices are retained. Do not publish this application as an
official Qualcomm release.
