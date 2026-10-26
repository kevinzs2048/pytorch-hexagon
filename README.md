# Qwen3 Hexagon Runtime

This is the **Qwen3 application and integration repository**. It uses a pinned
fork of Qualcomm Hexagon-MLIR as a compiler/backend dependency. It is not the
official Hexagon-MLIR repository and does not own a copied compiler source tree.

| Repository | Responsibility |
| --- | --- |
| **qwen3-hexagon-runtime** (this repository) | PyTorch/Qwen3 integration, native generation, logical KV lifecycle, deployment, profiling orchestration and application regression |
| **Hexagon-MLIR downstream fork** | MLIR dialects/passes, lowering/codegen, generic buffer ABI, DSP kernels and numerical compiler extensions |

The compiler dependency is a Git submodule at `third_party/hexagon-mlir`, pinned
by both its gitlink and `dependencies/hexagon-mlir.lock.json`. That directory is
a separate repository checkout, not application-owned compiler code. Our local
fork directory is named `hexagon-mlir-qwen3` for historical reasons; its package
names retain upstream compatibility. All compiler changes stay in that fork.

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
git clone -b main /path/to/application-source.bundle qwen3-hexagon-runtime
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
- `tools/qwen3/`: application build/deployment/regression orchestration (Python).
- `test/`, `qwen3/validation/`: tests and included reference hashes.
- `dependencies/`: compiler lock and application-side Transformers patch.
- `docs/qwen3/`: reproduction instructions and engineering reports.

Compiler C/C++/MLIR sources, DSP kernels and compiler patches are maintained in
the compiler repository. The application references those sources through the
pinned dependency. Logical StaticCache reset/update coordination belongs here;
compiler scatter lowering and bufferization belong there.

## Validation and current limits

```bash
python -m qwen3 test
```

The host command collects all retained application tests, including the numerical
policy tests. Tests requiring an explicit compiler/SDK skip when those tools are
not configured; a skipped test is not numerical qualification. Repository
separation has its historical record in `qwen3/validation/repository-split.json`.

The unified source build passed 699 configured host tests and nine IR checks in
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

The public inference entry imports no Python code from `tools/`. Historical
build/review tools remain for reading prior evidence and targeted diagnostics.
The maintained source-build route is `python -m qwen3.build`; historical recipe
consumers are outside that route and may require their original source revision
and artifacts.

The [source-build workflow](docs/qwen3/SOURCE_BUILD.md) builds the compiler profiles,
numerical tools, DSP helpers and both model domains in one output tree. LLVM and
the Qualcomm SDK remain explicit prebuilt dependencies. A build and a DSP
numerical qualification have separate receipts; see
[build status](docs/qwen3/BUILD_STATUS.md). Existing execution assets remain intact.

The previous combined tree is preserved at compiler-history commit
`1db3dae9ff09ba17c307c800777c443136050537`. Source origin records and original
copyright/license notices are retained. Do not publish this application as an
official Qualcomm release.
