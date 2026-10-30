# PyTorch Hexagon

PyTorch/Transformers integration for Qualcomm Hexagon DSP, using a separately
maintained Hexagon-MLIR compiler. **Qwen3-0.6B is the first supported model adapter.**
The repository layout accommodates other adapters; it does not claim that other
models already run correctly.

中文入口：[使用指南](USAGE.zh-CN.md)。Qwen3 的适配范围和入口见
[models/qwen3/README.md](models/qwen3/README.md)。

## Repository layout

| Directory | Responsibility |
| --- | --- |
| `hexagon_runtime/` | Shared compiler dependency resolution, RPC transport, packet packing, artifact caching and timing |
| `models/qwen3/` | Qwen3 model capture, generation/KV integration, numerical contracts, configuration and validation references |
| `models/qwen3/native/` | Qwen-specific C++ compiler policies, q1/q2 profiles and DSP helpers |
| `tools/qwen3/`, `tools/qwen3-review/` | Qwen build verification, deployment and diagnostic commands |
| `dependencies/` | Compiler/toolchain locks, LLVM reproduction recipe and framework patches |
| `third_party/hexagon-mlir/` | Pinned compiler submodule; generic compiler changes belong in that separate repository |
| `test/python/hexagon_runtime/` | Shared infrastructure tests |
| `test/python/qwen3/`, `test/python/torch-mlir/` | Model integration and numerical/compiler integration tests |
| `docs/qwen3/` | Qwen engineering and reproduction documentation |

Source language does not determine ownership. Qwen numerical algorithms remain
with the Qwen adapter even when implemented in C++ or MLIR. Shared transport does
not select tokens, manage model-specific KV rules or silently fall back to CPU.
The Qwen-specific AOT serialization and scalar guards also remain with the adapter.

## Run the supported model

Initialize the compiler submodule after cloning this application:

```bash
git submodule update --init third_party/hexagon-mlir
python -m hexagon_runtime --help
python -m hexagon_runtime qwen3 --help
```

On the configured QCS9075 board, with the matching execution asset bundle:

```bash
python3 -m hexagon_runtime qwen3 doctor --bundle /path/to/qwen3-qcs9075 --execution
python3 -m hexagon_runtime qwen3 generate --bundle /path/to/qwen3-qcs9075 \
  --prompt '你好，请用一句话介绍你自己。' --chat-template \
  --max-new-tokens 32 --output results/first
python3 -m hexagon_runtime qwen3 regress --bundle /path/to/qwen3-qcs9075 \
  --output results/regression
```

The asset bundle contains weights, compiled libraries, ARM/DSP runtime and pinned
ARM Python packages. Git source alone is insufficient for board execution. The
existing `python -m qwen3 ...` command forwards to the same Qwen adapter; it does
not contain a second implementation. Build/maintenance entries are now:

```bash
python -m models.qwen3.build --help
python -m models.qwen3.tools verify --help
```

See [source build](docs/qwen3/SOURCE_BUILD.md) and
[validation scope](docs/qwen3/BUILD_STATUS.md). The supported model still uses
batch 1, W8A8 linear layers, FP32 intermediates/KV, KV capacity 1024 and reusable
q1 / q2–32 execution domains. Model forward executes on DSP; ARM handles
Transformers generation, tokenization and stopping.

## Adding another adapter

Add its Python entry and model-specific code under `models/<name>/`, register its
name in `hexagon_runtime/__main__.py`, and place its tools/tests/docs in matching
model directories. Reuse shared primitives where their contracts apply. Keep
weights, captures, numerical policies, shape limits and acceptance references
separate. Validate the new model before advertising it as supported.

This repository was previously named `qwen3-hexagon-runtime`. Existing bundle
schemas, numerical policies and historical evidence retain their original names.
The compiler remains an independent fork of Qualcomm Hexagon-MLIR; this project
is not an official Qualcomm release. Copyright and license notices are retained.
