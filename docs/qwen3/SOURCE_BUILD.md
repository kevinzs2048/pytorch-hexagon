# Build the Qwen3 model from source

`python -m qwen3.build` is the build entry. It builds the Hexagon-MLIR plugin,
both Qwen compiler configurations, the four numerical policy tools and the DSP
helpers. With `--models`, it also lowers the captured PyTorch graphs through
torch-mlir, compiles the model, generates its callers and links its libraries.
`python -m qwen3.tools build` invokes the same implementation.

The application owns build orchestration and the model integration. Compiler
passes and DSP kernels come from the pinned `third_party/hexagon-mlir` repository.
There is no copy of those C/C++ sources in the application.

## Inputs

- The application checkout and its pinned compiler submodule.
- The pinned Triton Git revision recorded in the compiler's
  `dependencies/sources.lock.json`. The builder exports that Git tree; local
  changes in the supplied Triton checkout are not used.
- A configured LLVM/MLIR installation matching that source lock and its local
  patch, the host compiler, Hexagon SDK 6.5.0.0, Hexagon Tools 19.0.02, HexKL and
  the qualified `ld.lld`. These are explicit prebuilt toolchain dependencies;
  this command does **not** rebuild LLVM or Qualcomm's SDK.
- The pinned host Python environment: PyTorch `2.12.0+cpu`, torch-mlir
  `20260908.870`, torchao `0.17.0+git02105d46`, Transformers `5.0.0rc1`, and the
  existing importer/framework patches. Graph transport checks the exact PyTorch
  source revision, not just its package version.
- Model weights/configuration and the catalog from the qualified asset bundle.
  The source build uses the catalog to check graph, weight and ABI identities;
  it does not copy the bundle's model objects or libraries.
- Two trusted captures from the ARM native generation path: q1 and q2–32.
  Captures contain lifted AOT graphs and checked bindings, not concrete model
  weights. They use a version-specific internal PyTorch pickle format and must
  not be loaded from untrusted sources.

Copy `qwen3/config/build.example.json` into a local configuration and replace all
paths. Keep the linker basename `ld.lld`; the Hexagon compiler driver uses it to
select the linker argument dialect. The old, broad profiling configuration in
`tools/qwen3/config.example.json` is not a source-build configuration.

## Capture the build inputs on ARM

Use the normal qualified bundle to run the included controlled regression and
export the actual compiler callbacks:

```bash
mkdir /absolute/path/to/captures
python3 -m qwen3.verify_build \
  --bundle /absolute/path/to/qwen3-qcs9075 \
  --reference qwen3/validation/reference16.json \
  --capture-output /absolute/path/to/captures \
  --output /absolute/path/to/capture-validation
```

This executes the original DSP forward and checks the full outputs. It does not
substitute a CPU model forward. A fresh AOT cache ensures that both captures are
exported. The resulting `result.json` records `exported_captures` and each
capture's graph contract. Copy the capture directories to the compiler host.
Keep `graph.pkl`, `result.json` and `ready.json` together.

Fill `qwen3/config/model-inputs.example.json` with the host paths to the model
assets and the two producer `result.json` files, including their SHA256 values.
Assign profiles from the recorded query bounds. A q1 graph has `[1, 1]`; the
prefill graph has `[2, 32]`. The builder checks the actual derived contract.

## Build and verify

```bash
python -m qwen3.build \
  --config /absolute/path/to/toolchain.json \
  --models /absolute/path/to/model-inputs.json \
  --trusted-local-producers \
  --output /absolute/path/to/build --jobs 8

python -m qwen3.tools verify \
  --config /absolute/path/to/toolchain.json \
  --build /absolute/path/to/build \
  --output /absolute/path/to/host-verification
```

Omit `--models` to build only the compiler, policy tools and DSP helpers. Ninja
reuses unchanged compiler outputs. Model outputs require fresh directories;
existing model files are not overwritten. The frontend cache validates its
source and artifact hashes before reuse.

For development against a separate compiler checkout, pass
`--compiler-source /absolute/path/to/hexagon-mlir` to the build.
The receipt records that checkout's source contents and rejects changed build
products. Without this flag, the checked submodule is used.

The retained application verification suite also reads C/C++ helpers and
fixtures through its pinned compiler submodule. Verification therefore rejects
a different-content `--compiler-source` (byte-identical copies are allowed);
it must not combine a candidate optimizer
with helper tests from the old checkout. For a candidate compiler, use its
own tests first and a separate application checkout with the compiler
dependency and source lock deliberately updated. If that checkout is missing
historical diagnostic helpers, resolve their scope explicitly before claiming
the complete retained suite passed.

The output has one layout:

```text
build/
  build.json                 compiler inputs and hashes of 22 products
  compiler-sources.json       exact compiler source contents
  compiler/                  base compiler plus q1/q2 configurations
  policy/                    numerical policy tools
  runtime/                   source-built DSP helper objects
  frontend/                  actual torch-mlir cache and lowering evidence
  models/q1/                 prepared IR, objects, constants, caller, model.so
  models/q2/                 same layout for dynamic q2–32
```

Each model directory contains a `result.json` with commands, stage results and
artifact hashes. A successful build reports `source_model_built_NOT_DSP_qualified`;
compilation alone is not numerical acceptance. Compiler options are explicitly
versioned in `qwen3/config/compiler-options.json`.

## Check the new libraries on DSP

Copy each `model.so` to a separate board directory. Create a JSON map containing
`q1` and `q2`, each with its absolute board `path` and `sha256`. Before reusing the
bundle's constant files, check that all three freshly linked constant libraries
have exactly the same SHA256 values as `bundle.json`. A changed constant payload
requires its own deployment; changing only the model DSO is not sufficient.

```bash
python3 -m qwen3.verify_build \
  --bundle /absolute/path/to/qwen3-qcs9075 \
  --libraries /absolute/path/to/candidate-libraries.json \
  --reference qwen3/validation/reference16.json \
  --output /absolute/path/to/candidate-validation
```

This uses the original native generation entry with explicitly selected model
libraries. It compares complete input/output packet hashes, all logits and 56 KV
tensors, cache positions, generated tokens and termination. The candidate gets
its own AOT namespace. The qualified bundle is not overwritten or promoted.

Repeat the command in a second process with a different output directory under
the same parent to check AOT cache reuse. With an initially empty candidate
cache, `execution.json` should report two misses on the first run and two hits
on the second. The candidate namespace includes both the runtime implementation
and the selected model-library identities.

The existing state regression is also included in the repository:

```bash
set -e
for case_name in tails eos old-long32 prefill1024 decode993 code64; do
  python3 -m qwen3.verify_build \
    --bundle /absolute/path/to/qwen3-qcs9075 \
    --libraries /absolute/path/to/candidate-libraries.json \
    --reference "qwen3/validation/states/${case_name}.json" \
    --packet-recording hashes \
    --output "/absolute/path/to/state-validation/${case_name}"
done
```

These six cases preserve the original 304-forward checks for chunk tails, EOS,
long generation, capacity boundaries and request reset. Hash mode still checks
every input/output byte, including all KV tensors; it avoids retaining large raw
KV packets on disk. Reference provenance and the older q1 library identity are
explained in [the state-reference notes](../../qwen3/validation/states/README.md).

Compiler/algorithm tests and source review remain necessary; a finite prompt
list does not prove every possible floating-point result.

## What was replaced

The build no longer reconstructs a compiler by copying intermediate objects from
old worktrees. Compiler configurations and runtime objects are explicit CMake
targets. Callers are generated directly from the captured ABI; they no longer
require a saved reference wrapper or its standalone file-I/O test program. A
minimal `main` symbol remains because the SDK loader requires it.

The old `bind_recipe` implementation, its three exclusive templates and the
separate RoPE build launcher were removed. Their sources remain in Git history
at application revision `9a5ce27`. Older recipe consumers and numerical diagnostic
tools remain for reading historical evidence; they are not dependencies of this
source-build path.
