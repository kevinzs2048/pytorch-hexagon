# Build and Run Qwen3 on QCS9075

This guide walks through building the compiler and Qwen3 model libraries on an
x86 host, deploying the new libraries to a QCS9075 board, validating their outputs,
and generating text. **Run the commands in order from a single Bash session on
the development host.** Commands wrapped in `ssh` execute on the board.

The supported configuration is the complete 28-layer Qwen3-0.6B model, batch size
1, W8A8 linear layers, FP32 intermediate values and KV cache, and a KV capacity of
1024 tokens. ARM runs the Transformers generation loop; the model forward runs
on DSP. Two reusable execution domains cover q1 decoding and q2–32 chunked
prefill. Prompt text is a runtime input, not a compilation configuration.

## 0. Prerequisites

This workflow starts with an existing, validated execution asset bundle. It uses
that bundle to capture graphs from actual ARM generation, then rebuilds the model
from source. **It is not a clean-system bootstrap using only this Git repository
and Hugging Face weights.**

| Input | Requirements and purpose |
| --- | --- |
| Application source and pinned Hexagon-MLIR submodule | Use the same source revision on the host and board |
| Host Python environment | Configured x86 Python 3.11+ with PyTorch `2.12.0+cpu`, torch-mlir `20260908.870`, torchao `0.17.0+git02105d46`, Transformers `5.0.0rc1`, and the matching patches. Substituting current package releases is not supported by this baseline |
| Host build dependencies | CMake 3.24+, Ninja, a compatible host Clang, the pinned Triton source, and LLVM/MLIR/LLD with this project's patches |
| Qualcomm toolchain | Hexagon SDK 6.5.0.0, Hexagon Tools 19.0.02, and HexKL |
| QCS9075 board | Configured Linux/FastRPC/DSP environment, ARM64 Python 3.12.3, `flock`, `pgrep`, and writable shared memory |
| `qwen3-qcs9075` execution asset bundle | Weights, model configuration, tokenizer, validated model libraries, constant libraries, ARM/DSP runtime, and pinned ARM Python packages; one copy on the host and one on the board |
| Connectivity and transfer tools | SSH access from host to board; `tar` and `rsync` on both machines |

The bundle supplies the board's Python dependencies. The host compiler environment
must be prepared separately; its x86 packages cannot be replaced by the ARM
packages in the bundle. For a new host, obtain the matching environment snapshot
or installation packages and patches from the maintainer. This repository does
not yet provide an automated installer for all framework dependencies on a clean
system.

The execution bundle is supplied separately from Git. See
[execution assets](docs/qwen3/REPRODUCE.md) for its contents and
[build inputs](docs/qwen3/SOURCE_BUILD.md#inputs) for toolchain details.

## 1. Obtain the source and choose working directories

For a new checkout:

```bash
git clone --branch review/qwen3-integration --recurse-submodules \
  git@github.com:kevinzs2048/pytorch-hexagon.git
```

An existing checkout can be used directly. Offline source bundles are another
option, described at the end of this guide.

**Replace the example checkout, Python, board and asset paths below before
running the commands.** Use absolute paths without spaces or single quotes for
this walkthrough's SSH commands. `PH_CONFIG` can point to an existing verified
toolchain configuration; step 2 explains how to create one for a new host.

```bash
set -euo pipefail
cd /absolute/path/to/pytorch-hexagon
git submodule update --init third_party/hexagon-mlir
git log -1 --oneline
git submodule status

export PH_REPO="$PWD"
export PH_PYTHON=/absolute/path/to/host-python
export PH_BOARD=user@board-address
export PH_HOST_BUNDLE="$PH_REPO/artifacts/qwen3-qcs9075"
export PH_BOARD_BUNDLE=/absolute/path/on/board/qwen3-qcs9075
export PH_CONFIG="$PH_REPO/.local/unified-build/toolchain.json"

export PH_RUN_ID="$(date +%Y%m%d-%H%M%S)"
export PH_RUN="$PH_REPO/.local/reproduce-$PH_RUN_ID"
export PH_BUILD="$PH_RUN/build"
export PH_REMOTE_RUN="/absolute/path/on/board/work/reproduce-$PH_RUN_ID"
export PH_REMOTE_APP="$PH_REMOTE_RUN/application"
mkdir -p "$PH_REPO/.local"
mkdir "$PH_RUN"
```

The first group of variables identifies the environment and assets. The second
group creates a fresh directory for this run. Keep the source, captures and model
outputs unchanged while the build is running.

**Expected result:** the application revision is visible, the compiler submodule
is initialized, and `PH_RUN` is a new empty directory. If you close the terminal,
restore the same variables before resuming. Do not generate a new `PH_RUN_ID` and
then mix outputs from the preceding run.

## 2. Verify host assets and the toolchain

An existing development environment can use its verified `PH_CONFIG`. On a new
host, copy and edit the template first. Skip this copy when reusing an existing
configuration:

```bash
cp models/qwen3/config/build.example.json "$PH_RUN/toolchain.json"
export PH_CONFIG="$PH_RUN/toolchain.json"
# Edit PH_CONFIG and enter the absolute paths for your host.
```

| Configuration field | Value to supply |
| --- | --- |
| `python`, `host_clang`, `host_clangxx`, `cmake` | Host executables; `python` must use the same environment as `PH_PYTHON` |
| `triton_source` | Triton Git checkout containing the revision pinned in `dependencies/toolchain.lock.json` |
| `llvm_source`, `llvm_build`, `llvm_receipt` | Patched LLVM source, its build directory, and the build identity record generated for those products |
| `hexagon_tools`, `hexagon_sdk`, `hexkl` | Installation directories for the three Qualcomm dependencies |
| `lld` | Matching Hexagon linker; its basename must be `ld.lld` |
| `library_path` | Directories containing shared libraries required by host Clang/LLVM |

If LLVM has not been built, follow the
[LLVM source-build instructions](dependencies/llvm/README.md#source-build) to run
`prepare` and `build`, then `verify`. The `models.qwen3.build` entry does not build
LLVM or install the SDK. A failed identity check must be resolved against the
actual inputs and products; `seal-existing` is not a substitute for that check,
and a record from another toolchain cannot be reused.

Once the configuration is ready, run both checks:

```bash
"$PH_PYTHON" -m hexagon_runtime qwen3 doctor --bundle "$PH_HOST_BUNDLE"
"$PH_PYTHON" dependencies/llvm/toolchain.py verify --config "$PH_CONFIG"
```

**Expected result:** the first command reports `bundle_valid: true`; the second
exits successfully and reports the matching LLVM identity.
`clean_compiler_rebuild_verified: false` means that `doctor` did not perform a
source build. It is not an environment-check failure.

## 3. Deploy the same source revision to the board

These commands transfer the **committed HEAD**, excluding uncommitted edits, host
Python packages, model weights and previous build directories. The compiler
submodule is transferred separately. Use a clean application checkout and the
pinned submodule revision so that host and board sources agree.

```bash
ssh "$PH_BOARD" "mkdir -p '$PH_REMOTE_APP/third_party/hexagon-mlir'"
git archive HEAD | ssh "$PH_BOARD" "tar -xf - -C '$PH_REMOTE_APP'"
git -C third_party/hexagon-mlir archive HEAD | \
  ssh "$PH_BOARD" "tar -xf - -C '$PH_REMOTE_APP/third_party/hexagon-mlir'"

ssh "$PH_BOARD" "bash -s -- '$PH_REMOTE_APP' '$PH_BOARD_BUNDLE'" <<'SH'
set -euo pipefail
cd "$1"
unset PYTHONPATH
python3 -m hexagon_runtime qwen3 doctor --bundle "$2" --execution
SH
```

**Expected result:** the board reports `bundle_valid: true` and
`execution_environment_checked: true`. This checks the model assets and the
actual ARM framework files. Resolve any mismatch before exporting captures.

## 4. Export both graph captures on the board

This command runs the included 16-forward reference check using the bundle's
validated libraries. It also exports the graphs from the actual AOT compiler
callbacks for the subsequent source build. Hash recording checks complete tensor
contents without retaining large raw KV packets.

```bash
ssh "$PH_BOARD" "bash -s -- '$PH_REMOTE_APP' '$PH_REMOTE_RUN' '$PH_BOARD_BUNDLE'" <<'SH'
set -euo pipefail
cd "$1"
unset PYTHONPATH
mkdir "$2/captures"
python3 -m models.qwen3.verify_build \
  --bundle "$3" \
  --reference models/qwen3/validation/reference16.json \
  --capture-output "$2/captures" \
  --packet-recording hashes \
  --output "$2/capture-validation"
cat "$2/capture-validation/comparison.json"
SH
```

**Expected result:** `comparison.json` reports `status: identical` and
`DSP_calls: 16`, and two captures are created. Each capture contains `graph.pkl`,
`result.json` and `ready.json`.

Captures bind the producer's source hashes. After changing the adapter source,
export fresh captures using this step; do not edit old records to make them look
current. Existing compiled model libraries can still be used independently of
these model-build inputs.

## 5. Retrieve captures and create the model-input configuration

```bash
mkdir "$PH_RUN/captures" "$PH_RUN/capture-validation"
rsync -a "$PH_BOARD:$PH_REMOTE_RUN/captures/" "$PH_RUN/captures/"
rsync -a "$PH_BOARD:$PH_REMOTE_RUN/capture-validation/" "$PH_RUN/capture-validation/"

"$PH_PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
run = Path(os.environ['PH_RUN'])
bundle = Path(os.environ['PH_HOST_BUNDLE'])
result = json.loads((run/'capture-validation/result.json').read_text())
comparison = json.loads((run/'capture-validation/comparison.json').read_text())
assert comparison['status'] == 'identical'
assert len(result['captures']) == len(result['exported_captures']) == 2
profiles = {(1, 1): 'q1', (2, 32): 'q2'}
producers = {}
for capture, exported in zip(result['captures'], result['exported_captures'], strict=True):
    profile = profiles[tuple(capture['contract']['variables']['query'])]
    path = run/'captures'/Path(exported['path']).name/'result.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == exported['producer_sha256']
    assert json.loads(path.read_text())['graph_identity'] == capture['graph']
    assert profile not in producers
    producers[profile] = {'path': str(path), 'sha256': digest}
assert set(producers) == {'q1', 'q2'}
inputs = dict(context=str(bundle/'context.json'), hf=str(bundle/'hf'),
              catalog=str(bundle/'catalog.json'), producers=producers)
(run/'model-inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
print(run/'model-inputs.json')
PY
```

**Expected result:** `PH_RUN/model-inputs.json` records host paths and SHA256
values for q1 and q2. The script assigns profiles using the captured query bounds,
rather than assuming that `capture-00` and `capture-01` have a particular order.

## 6. Build the compiler and both model libraries from source

```bash
"$PH_PYTHON" -m models.qwen3.build \
  --config "$PH_CONFIG" \
  --models "$PH_RUN/model-inputs.json" \
  --trusted-local-producers \
  --output "$PH_BUILD" --jobs 8

"$PH_PYTHON" -m models.qwen3.tools verify \
  --config "$PH_CONFIG" --build "$PH_BUILD" \
  --output "$PH_RUN/host-verification"
```

The entry builds the Hexagon-MLIR backend, Qwen compiler policies and DSP helpers,
then lowers both graphs through torch-mlir, emits Hexagon objects and links the
model libraries. It does not copy model objects from previous build directories.

The principal outputs are:

```text
$PH_BUILD/
  build.json                 Backend build record and hashes of 22 products
  compiler/                  Base compiler and q1/q2 compiler configurations
  policy/                    Numerical policy tools
  runtime/                   DSP helper objects
  frontend/                  torch-mlir lowering cache and records
  models/q1/model.so          Decode model library
  models/q2/model.so          Chunked-prefill model library
  models/q1/constants/        Constant libraries linked during this build
  models/q2/constants/        Constant libraries linked during this build
  models/q1/result.json       q1 build record
  models/q2/result.json       q2 build record
```

**Expected result:** both model `result.json` files report
`source_model_built_NOT_DSP_qualified`; host verification reports
`policy_framework_and_IR_checks_passed`. The current baseline contains 622 host
checks and nine IR checks. Use the result record from your own run as evidence.

To build only the backend, omit `--models` and `--trusted-local-producers`. That
mode does not produce the two `model.so` files. Build logs are under
`PH_BUILD/*.log` and the two model directories. Inspect the relevant stage log if
a command fails. Backend builds can be incremental; model outputs require fresh
directories and cannot overwrite existing `models/q1` or `models/q2` outputs.

## 7. Verify constants and deploy the new model libraries

This workflow reuses the bundle's constant libraries. It therefore checks that
all newly generated constant libraries are byte-identical to those in the bundle.
If this check fails, stop before deployment. Different constants require a
separate matching asset deployment; replacing only the two model libraries is
insufficient, and this walkthrough does not automate that case.

```bash
"$PH_PYTHON" - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
run, build = Path(os.environ['PH_RUN']), Path(os.environ['PH_BUILD'])
bundle = Path(os.environ['PH_HOST_BUNDLE'])
manifest = json.loads((bundle/'bundle.json').read_text())
deploy = run/'deploy'
deploy.mkdir()
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
libraries = {}
for profile in ('q1', 'q2'):
    record = json.loads((build/'models'/profile/'result.json').read_text())
    assert record['status'] == 'source_model_built_NOT_DSP_qualified'
    expected = {f'lib_mlir_ciface_Qwen3DynamicBoundProbe-consts-{i}.so' for i in (1,2,3)}
    assert {Path(row['path']).name for row in record['constants']} == expected
    for row in record['constants']:
        path = Path(row['path'])
        assert sha(path) == row['sha256'] == manifest['files']['constants/'+path.name]['sha256']
    library = Path(record['library']['path'])
    digest = sha(library)
    assert digest == record['library']['sha256']
    shutil.copy2(library, deploy/(profile+'.so'))
    libraries[profile] = dict(path=os.environ['PH_REMOTE_RUN']+'/libraries/'+profile+'.so',
                              sha256=digest)
(deploy/'candidate-libraries.json').write_text(json.dumps(libraries, indent=2)+'\n')
print('Both model libraries and all constants verified:', deploy)
PY

ssh "$PH_BOARD" "mkdir '$PH_REMOTE_RUN/libraries'"
rsync -a "$PH_RUN/deploy/" "$PH_BOARD:$PH_REMOTE_RUN/libraries/"
```

**Expected result:** `PH_REMOTE_RUN/libraries/` on the board contains `q1.so`,
`q2.so` and `candidate-libraries.json`, which specifies their absolute board paths
and hashes. The validated execution bundle remains intact.

## 8. Validate the newly built libraries on DSP

```bash
ssh "$PH_BOARD" "bash -s -- '$PH_REMOTE_APP' '$PH_REMOTE_RUN' '$PH_BOARD_BUNDLE'" <<'SH'
set -euo pipefail
cd "$1"
unset PYTHONPATH
python3 -m models.qwen3.verify_build \
  --bundle "$3" \
  --libraries "$2/libraries/candidate-libraries.json" \
  --reference models/qwen3/validation/reference16.json \
  --packet-recording hashes \
  --output "$2/candidate-validation"
cat "$2/candidate-validation/comparison.json"
SH
```

**Expected result:** `status: identical` and `DSP_calls: 16`. The comparison checks
complete inputs, logits, all 56 KV tensors, positions, tokens and termination,
not just generated text. The `candidate_libraries` field in `execution.json`
should contain this build's q1/q2 hashes, confirming which libraries ran.

This is the basic functional acceptance check. Additional generation lengths,
capacity boundaries and state transitions are covered by the
[extended regression commands](docs/qwen3/SOURCE_BUILD.md#check-the-new-libraries-on-dsp).
Those longer regressions need not be repeated for every documentation or directory
change.

## 9. Generate text with the newly built libraries

The ordinary `python -m hexagon_runtime qwen3 generate` command loads the validated
bundle by default and has no `--libraries` option. To run this build's candidate
libraries, use the existing Python interface below with the same explicit library
mapping. Do not overwrite the validated bundle or change its resource locks.

```bash
ssh "$PH_BOARD" "bash -s -- '$PH_REMOTE_APP' '$PH_REMOTE_RUN' '$PH_BOARD_BUNDLE'" <<'SH'
set -euo pipefail
cd "$1"
unset PYTHONPATH
python3 - "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
from types import SimpleNamespace
from models.qwen3.cli import generate
from models.qwen3.verify_build import stage_candidates
run, bundle = map(Path, sys.argv[1:])
libraries = json.loads((run/'libraries/candidate-libraries.json').read_text())
args = SimpleNamespace(
    bundle=bundle, output=run/'candidate-example', cache_dir=run/'candidate-aot-cache',
    prompt='Hello. Please introduce yourself in one sentence.', prompts_json=None, requests_json=None,
    chat_template=True, max_new_tokens=32, chunk_size=32, packet_recording='hashes')
generate(args, prepare_candidate=lambda catalog: stage_candidates(catalog, libraries))
execution = json.loads((args.output/'execution.json').read_text())
assert execution['status'] == 'completed'
assert execution['candidate_libraries'] == {k:v['sha256'] for k,v in libraries.items()}
PY
SH
```

Change `prompt` to try another input, and choose a new `output` directory for each
run. The command prints the generated text. `candidate-example/result.json`
records tokens and termination; `execution.json` records the actual library
identities and DSP calls. Generation can stop early at EOS. The model's capacity
check is `prompt_tokens + max_new_tokens - 1 <= 1024`.

If you only need to run the already validated bundle, skip steps 4–8 and use the
ordinary `generate` command shown in the [README](README.md#run-the-supported-model).
That run does not validate the newly built libraries.

## 10. Retain the reproducibility records

Keep the source revision, toolchain configuration and identity record, both
captures, `model-inputs.json`, model and constant libraries,
`host-verification/result.json`, and the board's `comparison.json` and
`execution.json`. These belong in this run's `PH_RUN` / `PH_REMOTE_RUN` directories,
not in Git.

| Failure | Where to investigate |
| --- | --- |
| SSH or asset validation fails | Steps 0–3: board address, environment and matching asset bundle |
| Capture source hash differs | Step 4: export again with current source; preserve all three capture files |
| LLVM/compiler identity differs | Step 2: check configuration and actual products; do not edit hashes to bypass validation |
| Compilation fails | The failed stage's log from step 6; resolve it before deployment |
| Constant hashes differ | Stop at step 7; do not deploy new model libraries with unmatched old constants |
| Board outputs differ | Retain the step 8 results and investigate the candidate libraries; the run has not passed acceptance |
| An output directory already exists | Use a fresh run directory and preserve the earlier records |

## Offline source checkout

For offline use, the maintainer can supply `application-source.bundle` and
`compiler-source.bundle`. Confirm the included revision with `git log -1` after
cloning.

```bash
PH_DELIVERY=/absolute/path/to/delivery
git clone -b review/qwen3-integration "$PH_DELIVERY/application-source.bundle" pytorch-hexagon
cd pytorch-hexagon
git config submodule.hexagon-mlir.url "$PH_DELIVERY/compiler-source.bundle"
git -c protocol.file.allow=always submodule update --init third_party/hexagon-mlir
```

## Validation scope

The repository reorganization passed a backend source build, 622 host checks,
nine IR checks and a 16-forward board comparison using the existing model
libraries. This guide documents fresh capture export, full model rebuilding,
deployment and inference with candidate libraries. Updating or translating the
guide does not constitute another complete model rebuild or acceptance run.
See [build status](docs/qwen3/BUILD_STATUS.md) for completed validation and evidence.
