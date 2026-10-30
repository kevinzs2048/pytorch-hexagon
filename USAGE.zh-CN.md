# 从源码构建到板端推理：Qwen3 示例

按下列步骤，可以在开发主机构建编译器和 Qwen3 模型，将新模型库部署到 QCS9075，然后验证并生成文本。**全部命令从开发主机的同一个 Bash 终端依次执行**；标有 `ssh` 的代码会自动到板端运行，不需要手工来回切换目录。

当前范围：Qwen3-0.6B 完整 28 层、batch=1、W8A8 线性层、FP32 中间计算及 KV、容量 1024。ARM 负责 Transformers 生成循环，DSP 执行模型前向。q1 用于解码，q2–32 用于分块预填充，提示词不是编译配置。

## 0. 先备齐输入

本流程需要已有的已验收执行资源包，用它导出真实 ARM 生成过程中的计算图，再从源码重新编译模型。**目前不是仅凭 Git 仓库和 Hugging Face 权重就能从空白系统启动的安装流程。**

| 输入 | 用途与要求 |
| --- | --- |
| 应用源码及固定的 Hexagon-MLIR 子模块 | 主机和板端必须使用相同版本 |
| 主机 Python 环境 | 已配置的 x86 Python 3.11+；PyTorch `2.12.0+cpu`、torch-mlir `20260908.870`、torchao `0.17.0+git02105d46`、Transformers `5.0.0rc1` 及匹配补丁。不能直接安装最新版替代 |
| 主机构建依赖 | CMake 3.24+、Ninja、兼容的 Host Clang、固定 Triton 源码、带本项目补丁的 LLVM/MLIR/LLD |
| 高通工具链 | Hexagon SDK 6.5.0.0、Hexagon Tools 19.0.02、HexKL |
| QCS9075 开发板 | 已配置的 Linux/FastRPC/DSP 环境、ARM64 Python 3.12.3、`flock`、`pgrep`、可写共享内存 |
| `qwen3-qcs9075` 执行资源包 | 权重、配置、分词器、已验收模型库、常量库、ARM/DSP 运行库及固定 ARM Python 包；主机和板端各一份 |
| 连接与传输 | 主机能 SSH 登录开发板；两端有 `tar`、`rsync` |

板端 Python 依赖由资源包提供。主机的编译环境要另行准备，不能使用资源包里的 ARM Python 包代替。换机器时需取得维护者提供的匹配主机环境/安装包及补丁；本仓库尚未提供从空白系统自动安装全部框架依赖的脚本。资源包不在 Git 中，需从维护者取得；其组成见[资源包说明](docs/qwen3/REPRODUCE.md)。完整构建依赖见[源码构建说明](docs/qwen3/SOURCE_BUILD.md#inputs)。

## 1. 固定源码，并设置本轮目录

现有开发机直接进入下面的目录。新机器需要先取得应用仓库：远端创建并推送后，可执行 `git clone -b review/qwen3-integration --recurse-submodules git@github.com:kevinzs2048/pytorch-hexagon.git`；也可使用文末的离线源码包。

```bash
set -euo pipefail
cd /home/kevin/hexagon/pytorch-hexagon
git submodule update --init third_party/hexagon-mlir
git log -1 --oneline
git submodule status

export PH_REPO="$PWD"
export PH_PYTHON=/home/kevin/miniconda3/envs/main/bin/python
export PH_BOARD=radxa@192.168.31.135
export PH_HOST_BUNDLE="$PH_REPO/artifacts/qwen3-qcs9075"
export PH_BOARD_BUNDLE=/mnt/nvme/qwen3-work/qwen3-hexagon-runtime/artifacts/qwen3-qcs9075
export PH_CONFIG="$PH_REPO/.local/unified-build/toolchain.json"

export PH_RUN_ID="$(date +%Y%m%d-%H%M%S)"
export PH_RUN="$PH_REPO/.local/reproduce-$PH_RUN_ID"
export PH_BUILD="$PH_RUN/build"
export PH_REMOTE_RUN="/mnt/nvme/qwen3-work/reproduce-$PH_RUN_ID"
export PH_REMOTE_APP="$PH_REMOTE_RUN/application"
mkdir "$PH_RUN"
```

这些变量中，前半部分是机器和资源位置，需要按实际环境修改；后半部分为本轮创建新目录，避免覆盖以前的结果。源码路径、图捕获和模型输出在构建期间应保持不变。

成功标志：应用提交可查看，子模块已初始化；`PH_RUN` 是新建的空目录。结束终端后变量会丢失，继续执行时需恢复相同变量，不能重新生成 `PH_RUN_ID` 后混用前一轮产物。

## 2. 检查主机资源和工具链

现有开发环境可直接使用已配置好的 `PH_CONFIG`。新主机先完成以下配置准备；已有配置时跳过模板复制。

**新主机：**先复制模板并编辑：

```bash
cp models/qwen3/config/build.example.json "$PH_RUN/toolchain.json"
export PH_CONFIG="$PH_RUN/toolchain.json"
# 用编辑器填写 PH_CONFIG 中的本机绝对路径。
```

| 配置字段 | 应填写什么 |
| --- | --- |
| `python`、`host_clang`、`host_clangxx`、`cmake` | 主机可执行文件；`python` 与本节 `PH_PYTHON` 使用同一环境 |
| `triton_source` | 包含 `dependencies/toolchain.lock.json` 中固定提交的 Triton Git 仓库 |
| `llvm_source`、`llvm_build`、`llvm_receipt` | 匹配补丁的源码、构建目录和构建生成的身份记录 |
| `hexagon_tools`、`hexagon_sdk`、`hexkl` | 三个高通依赖的安装目录 |
| `lld` | 匹配的 Hexagon 链接器，文件名必须是 `ld.lld` |
| `library_path` | Host Clang/LLVM 需要的共享库目录列表 |

没有 LLVM 构建产物时，先按[LLVM 构建步骤](dependencies/llvm/README.md#source-build)执行 `prepare` 和 `build`，再执行 `verify`。`models.qwen3.build` 不会替你构建 LLVM 或安装 SDK。不要使用 `seal-existing` 掩盖校验失败，也不要沿用另一套工具链的记录。

**配置准备好后，两种环境都执行以下检查：**

```bash
"$PH_PYTHON" -m hexagon_runtime qwen3 doctor --bundle "$PH_HOST_BUNDLE"
"$PH_PYTHON" dependencies/llvm/toolchain.py verify --config "$PH_CONFIG"
```

成功标志：第一个命令输出 `bundle_valid: true`；第二个命令正常退出并输出匹配的 LLVM 身份记录。`clean_compiler_rebuild_verified: false` 仅表示 `doctor` 不负责源码构建，不代表环境检查失败。

## 3. 把同一份源码送到开发板

下面发送当前 **已提交的 HEAD**，不发送未提交改动、主机 Python 环境、模型权重或历史构建目录。编译器子模块单独发送。

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

成功标志：板端输出 `bundle_valid: true`、`execution_environment_checked: true`。这里校验了模型资源和实际 ARM 框架文件。若失败，先处理资源/环境不匹配，不要继续导出图。

## 4. 在板端导出当前版本的两份图捕获

该命令使用资源包里的已验收模型执行内置 16 次前向对照，同时导出实际 AOT 回调中的图，供后续源码编译使用。只保留张量哈希，避免保存大量 KV 原始数据。

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

成功标志：`comparison.json` 为 `status: identical`、`DSP_calls: 16`，并生成两份捕获。每份捕获包含 `graph.pkl`、`result.json`、`ready.json`。

旧版本捕获绑定了旧源码的哈希，不能修改记录冒充当前版本。已有模型库可继续使用；源码版本变化后，用本步骤重新捕获即可。

## 5. 把捕获取回主机，生成模型输入配置

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

成功标志：生成 `PH_RUN/model-inputs.json`，明确记录 q1、q2 的主机路径和 SHA256。按图的查询维度归类，不凭 `capture-00`、`capture-01` 的编号猜测。

## 6. 从源码构建编译器和两个模型库

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

这个入口依次构建 Hexagon-MLIR 后端、Qwen 编译策略和 DSP 辅助代码，再经 torch-mlir 降低两份图、生成 Hexagon 对象和链接模型库。它不会从旧构建目录复制模型对象。

主要输出：

```text
$PH_BUILD/
  build.json                 后端构建记录及 22 个产物的哈希
  compiler/                  基础编译器及 q1/q2 编译配置
  policy/                    数值处理工具
  runtime/                   DSP 辅助对象
  frontend/                  torch-mlir 转换缓存和记录
  models/q1/model.so          解码模型库
  models/q2/model.so          分块预填充模型库
  models/q1/constants/        本轮重新链接的常量库
  models/q2/constants/        本轮重新链接的常量库
  models/q1/result.json       q1 构建记录
  models/q2/result.json       q2 构建记录
```

成功标志：两个模型的 `result.json` 均为 `source_model_built_NOT_DSP_qualified`，主机检查为 `policy_framework_and_IR_checks_passed`。当前基线是 622 项主机检查及 9 项 IR 检查；以当次输出记录为准。

只想构建后端可省略 `--models` 和 `--trusted-local-producers`，但这不会生成两个 `model.so`。编译过程日志在 `PH_BUILD/*.log` 和两个模型目录内；出错先看相应日志。后端可增量构建，模型输出要求新目录，不能直接覆盖已有 `models/q1`、`models/q2`。

## 7. 校验常量，部署本轮模型库

本流程复用资源包的常量库，所以必须先确认本轮生成的常量与资源包逐字节一致。以下检查失败就停止，不要只替换两个模型库；不同常量需要单独整理匹配的资源包，本流程不自动完成该情况。

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
print('两份模型及全部常量校验通过：', deploy)
PY

ssh "$PH_BOARD" "mkdir '$PH_REMOTE_RUN/libraries'"
rsync -a "$PH_RUN/deploy/" "$PH_BOARD:$PH_REMOTE_RUN/libraries/"
```

输出：板端 `PH_REMOTE_RUN/libraries/` 内有 `q1.so`、`q2.so` 和明确指定两者路径、哈希的 `candidate-libraries.json`。已验收资源包保持原样。

## 8. 明确加载本轮模型，做板端对照

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

成功标志：`status: identical`、`DSP_calls: 16`。这里检查完整输入、logits、全部 56 个 KV 张量、位置、token 和停止条件，不只比较输出文字。`execution.json` 的 `candidate_libraries` 应包含本轮 q1/q2 的哈希；这才表明用的是新模型库。

这一步是基本功能验收；长期运行、更多生成长度及容量边界的回归另见[完整回归命令](docs/qwen3/SOURCE_BUILD.md#check-the-new-libraries-on-dsp)，不要求每次文档或目录调整都重复全部长回归。

## 9. 用本轮模型生成自己的文本

普通 `python -m hexagon_runtime qwen3 generate` 默认加载已验收资源包，没有 `--libraries` 参数。要使用刚构建的候选库，可通过下面的现有 Python 接口显式传入同一份库映射；不要覆盖资源包里的文件或修改资源锁。

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
    bundle=bundle, output=run/'candidate-chinese', cache_dir=run/'candidate-aot-cache',
    prompt='你好，请用一句话介绍你自己。', prompts_json=None, requests_json=None,
    chat_template=True, max_new_tokens=32, chunk_size=32, packet_recording='hashes')
generate(args, prepare_candidate=lambda catalog: stage_candidates(catalog, libraries))
execution = json.loads((args.output/'execution.json').read_text())
assert execution['status'] == 'completed'
assert execution['candidate_libraries'] == {k:v['sha256'] for k,v in libraries.items()}
PY
SH
```

修改 `prompt` 即可换输入；再次运行要换 `output` 目录。终端会打印文本，`candidate-chinese/result.json` 保存生成 token 和停止原因，`execution.json` 保存实际库身份与 DSP 调用次数。遇到 EOS 可提前停止；提示词和生成总长度受 1024 容量限制。

如果只是运行已验收模型，直接跳过第 4–8 步，用普通 `generate` 即可；它的结果不能算作本轮新模型的验收。

## 10. 保存哪些内容

保留源码版本、工具链配置和身份记录、两份捕获、`model-inputs.json`、模型及常量库、`host-verification/result.json`、板端 `comparison.json` 和 `execution.json`。这些保存在本轮 `PH_RUN` / `PH_REMOTE_RUN`，不提交到 Git。

| 遇到的问题 | 处理位置 |
| --- | --- |
| SSH 或资源校验失败 | 第 0–3 步；检查地址、板端环境与匹配资源包 |
| 捕获源码哈希不匹配 | 第 4 步；用当前源码重新导出，保留三件套 |
| LLVM/编译器身份不匹配 | 第 2 步；核对配置与实际产物，不能改哈希跳过校验 |
| 编译失败 | 第 6 步对应阶段的日志；先解决失败再部署 |
| 常量哈希不同 | 停在第 7 步，不复用旧常量部署新模型 |
| 板端输出不同 | 保留第 8 步结果，排查新库；不要宣称验收通过 |
| 输出目录已存在 | 使用新的运行目录，不覆盖旧记录 |

## 离线取得源码

没有应用远端时，维护者可交付 `application-source.bundle` 和 `compiler-source.bundle`。含本指南的源码包放在主机 `.local/docs-walkthrough-20261031/delivery/`；包对应哪个版本，以克隆后的 `git log -1` 为准。

```bash
PH_DELIVERY=/absolute/path/to/delivery
git clone -b review/qwen3-integration "$PH_DELIVERY/application-source.bundle" pytorch-hexagon
cd pytorch-hexagon
git config submodule.hexagon-mlir.url "$PH_DELIVERY/compiler-source.bundle"
git -c protocol.file.allow=always submodule update --init third_party/hexagon-mlir
```

**验证范围：**目录重构已通过后端源码构建、622 项主机检查、9 项 IR 检查，以及使用已有模型库的 16 次板端对照。本指南补齐了重新捕获、完整模型重建、部署和候选库推理的操作顺序；本次文档调整没有重新跑完整模型编译，不能把操作说明当作新增验收记录。已有验收和证据位置见[构建状态](docs/qwen3/BUILD_STATUS.md)。
