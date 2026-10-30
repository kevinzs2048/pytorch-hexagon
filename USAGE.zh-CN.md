# Qwen3 Hexagon Runtime 使用指南

想先运行模型，直接看第 1、2 节。已有编译产物可以直接推理，不需要先构建 LLVM。

当前支持 QCS9075、Qwen3-0.6B 完整 28 层、batch=1。模型前向在 DSP 执行，ARM 运行分词和 Transformers 生成循环。线性层使用 W8A8，其他中间计算及 KV 使用 FP32；没有 CPU 模型前向回退。

本文的已验收代码基线是应用 `426f7f5`、编译器 `23cb318`。本机路径记录于 2026-10-30；迁移到别的机器时需要调整路径。

## 1. 先分清在哪台机器操作

| 用途 | 当前路径或地址 |
| --- | --- |
| 开发主机上的应用仓库 | `/home/kevin/hexagon/qwen3-hexagon-runtime` |
| 开发主机上的编译器仓库 | `/home/kevin/hexagon/hexagon-mlir-review` |
| 开发板 SSH | `radxa@192.168.31.135` |
| 开发板上已验证的应用源码副本 | `/mnt/nvme/qwen3-work/llvm-toolchain-20261030/candidate-source` |
| 开发板上的执行资源包 | `/mnt/nvme/qwen3-work/qwen3-hexagon-runtime/artifacts/qwen3-qcs9075` |

主机负责开发和编译；下面的 `generate`、`regress` 在开发板上运行。板端源码副本用于执行，修改代码和提交 Git 请回到主机上的仓库。

执行资源包包含模型权重、DSP 动态库、ARM/DSP 运行库、分词器和匹配的 ARM Python 包。**只克隆 Git 仓库不能直接推理，还需要这个资源包以及匹配的板端系统环境。**

## 2. 在现有开发板上生成文本

从开发主机登录：

```bash
ssh radxa@192.168.31.135
```

以下命令在登录后的同一个板端终端执行：

```bash
cd /mnt/nvme/qwen3-work/llvm-toolchain-20261030/candidate-source
unset PYTHONPATH

qwen_bundle=/mnt/nvme/qwen3-work/qwen3-hexagon-runtime/artifacts/qwen3-qcs9075
qwen_cache=/mnt/nvme/qwen3-work/user-cache/qwen3
qwen_run_root="/mnt/nvme/qwen3-work/user-runs/$(date +%Y%m%d-%H%M%S)"

python3 -m qwen3 doctor --bundle "$qwen_bundle" --execution

python3 -m qwen3 generate \
  --bundle "$qwen_bundle" \
  --cache-dir "$qwen_cache" \
  --prompt '你好，请用一句话介绍你自己。' \
  --chat-template \
  --max-new-tokens 32 \
  --output "$qwen_run_root/chinese"
```

`doctor` 校验资源包、编译器源码和板端 Python 环境。成功时输出中有 `bundle_valid: true` 和 `execution_environment_checked: true`。当前板端使用 ARM64 Python **3.12.3**，运行器自行加载资源包中的 Python 依赖，无需临时安装最新版 PyTorch 或 Transformers。

生成结束后，终端打印生成结果和结果目录。主要文件是：

| 文件 | 内容 |
| --- | --- |
| `result.json` | 每个请求的生成文本、token、停止条件和每次前向记录 |
| `execution.json` | 是否完成、DSP 调用次数、AOT 缓存命中情况及资源身份 |

只查看生成文本：

```bash
python3 - "$qwen_run_root/chinese/result.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    result = json.load(stream)
for request in result['requests']:
    print(request.get('generated_text', request['generated_ids']))
PY
```

换提示词只需修改 `--prompt`。多个请求可用以下形式，它们按顺序执行，每个请求重置 KV 状态：

```bash
python3 -m qwen3 generate \
  --bundle "$qwen_bundle" \
  --cache-dir "$qwen_cache" \
  --prompts-json '["用一句话介绍北京。", "写一个 Python 加法函数。"]' \
  --chat-template --max-new-tokens 64 \
  --output "$qwen_run_root/multiple"
```

使用时注意：

- `--output` 必须是尚不存在的目录。重跑时换目录名或重新设置 `qwen_run_root`。
- `--max-new-tokens` 是生成上限，遇到 EOS 可以提前结束。KV 容量限制为 `提示词 token 数 + max_new_tokens - 1 <= 1024`；应用聊天模板后的 token 也计入。
- `--chunk-size` 默认为 32，支持 1–32。长提示词会分块预填充，随后逐 token 解码；q1 和 q2–32 两个执行域复用已有 DSP 产物，不按提示词重新编译模型。
- 首次运行仍需要校验、加载和 Python/AOT 图捕获。后续运行保留同一 `--cache-dir` 可以复用匹配的 AOT 缓存；这不代表没有其他启动开销。
- 默认只保存完整输入输出的哈希。调试确需原始张量时才加 `--packet-recording full`，它会明显增加磁盘占用。

## 3. 验证是否仍然正确

在同一个板端终端运行内置回归：

```bash
python3 -m qwen3 regress \
  --bundle "$qwen_bundle" \
  --cache-dir "$qwen_cache" \
  --output "$qwen_run_root/regression"
```

该命令使用仓库里的固定输入完成 **16 次完整模型前向**，对照已保存的参考结果，包括完整 logits、全部 56 个 KV 张量的哈希、位置、生成 token 和停止条件。成功时生成 `comparison.json`。它不是现场重新运行 ARM PyTorch 参考，也不是全部长序列回归。

如果要复查本轮从源码重建的两个 DSP 库，板端已保留对应路径映射：

```bash
python3 -m qwen3.verify_build \
  --bundle "$qwen_bundle" \
  --libraries /mnt/nvme/qwen3-work/llvm-toolchain-20261030/candidate-libraries.json \
  --reference qwen3/validation/reference16.json \
  --packet-recording hashes \
  --output "$qwen_run_root/rebuilt/cold"
```

这会显式选择重建库，不覆盖原资源包。中文/代码长生成、EOS、1024 容量边界、请求重置等已有回归的运行方法见 [源码构建指南](docs/qwen3/SOURCE_BUILD.md#check-the-new-libraries-on-dsp)。

## 4. 给其他人拿到同一份源码

应用和编译器是两个仓库。应用通过 `third_party/hexagon-mlir` 子模块固定编译器版本；不要把另外一个目录的编译器源码随意覆盖进去。

当前主机上已验收代码的离线源码包在：

```text
/home/kevin/hexagon/qwen3-hexagon-runtime/.local/llvm-toolchain-20261030/delivery/application-source.bundle
/home/kevin/hexagon/qwen3-hexagon-runtime/.local/llvm-toolchain-20261030/delivery/compiler-source.bundle
```

将两个文件交给接收方，在目标机器上按实际文件位置执行以下命令。示例中的 `/absolute/path/to/delivery` 需要替换，目标 `qwen3-hexagon-runtime` 目录应尚不存在：

```bash
qwen_delivery=/absolute/path/to/delivery
git clone -b review/qwen3-llvm-toolchain \
  "$qwen_delivery/application-source.bundle" qwen3-hexagon-runtime
cd qwen3-hexagon-runtime
git config submodule.hexagon-mlir.url "$qwen_delivery/compiler-source.bundle"
git -c protocol.file.allow=always submodule update --init third_party/hexagon-mlir
git log -1 --oneline
git -C third_party/hexagon-mlir log -1 --oneline
```

这两个离线包对应已验收代码 `426f7f5` / `23cb318`，不自动包含后续文档提交。源码推送到远端后，也可从应用远端克隆并初始化子模块；需先确保固定的编译器提交已推送到子模块远端。

在另一块板上执行，还需要单独交付完整执行资源包，以及匹配的 QCS9075 Linux/FastRPC/DSP 环境。包内路径可以迁移，但不意味着任意系统版本或其他 DSP 都已经支持。将第 2 节的源码目录和 `qwen_bundle` 改为目标路径，先运行 `doctor --execution`，再运行 `regress`。详细资源导出方式见 [复现说明](docs/qwen3/REPRODUCE.md)。

## 5. 在现有开发主机上重新编译

只运行推理可以跳过本节。以下命令在 **开发主机** 上执行，复用当前已验证的工具链和编译目录：

```bash
cd /home/kevin/hexagon/qwen3-hexagon-runtime
qwen_python=/home/kevin/miniconda3/envs/main/bin/python
qwen_config="$PWD/.local/unified-build/toolchain.json"
qwen_compiler=/home/kevin/hexagon/hexagon-mlir-review
qwen_build="$PWD/.local/submission-review-20261028/process-codegen/build"

"$qwen_python" third_party/hexagon-mlir/dependencies/llvm/toolchain.py verify \
  --config "$qwen_config"

"$qwen_python" -m qwen3.build \
  --config "$qwen_config" --compiler-source "$qwen_compiler" \
  --models "$PWD/.local/unified-build/inputs/models.json" \
  --trusted-local-producers --output "$qwen_build" --jobs 8

"$qwen_python" -m qwen3.tools verify \
  --config "$qwen_config" --compiler-source "$qwen_compiler" \
  --build "$qwen_build" \
  --output "$PWD/.local/host-verification-$(date +%Y%m%d-%H%M%S)"
```

`qwen_build` 虽然沿用了历史目录名，但现在保存的是当前工具链构建的后端。Ninja 会复用未变化的编译器和 DSP 辅助对象。省略 `--models` 和 `--trusted-local-producers` 可只构建后端。

模型输出要求 `models/q1`、`models/q2` 尚不存在。本轮验收后已清理这两个中间目录，交付模型另存于 `.local/llvm-toolchain-20261030/delivery/models/`。再次构建后若要重跑，应先保存需要的模型和记录，再处理旧输出，不能直接覆盖或误删交付目录。

生成的 `models/q1/model.so`、`models/q2/model.so` 需要上板验收。复用资源包的常量库前，必须确认三个新常量库的哈希与包内对应文件一致；若不同，不能只替换两个模型库。部署映射和完整回归步骤见 [源码构建指南](docs/qwen3/SOURCE_BUILD.md)。

## 6. 换一台主机从头构建

现有 `.local/` 配置和产物不在 Git 中，主机绝对路径不能照搬。需要准备：

1. 应用及固定版本的编译器子模块、固定版本的 Triton 源码。
2. Hexagon SDK 6.5.0.0、Hexagon Tools 19.0.02、HexKL、兼容的主机 Clang，以及 CMake/Ninja 和主机系统库。
3. [源码构建指南](docs/qwen3/SOURCE_BUILD.md#inputs) 列出的固定 host Python/PyTorch/torch-mlir 等环境和补丁。板端资源包里的 ARM Python 库不能用于 x86 主机构建。
4. 模型资源，以及 ARM 实际生成流程导出的 q1、q2–32 两份可信图捕获。捕获及 `model-inputs.json` 配置方法也在该指南中。

先复制 `qwen3/config/build.example.json` 填写本机路径，按 [LLVM 重建说明](third_party/hexagon-mlir/dependencies/llvm/README.md) 准备独立源码并构建 LLVM/MLIR/LLD。配置中的 `llvm_receipt` 指向构建生成的校验记录，`lld` 指向该构建的 `bin/ld.lld`。然后运行统一入口 `python -m qwen3.build`，最后做主机及板端验证。

`seal-existing` 仅记录一套已有工具链，不会从源码重建，也不能代替新环境的构建验收。

## 7. 常见问题与结果保存位置

| 现象 | 处理办法 |
| --- | --- |
| `Output must be a fresh directory` | 换一个输出目录；不要复用已有结果目录。 |
| `Compiler dependency is missing` / `Pinned compiler source differs` | 初始化固定版本子模块；检查是否手工修改或混用了编译器版本。 |
| Python、框架文件或资源哈希校验失败 | 检查是否选对资源包和板端 Python，恢复匹配依赖。不要删掉校验来绕过问题。 |
| 提示词超出容量 | 缩短包含聊天模板的提示词，或降低 `--max-new-tokens`。 |
| 主机不能运行 `generate` | 连接 ARM 开发板执行；x86 主机用于构建和主机测试。 |
| LLVM 校验记录缺失或不匹配 | 按配置检查源码、库、链接器和校验记录；有意更换工具链后重新构建验证。 |

主机上的当前验证总记录：`.local/llvm-toolchain-20261030/review.json`。其中记录了源码提交、LLVM 重建、主机/离线检查、模型产物一致性和板端结果。详细验收范围见 [构建状态](docs/qwen3/BUILD_STATUS.md)。

运行结果放在 `results/`、`.local/` 或仓库外；权重、SDK、编译产物和一次性日志不提交到 Git。应用逻辑在本仓库维护，编译器变换与 DSP 内核在编译器仓库维护。
