# PyTorch Hexagon 使用指南（Qwen3 示例）

想先运行模型，直接看第 1、2 节。已有编译产物可以直接推理，不需要先构建 LLVM。

当前支持 QCS9075、Qwen3-0.6B 完整 28 层、batch=1。模型前向在 DSP 执行，ARM 运行分词和 Transformers 生成循环。线性层使用 W8A8，其他中间计算及 KV 使用 FP32；没有 CPU 模型前向回退。

当前仓库名为 `pytorch-hexagon`，共享代码位于 `hexagon_runtime/`，Qwen3 适配位于 `models/qwen3/`。编译器仍固定为 `f13109d`。此前完整模型验收对应应用 `426f7f5`、编译器 `23cb318`；第 2、3 节使用本次目录重构后已通过 16 次前向对照的板端副本，复用已有模型库。本次目录迁移的验证范围见 [构建状态](docs/qwen3/BUILD_STATUS.md)，不与此前模型验收混计。本机路径记录于 2026-10-30。

## 1. 先分清在哪台机器操作

| 用途 | 当前路径或地址 |
| --- | --- |
| 开发主机上的应用仓库 | `/home/kevin/hexagon/pytorch-hexagon` |
| 开发主机上的编译器仓库 | `/home/kevin/hexagon/hexagon-mlir-review` |
| 开发板 SSH | `radxa@192.168.31.135` |
| 开发板上已验证的应用源码副本 | `/mnt/nvme/qwen3-work/pytorch-hexagon-layout-20261030T151430Z/application` |
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
cd /mnt/nvme/qwen3-work/pytorch-hexagon-layout-20261030T151430Z/application
unset PYTHONPATH

qwen_bundle=/mnt/nvme/qwen3-work/qwen3-hexagon-runtime/artifacts/qwen3-qcs9075
qwen_cache=/mnt/nvme/qwen3-work/user-cache/qwen3
qwen_run_root="/mnt/nvme/qwen3-work/user-runs/$(date +%Y%m%d-%H%M%S)"

python3 -m hexagon_runtime qwen3 doctor --bundle "$qwen_bundle" --execution

python3 -m hexagon_runtime qwen3 generate \
  --bundle "$qwen_bundle" \
  --cache-dir "$qwen_cache" \
  --prompt '你好，请用一句话介绍你自己。' \
  --chat-template \
  --max-new-tokens 32 \
  --output "$qwen_run_root/chinese"
```

`doctor` 校验资源包、编译器源码和板端 Python 环境。成功时输出中有 `bundle_valid: true` 和 `execution_environment_checked: true`。当前板端使用 ARM64 Python **3.12.3**，运行器自行加载资源包中的 Python 依赖，无需临时安装最新版 PyTorch 或 Transformers。

其中 `clean_compiler_rebuild_verified: false` 表示这个命令没有执行编译器源码重建，不是环境检查失败；源码构建验收看第 5 节的构建和验证记录。

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
python3 -m hexagon_runtime qwen3 generate \
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
python3 -m hexagon_runtime qwen3 regress \
  --bundle "$qwen_bundle" \
  --cache-dir "$qwen_cache" \
  --output "$qwen_run_root/regression"
```

该命令使用仓库里的固定输入完成 **16 次完整模型前向**，对照已保存的参考结果，包括完整 logits、全部 56 个 KV 张量的哈希、位置、生成 token 和停止条件。成功时生成 `comparison.json`。它不是现场重新运行 ARM PyTorch 参考，也不是全部长序列回归。

如果要复查此前已验收、从源码重建的两个 DSP 库，板端已保留对应路径映射：

```bash
python3 -m models.qwen3.verify_build \
  --bundle "$qwen_bundle" \
  --libraries /mnt/nvme/qwen3-work/llvm-toolchain-20261030/candidate-libraries.json \
  --reference models/qwen3/validation/reference16.json \
  --packet-recording hashes \
  --output "$qwen_run_root/rebuilt/cold"
```

这会显式选择重建库，不覆盖原资源包。中文/代码长生成、EOS、1024 容量边界、请求重置等已有回归的运行方法见 [源码构建指南](docs/qwen3/SOURCE_BUILD.md#check-the-new-libraries-on-dsp)。

## 4. 给其他人拿到同一份源码

应用和编译器是两个仓库。Qwen 专用编译策略及 DSP 辅助实现位于应用的 `models/qwen3/native/`，LLVM 复现配方位于应用的 `dependencies/llvm/`。应用通过 `third_party/hexagon-mlir` 子模块固定编译器版本；不要把另外一个目录的编译器源码随意覆盖进去。

当前模型目录拆分版本的离线源码包在：

```text
/home/kevin/hexagon/pytorch-hexagon/.local/model-layout-20261031/delivery/application-source.bundle
/home/kevin/hexagon/pytorch-hexagon/.local/model-layout-20261031/delivery/compiler-source.bundle
```

将两个文件交给接收方，在目标机器上按实际文件位置执行以下命令。示例中的 `/absolute/path/to/delivery` 需要替换，目标 `pytorch-hexagon` 目录应尚不存在：

```bash
qwen_delivery=/absolute/path/to/delivery
git clone -b review/qwen3-integration \
  "$qwen_delivery/application-source.bundle" pytorch-hexagon
cd pytorch-hexagon
git config submodule.hexagon-mlir.url "$qwen_delivery/compiler-source.bundle"
git -c protocol.file.allow=always submodule update --init third_party/hexagon-mlir
git log -1 --oneline
git -C third_party/hexagon-mlir log -1 --oneline
```

这两个离线包对应当前源码迁移版本，编译器为 `f13109d`；克隆后用上面的 `git log` 查看应用提交。旧版完整模型验收包仍保留在 `.local/llvm-toolchain-20261030/delivery/`。源码推送到远端后，也可从应用远端克隆并初始化子模块；需先确保固定的编译器提交已推送到子模块远端。

在另一块板上执行，还需要单独交付完整执行资源包，以及匹配的 QCS9075 Linux/FastRPC/DSP 环境。包内路径可以迁移，但不意味着任意系统版本或其他 DSP 都已经支持。将第 2 节的源码目录和 `qwen_bundle` 改为目标路径，先运行 `doctor --execution`，再运行 `regress`。详细资源导出方式见 [复现说明](docs/qwen3/REPRODUCE.md)。

## 5. 在现有开发主机上重新编译

新源码推荐用 `python -m hexagon_runtime qwen3 …`；旧的 `python -m qwen3 …` 命令仍转发到同一适配实现。第 2、3 节给出了当前已验证板端副本的实际命令和路径。

只运行推理可以跳过本节。以下命令在 **开发主机** 上执行，复用已验证的工具链；默认只构建编译器及应用扩展：

```bash
cd /home/kevin/hexagon/pytorch-hexagon
qwen_python=/home/kevin/miniconda3/envs/main/bin/python
qwen_config="$PWD/.local/unified-build/toolchain.json"
qwen_compiler=/home/kevin/hexagon/hexagon-mlir-review
qwen_build="$PWD/.local/model-layout-20261031/build"

"$qwen_python" dependencies/llvm/toolchain.py verify \
  --config "$qwen_config"

"$qwen_python" -m qwen3.build \
  --config "$qwen_config" --compiler-source "$qwen_compiler" \
  --output "$qwen_build" --jobs 8

"$qwen_python" -m qwen3.tools verify \
  --config "$qwen_config" --compiler-source "$qwen_compiler" \
  --build "$qwen_build" \
  --output "$PWD/.local/host-verification-$(date +%Y%m%d-%H%M%S)"
```

`qwen_build` 是此次扩展迁移后的构建目录。Ninja 会复用未变化的编译器和 DSP 辅助对象。需要重新编译模型时，先按[源码构建指南](docs/qwen3/SOURCE_BUILD.md#capture-the-build-inputs-on-arm)用当前版本重新导出两份 ARM 捕获，填写新的模型输入配置，再增加 `--models /absolute/path/to/model-inputs.json --trusted-local-producers`。旧的 `.local/unified-build/inputs/models.json` 属于此前源码版本；捕获记录校验源码哈希，不能直接当作当前版本输入。

模型输出要求 `models/q1`、`models/q2` 尚不存在。此前模型验收后已清理其两个中间目录，已验收模型另存于 `.local/llvm-toolchain-20261030/delivery/models/`。再次构建后若要重跑，应先保存需要的模型和记录，再处理旧输出，不能直接覆盖或误删交付目录。

生成的 `models/q1/model.so`、`models/q2/model.so` 需要上板验收。复用资源包的常量库前，必须确认三个新常量库的哈希与包内对应文件一致；若不同，不能只替换两个模型库。部署映射和完整回归步骤见 [源码构建指南](docs/qwen3/SOURCE_BUILD.md)。

## 6. 换一台主机从头构建

本节以获得相应 SDK、固定版本的主机 Python 环境及其补丁为前提，是构建准备清单。它尚不是从干净操作系统自动安装全部依赖的脚本；在现有开发机上的复现不能替代新主机或新板卡的环境验收。

现有 `.local/` 配置和产物不在 Git 中，主机绝对路径不能照搬。需要准备：

1. 应用及固定版本的编译器子模块、固定版本的 Triton 源码。
2. Hexagon SDK 6.5.0.0、Hexagon Tools 19.0.02、HexKL、兼容的主机 Clang，以及 CMake/Ninja 和主机系统库。
3. [源码构建指南](docs/qwen3/SOURCE_BUILD.md#inputs) 列出的固定 host Python/PyTorch/torch-mlir 等环境和补丁。板端资源包里的 ARM Python 库不能用于 x86 主机构建。
4. 模型资源，以及 ARM 实际生成流程导出的 q1、q2–32 两份可信图捕获。捕获及 `model-inputs.json` 配置方法也在该指南中。

先复制 `models/qwen3/config/build.example.json` 填写本机路径，按 [LLVM 重建说明](dependencies/llvm/README.md) 准备独立源码并构建 LLVM/MLIR/LLD。配置中的 `llvm_receipt` 指向构建生成的校验记录，`lld` 指向该构建的 `bin/ld.lld`。然后运行统一入口 `python -m models.qwen3.build`，最后做主机及板端验证。

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

运行结果放在 `results/`、`.local/` 或仓库外；权重、SDK、编译产物和一次性日志不提交到 Git。应用逻辑及 Qwen 专用编译/DSP 扩展在本仓库维护，通用后端实现与修复在 Hexagon-MLIR 仓库维护。

## 8. 按本文命令实际执行的结果（2026-10-30）

已在开发板上从上述离线包重新克隆应用、初始化编译器子模块，并在新源码目录、独立缓存和结果目录下执行。没有设置额外的 `PYTHONPATH`，使用的是现有已验收资源包和板端系统。

| 操作 | 本轮实际结果 |
| --- | --- |
| `doctor --execution` | 资源包、源码和框架检查通过 |
| 中文示例，最多 32 token | 成功生成“我是AI助手，可以为您提供帮助和支持。”；10 次 DSP 前向，EOS 提前结束 |
| 北京介绍 + Python 加法函数，最多各 64 token | 两个请求完成，共 82 次 DSP 前向；生成了北京介绍和包含 `return a + b` 的函数 |
| `regress` | 16 次完整前向对照一致，生成 `comparison.json` |
| 第 5 节源码构建 | 后端增量构建成功；新生成的 q1 对象、调用代码、三个常量库及最终 DSP 库均与已验收产物逐字节一致 |

合计 **108 次 DSP 前向**，执行记录均确认没有 CPU 模型前向回退。普通生成示例验证命令和生成流程可用；完整数值对照来自 `regress`，不把普通生成样例算成新增参考对照。

按要求控制重复验证量，本轮停止了后续 q2 重编译、整套主机测试和追加板端测试，没有将取消项记为通过。完整 LLVM、q1/q2 重建及主机验收仍由第 7 节列出的上一轮记录支持。本轮也没有验证全新操作系统的依赖安装。

本轮运行记录保存在主机 `.local/usage-repro-20261030/`；板端对应目录为 `/mnt/nvme/qwen3-work/usage-repro-20261030/`。其 `qwen3-hexagon-runtime/` 是此次实际运行的独立源码副本，`results/` 保存生成与回归结果。
