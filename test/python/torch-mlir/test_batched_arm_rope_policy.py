"""Check the real four-lane reuse pass and its numerical grouping boundaries."""
import os
from pathlib import Path
import re
import subprocess

import pytest


@pytest.fixture(scope="module")
def tools():
    selected = os.environ.get("QWEN_SILU_EFFECT_OPT")
    if not selected:
        pytest.skip("Explicit built policy optimizer required")
    batch = Path(selected)
    sums = batch.with_name("qwen-policy-sums-opt")
    assert batch.is_file() and sums.is_file()
    return str(batch), str(sums)


def invoke(tool, text, *options):
    return subprocess.run([tool, "-", "--verify-each", *options], input=text,
                          text=True, capture_output=True, timeout=30)


def original(tools, dynamic=False, canonical=False):
    root = Path(__file__).resolve().parents[3]
    text = (root / "models/qwen3/native/policy/test/arm-rope-trig-order.mlir").read_text()
    if dynamic:
        text = text.replace("tensor<1x1x128xf32>", "tensor<1x?x128xf32>")
        text = text.replace("%empty = tensor.empty()", """
        %axis = arith.constant 1 : index
        %rows = tensor.dim %x, %axis : tensor<1x?x128xf32>
        %empty = tensor.empty(%rows)""")
    passes = "func.func(stabilize-fp32-sum)" + (",canonicalize" if canonical else "")
    options = ["--pass-pipeline=builtin.module(" + passes + ")"]
    result = invoke(tools[1], text, *options)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("dynamic,canonical", [(False, False), (True, False), (True, True)])
def test_reuses_original_group_through_explicit_destination_abi(tools, dynamic, canonical):
    text = original(tools, dynamic, canonical)
    disabled = invoke(tools[0], text)
    assert disabled.returncode == 0, disabled.stderr
    assert "func.call @__hexagon_arm_sin4_lane" in disabled.stdout
    result = invoke(tools[0], text, "--qwen-batch-arm-rope")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("func.call @__hexagon_arm_sin4(") == 1
    assert result.stdout.count("func.call @__hexagon_arm_cos4(") == 1
    assert "func.call @__hexagon_arm_sin4_lane" not in result.stdout
    assert "memref<4xf32>, f32, f32, f32, f32" in result.stdout
    assert result.stdout.count("hexagon.fp32_arm_rope_trig4_batch") == 2
    assert result.stdout.count("memref.load") == 8


@pytest.mark.parametrize("change", [
    lambda s: s.replace("linalg.index 2", "linalg.index 1"),
    lambda s: s.replace("arith.constant 4 : index", "arith.constant 8 : index"),
    lambda s: s.replace("d2 floordiv 4", "d2 floordiv 8"),
    lambda s: s.replace("hexagon.fp32_arm_rope_trig4}", "hexagon.fp32_arm_rope_trig4 = false}"),
    lambda s: re.sub(r"func.call @__hexagon_arm_sin4_lane\((%\w+), (%\w+)",
                     r"func.call @__hexagon_arm_sin4_lane(\2, \1", s),
])
def test_rejects_changed_lane_group_or_marker(tools, change):
    text = original(tools)
    changed = change(text)
    assert changed != text
    result = invoke(tools[0], changed, "--qwen-batch-arm-rope")
    assert result.returncode > 0 and "error:" in result.stderr


def test_requires_exact_coverage_and_rejects_runtime_symbol_collision(tools):
    text = original(tools)
    assert invoke(tools[0], text, "--qwen-batch-arm-rope=expected-count=1").returncode > 0
    collision = text.replace("module {", "module {\nfunc.func private @__hexagon_arm_sin4()")
    result = invoke(tools[0], collision, "--qwen-batch-arm-rope")
    assert result.returncode > 0 and "symbol already exists" in result.stderr


@pytest.mark.parametrize("repeated", [True, False])
def test_reuses_half_only_when_concat_inputs_are_the_same_value(tools, repeated):
    text = original(tools, dynamic=True)
    argument = re.search(r"func.func @trig\((%\w+): tensor<1x\?x128xf32>\)", text)
    assert argument
    value = argument[1]
    text = text.replace(argument[0], "func.func @trig(%a: tensor<1x?x64xf32>, "
                        "%b: tensor<1x?x64xf32>)")
    start = text.index(" {\n", text.index("func.func @trig(")) + 3
    concat = (f"    {value} = tensor.concat dim(2) %a, "
              f"{'%a' if repeated else '%b'} : "
              "(tensor<1x?x64xf32>, tensor<1x?x64xf32>) -> tensor<1x?x128xf32>\n")
    text = text[:start] + concat + text[start:]
    result = invoke(tools[0], text, "--qwen-batch-arm-rope")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("tensor.insert ") == (16 if repeated else 8)
    assert result.stdout.count("func.call @__hexagon_arm_sin4(") == 1
    assert result.stdout.count("func.call @__hexagon_arm_cos4(") == 1
