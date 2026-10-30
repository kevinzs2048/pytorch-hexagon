"""Exercise the real rewrite's opt-in, shape and external ABI boundaries."""
from qwen3.compiler_dependency import native_root
import os
from pathlib import Path
import subprocess

import pytest
from apply_dynamic_bound_qk_policy import FunctionPolicyPatch

ROOT = Path(__file__).resolve().parents[3]


def optimize(source):
    optimizer = os.environ.get('QWEN_SINGLETON_SUM_OPT')
    if not optimizer:
        pytest.skip('Explicit isolated optimizer required')
    return subprocess.run([optimizer, '-', '--verify-each',
        '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum))'],
        input=source, text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize('shape,expected', [
    ('1x1x128', 2), ('1x32x128', 2), ('1x33x128', 0),
    ('1x0x128', 0), ('2x1x128', 0), ('1x1x64', 0),
])
def test_admitted_shapes_and_idempotence(shape, expected):
    source = (native_root()/'policy/test/arm-rope-trig-order.mlir').read_text()
    result = optimize(source.replace('1x1x128', shape))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('func.call @__hexagon_arm_') == expected
    second = optimize(result.stdout)
    assert second.returncode == 0 and second.stdout == result.stdout


@pytest.mark.parametrize('change,fails', [
    ('no_opt_in', False), ('false', True), ('wrong_type', True),
    ('missing_ABI', True), ('wrong_ABI', True), ('modified_body', False),
])
def test_no_implicit_or_malformed_rewrite(change, fails):
    source = (native_root()/'policy/test/arm-rope-trig-order.mlir').read_text()
    source = '\n'.join(line for line in source.splitlines() if not line.startswith('//'))
    if change == 'no_opt_in':
        source = source.replace(',hexagon.fp32_arm_rope_trig4=true', '')
    elif change == 'false':
        source = source.replace('trig4=true', 'trig4=false')
    elif change == 'wrong_type':
        source = source.replace('trig4=true', 'trig4=1 : i64')
    elif change == 'missing_ABI':
        source = '\n'.join(line for line in source.splitlines() if 'func.func private' not in line)
    elif change == 'wrong_ABI':
        source = source.replace('f32,f32,f32,f32,i32', 'f32,f32,f32,f32,i64')
    elif change == 'modified_body':
        source = source.replace('linalg.yield %v : f32',
            '%extra = arith.addf %v, %in : f32\n    linalg.yield %extra : f32')
    result = optimize(source)
    assert (result.returncode != 0) == fails, result.stderr
    assert 'func.call @__hexagon_arm_' not in result.stdout


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('chunk_size', [1, 7, 4096])
def test_streamed_policy_keeps_old_recipes_unchanged(enabled, chunk_size):
    source = (b'module {\n  func.func @Qwen3DynamicBoundProbe() {\n'
              b'    return\n  }\n}\n')
    chunks = []
    patch = FunctionPolicyPatch(chunks.append, arm_rope_trig=enabled)
    for offset in range(0, len(source), chunk_size):
        patch.write(source[offset:offset+chunk_size])
    patch.finish()
    result = b''.join(chunks)
    assert result.count(b'hexagon.fp32_arm_rope_trig4 = true') == int(enabled)
    assert result.count(b'func.func private @__hexagon_arm_') == 2*int(enabled)
