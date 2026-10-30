"""Run the real isolated optimizer; not DSP or whole-model acceptance."""
from qwen3.compiler_dependency import native_root
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
OPT = os.environ.get('QWEN_SINGLETON_SUM_OPT')
pytestmark = pytest.mark.skipif(not OPT, reason='Explicit isolated optimizer required')


def source(singleton):
    text = (native_root()/'policy/test/bounded-dynamic-softmax.mlir').read_text()
    # Keep this parametrized fixture independent of the additional FileCheck cases.
    text = text.split('// CHECK-LABEL: func.func @singleton')[0]
    if singleton:
        text = (text.replace('1x16x?x?', '1x16x1x?').replace('1x16x?x1', '1x16x1x1')
                .replace('tensor.empty(%q, %k)', 'tensor.empty(%k)')
                .replace('tensor.empty(%q)', 'tensor.empty()'))
    return text


def optimize(text):
    result = subprocess.run([OPT, '-', '--verify-each',
        '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum))'],
        input=text, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize('singleton', [False, True])
def test_dynamic_and_framework_singleton_receive_same_opt_in(singleton):
    text = optimize(source(singleton))
    correct, wrong = text.split('func.func @wrong_chain')
    assert 'hexagon.fp32_arm_softmax_' in correct
    assert 'bounded dynamic Softmax requires 1024 KV slots' in correct
    assert 'hexagon.fp32_arm_softmax_partial' in correct
    assert 'hexagon.fp32_arm_softmax_sum' in correct
    assert 'hexagon.fp32_arm_softmax_reciprocal' in correct
    assert 'hexagon.fp32_arm_softmax_normalization' in correct
    assert 'scf.for' in correct
    assert 'hexagon.fp32_arm_softmax_' not in wrong
    assert 'scf.for' not in wrong


@pytest.mark.parametrize('static_cache', [False, True])
def test_static_rows_share_the_arm_softmax_rule(static_cache):
    text = (source(False).replace('1x16x?x?', '1x16x8x?')
            .replace('1x16x?x1', '1x16x8x1')
            .replace('tensor.empty(%q, %k)', 'tensor.empty(%k)')
            .replace('tensor.empty(%q)', 'tensor.empty()'))
    if static_cache:
        text = text.replace('1x16x8x?', '1x16x8x1024').replace('tensor.empty(%k)', 'tensor.empty()')
    result = optimize(text)
    assert result.count('hexagon.fp32_arm_softmax_normalization') == 1
    assert result.count('hexagon.fp32_arm_softmax_partial') == 1
    assert 'tensor<16x256x4xf32>' in result


@pytest.mark.parametrize('singleton', [False, True])
def test_no_annotation_no_reassociation(singleton):
    text = source(singleton).replace(
        'attributes {hexagon.fp32_dynamic_softmax_max_q = 32 : i64}', '')
    assert 'hexagon.fp32_arm_softmax_' not in optimize(text)


@pytest.mark.parametrize('change', ['unshifted', 'wrong_division', 'wrong_heads', 'wrong_initializer'])
def test_singleton_requires_actual_softmax_chain(change):
    text = source(True)
    if change == 'unshifted':
        text = text.replace('ins(%shift :', 'ins(%scores :')
    elif change == 'wrong_division':
        text = text.replace('arith.divf %value, %denominator', 'arith.divf %denominator, %value')
    elif change == 'wrong_heads':
        text = text.replace('1x16x', '1x8x')
    else:
        text = text.replace('arith.constant 0.0 : f32', 'arith.constant 1.0 : f32')
    assert 'hexagon.fp32_arm_softmax_' not in optimize(text)
