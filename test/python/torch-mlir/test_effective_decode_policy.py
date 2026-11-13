"""Check native decode dispatch while keeping unrelated contractions unchanged."""
import os
import subprocess

import pytest


OPT = os.environ.get('QWEN_EFFECTIVE_FIRST_OPT')
pytestmark = pytest.mark.skipif(not OPT, reason='Explicit numerical optimizer required')
SOURCE = '''
func.func @decode(%q: tensor<16x1x128xf32>, %k: tensor<16x128x?xf32>,
                  %p: tensor<16x1x?xf32>, %v: tensor<16x?x128xf32>)
    -> (tensor<16x1x?xf32>, tensor<16x1x128xf32>)
    attributes {hexagon.fp32_effective_decode_attention = true} {
  %c2 = arith.constant 2 : index
  %length = tensor.dim %k, %c2 : tensor<16x128x?xf32>
  %zero = arith.constant 0.0 : f32
  %qe = tensor.empty(%length) : tensor<16x1x?xf32>
  %qi = linalg.fill ins(%zero : f32) outs(%qe : tensor<16x1x?xf32>) -> tensor<16x1x?xf32>
  %qk = linalg.batch_matmul ins(%q, %k : tensor<16x1x128xf32>, tensor<16x128x?xf32>)
      outs(%qi : tensor<16x1x?xf32>) -> tensor<16x1x?xf32>
  %pe = tensor.empty() : tensor<16x1x128xf32>
  %pi = linalg.fill ins(%zero : f32) outs(%pe : tensor<16x1x128xf32>) -> tensor<16x1x128xf32>
  %pv = linalg.batch_matmul ins(%p, %v : tensor<16x1x?xf32>, tensor<16x?x128xf32>)
      outs(%pi : tensor<16x1x128xf32>) -> tensor<16x1x128xf32>
  return %qk, %pv : tensor<16x1x?xf32>, tensor<16x1x128xf32>
}
'''


def optimize(text):
    result = subprocess.run([OPT, '-', '--verify-each',
        '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum))'],
        input=text, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_qk_dispatches_short_rows_and_pv_preserves_fma():
    text = optimize(SOURCE)
    assert text.count('math.fma') == 2
    assert text.count('scf.if') == 1
    assert text.count('linalg.batch_matmul') == 1  # Small native QK branch.
    assert 'arith.cmpi ult' in text and 'arith.constant 4 : index' in text
    assert text.count('hexagon.fp32_effective_decode_order') == 2


@pytest.mark.parametrize('length', [1, 2, 3])
def test_native_singleton_specialization_keeps_the_decode_policy(length):
    source = SOURCE.replace('?', str(length)).replace(
        'tensor.empty(%length)', 'tensor.empty()')
    result = optimize(source)
    assert result.count('hexagon.fp32_effective_decode_order') == 2
    assert result.count('math.fma') == 2
    unchanged = optimize(source.replace(
        'attributes {hexagon.fp32_effective_decode_attention = true}', ''))
    assert unchanged.count('linalg.batch_matmul') == 2
    assert 'math.fma' not in unchanged


@pytest.mark.parametrize('change', ['unannotated', 'nonzero', 'multi_query'])
def test_unrelated_contractions_keep_their_original_arithmetic(change):
    text = SOURCE
    if change == 'unannotated':
        text = text.replace('attributes {hexagon.fp32_effective_decode_attention = true}', '')
    elif change == 'nonzero':
        text = text.replace('arith.constant 0.0 : f32', 'arith.constant 1.0 : f32')
    else:
        text = text.replace('16x1x', '16x2x')
    result = optimize(text)
    assert result.count('linalg.batch_matmul') == 2
    assert 'math.fma' not in result


def test_prefill_uses_effective_reduction_blocks_with_native_dispatch():
    text = SOURCE.replace('16x1x', '16x2x').replace(
        'hexagon.fp32_effective_decode_attention', 'hexagon.fp32_effective_prefill_attention')
    result = optimize(text)
    assert 'linalg.batch_matmul' not in result
    assert result.count('math.fma') == 2
    assert 'hexagon.fp32_effective_prefill_pv_order' in result
    assert 'hexagon.fp32_effective_pv_block' in result
    assert 'arith.constant 4096 : index' in result
    assert 'arith.constant 352 : index' in result
    assert 'effective prefill requires query2..32 and query<=KV<=1024' in result
    assert 'requires 1024 KV slots' not in result
