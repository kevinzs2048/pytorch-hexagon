"""Admit only checked row scales; preserve failure checks and exact division."""
import os
import subprocess
import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')
pytestmark = pytest.mark.skipif(not OPT, reason='QWEN_PACKED_I8_OPT required')


def source():
    return '''#id = affine_map<(a,b,c)->(a,b,c)>
module {
 func.func @Qwen3DynamicPrefill(%a: tensor<1x?x1xf32>, %init: tensor<1x?x1xf32>) -> tensor<1x?x1xf32> {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %inverse = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel","parallel"]}
    ins(%a: tensor<1x?x1xf32>) outs(%init: tensor<1x?x1xf32>) {
    ^bb0(%x: f32, %unused: f32):
     %ok = arith.cmpf one, %x, %zero : f32
     cf.assert %ok, "nonzero scale required"
     %r = arith.divf %one, %x : f32
     linalg.yield %r : f32
  } -> tensor<1x?x1xf32>
  %result = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel","parallel"]}
    ins(%inverse: tensor<1x?x1xf32>) outs(%init: tensor<1x?x1xf32>) {
    ^bb0(%x: f32, %unused: f32):
     %r = math.roundeven %x : f32
     linalg.yield %r : f32
  } -> tensor<1x?x1xf32>
  return %result : tensor<1x?x1xf32>
 }
}'''


def run(text, count=1):
    return subprocess.run([OPT, '-', f'--qwen-vectorize-activation=expected-count={count}',
                           '--verify-each'], input=text, text=True,
                          capture_output=True, timeout=30)


def test_checks_are_before_division_and_rounding_preserves_ieee_operations():
    result = run(source())
    assert result.returncode == 0, result.stderr
    text = result.stdout
    assert text.count('cf.assert') == text.count('arith.divf') == 1
    assert text.index('scf.for') < text.index('cf.assert') < text.index('arith.divf')
    assert 'math.roundeven' not in text
    assert 'arith.bitcast' in text and 'arith.cmpi uge' in text
    assert text.count('arith.addf') == text.count('arith.subf') == 1
    assert text.index('arith.addf') < text.index('arith.subf')
    assert '0x4B000000 : f32' in text and '1258291200' in text
    assert 'fastmath' not in text and 'arith.shli' not in text


@pytest.mark.parametrize('old,new', [
    ('@Qwen3DynamicPrefill', '@foreign_model'),
    ('@Qwen3DynamicPrefill', '@Qwen3DynamicCache1'),
    ('cmpf one', 'cmpf ogt'),
    ('divf %one', 'divf %zero'),
    ('1x?x1', '1x0x1'),
    ('1x?x1', '1x33x1'),
    ('1x?x1', '1x?x2'),
    ('(a,b,c)->(a,b,c)', '(a,b,c)->(a,c,b)'),
    ('math.roundeven', 'math.round'),
])
def test_rejects_unsupported_contract(old, new):
    result = run(source().replace(old, new))
    assert result.returncode != 0


def test_coverage_is_fail_closed():
    assert run(source(), count=2).returncode != 0


def test_prefill_last_token_projection_keeps_single_row_scale():
    result = run(source().replace('1x?x1', '1x1x1'))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('owner', ['Qwen3DynamicSingle0', 'Qwen3DynamicCache1'])
@pytest.mark.parametrize('rows', ['1', '2', '?'])
def test_single_token_entries_require_one_row(owner, rows):
    text = source().replace('Qwen3DynamicPrefill', owner).replace('1x?x1', f'1x{rows}x1')
    result = run(text)
    assert (result.returncode == 0) == (rows == '1'), result.stderr
