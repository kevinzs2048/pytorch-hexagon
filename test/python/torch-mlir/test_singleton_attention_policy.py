"""Real isolated optimizer checks for q1 coverage; no numerical/DSP claim."""
import os
from pathlib import Path
import subprocess

import pytest

from test_bounded_softmax_singleton_policy import source as softmax_source

OPT = os.environ.get('QWEN_SINGLETON_ATTENTION_OPT')
pytestmark = pytest.mark.skipif(not OPT, reason='Explicit isolated attention optimizer required')
POLICY = ('attributes {hexagon.fp32_prefill_query = 17 : i64, '
          'hexagon.fp32_prefill_capacity = 1024 : i64, '
          'hexagon.fp32_prefill_order = "qk-fma-pv-openblas-armv8-k1024"}')


def optimize(text, flags):
    result = subprocess.run([OPT, '-', '--verify-each',
        '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum{'+flags+'}))'],
        input=text, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def dots(query, annotation=True, initializer='0.0'):
    q = '?' if query is None else str(query)
    a, b, p, v = (f'tensor<16x{q}x128xf32>', 'tensor<16x128x?xf32>',
                  f'tensor<16x{q}x?xf32>', 'tensor<16x?x128xf32>')
    sizes = '%q, %k' if query is None else '%k'
    row_size = '%q' if query is None else ''
    return f'''func.func @dots(%a: {a}, %b: {b}, %p: {p}, %v: {v}) -> ({p}, {a})
      {POLICY if annotation else ''} {{
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %q = tensor.dim %a, %c1 : {a}
      %k = tensor.dim %b, %c2 : {b}
      %zero = arith.constant {initializer} : f32
      %qe = tensor.empty({sizes}) : {p}
      %qi = linalg.fill ins(%zero : f32) outs(%qe : {p}) -> {p}
      %qk = linalg.batch_matmul ins(%a, %b : {a}, {b}) outs(%qi : {p}) -> {p}
      %pe = tensor.empty({row_size}) : {a}
      %pi = linalg.fill ins(%zero : f32) outs(%pe : {a}) -> {a}
      %pv = linalg.batch_matmul ins(%p, %v : {p}, {v}) outs(%pi : {a}) -> {a}
      return %qk, %pv : {p}, {a}
    }}'''


@pytest.mark.parametrize('query', [None, 1, 2, 8, 32])
def test_same_qk_and_pv_policy_for_symbolic_and_static_rows(query):
    text = optimize(dots(query), 'fp32-attention-dot-order=true')
    assert text.count('hexagon.fp32_prefill_dot_order') == 2
    assert text.count('hexagon.fp32_prefill_pv_block_start') == (4 if query is None else 1 if query == 1 else 3)
    assert 'pv-armv8-gemv1-gemm352-336-336-fma-ordered-add' in text
    for start in ((0,) if query == 1 else (0, 352, 688)):
        assert f'hexagon.fp32_prefill_pv_block_start = {start} : i64' in text
    assert ('scf.if' in text) == (query is None)
    assert 'linalg.batch_matmul' not in text
    assert text.count('cf.assert') == 2


def test_obsolete_short_context_pv_order_is_rejected():
    text = dots(None).replace('qk-fma-pv-openblas-armv8-k1024', 'qk-fma-pv192')
    result = subprocess.run([OPT, '-', '--verify-each',
        '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum{fp32-attention-dot-order=true}))'],
        input=text, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'explicit prefill order requires' in result.stderr


@pytest.mark.parametrize('query,annotation,initializer', [(1, False, '0.0'),
                                                        (1, True, '1.0'), (33, True, '0.0')])
def test_does_not_expand_unannotated_or_unmatched_policy(query, annotation, initializer):
    text = optimize(dots(query, annotation, initializer), 'fp32-attention-dot-order=true')
    assert 'hexagon.fp32_prefill_dot_order' not in text
    assert text.count('linalg.batch_matmul') == 2


@pytest.mark.parametrize('singleton', [False, True])
@pytest.mark.parametrize('stable', [False, True])
def test_softmax_exponent_requires_complete_stable_chain(singleton, stable):
    text = softmax_source(singleton).replace(
        'attributes {hexagon.fp32_dynamic_softmax_max_q = 32 : i64}', POLICY)
    if not stable:
        text = text.replace('ins(%shift :', 'ins(%scores :')
    text = 'func.func private @__hexagon_softmax_exp_compat(f32) -> f32\n'+text
    result = optimize(text, 'short-softmax-exp-compat=true')
    assert result.count('call @__hexagon_softmax_exp_compat') == int(stable)


def silu(query):
    q = '?' if query is None else str(query)
    t = f'tensor<1x{q}x3072xf32>'
    size = '%q' if query is None else ''
    return f'''func.func private @__hexagon_silu_exp_compat(f32) -> f32
    func.func @silu(%x: {t}, %other: {t}) -> {t} {POLICY} {{
      %c1 = arith.constant 1 : index
      %q = tensor.dim %x, %c1 : {t}
      %empty = tensor.empty({size}) : {t}
      %one = arith.constant 1.0 : f32
      %sig = linalg.generic {{indexing_maps = [affine_map<(a,b,c)->(a,b,c)>,
          affine_map<(a,b,c)->(a,b,c)>], iterator_types = ["parallel", "parallel", "parallel"]}}
          ins(%x : {t}) outs(%empty : {t}) {{
        ^bb0(%value: f32, %unused: f32):
          %neg = arith.negf %value : f32
          %exp = math.exp %neg : f32
          %sum = arith.addf %exp, %one : f32
          %reciprocal = arith.divf %one, %sum : f32
          linalg.yield %reciprocal : f32
      }} -> {t}
      %out = linalg.generic {{indexing_maps = [affine_map<(a,b,c)->(a,b,c)>,
          affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,b,c)>],
          iterator_types = ["parallel", "parallel", "parallel"]}}
          ins(%sig, %x : {t}, {t}) outs(%empty : {t}) {{
        ^bb0(%sigmoid: f32, %value: f32, %unused: f32):
          %mul = arith.mulf %sigmoid, %value : f32
          linalg.yield %mul : f32
      }} -> {t}
      return %out : {t}
    }}'''


@pytest.mark.parametrize('query', [None, 1, 8])
def test_inline_silu_symbolic_and_static_rows(query):
    result = optimize(silu(query), 'fp32-silu-exp-compat=true')
    assert result.count('call @__hexagon_silu_exp_compat') == 1


@pytest.mark.parametrize('change', ['wrong_consumer', 'wrong_division'])
def test_silu_shape_alone_never_suffices(change):
    text = silu(1)
    if change == 'wrong_consumer':
        text = text.replace('ins(%sig, %x :', 'ins(%sig, %other :')
    else:
        text = text.replace('arith.divf %one, %sum', 'arith.divf %sum, %one')
    assert 'call @__hexagon_silu_exp_compat' not in optimize(text, 'fp32-silu-exp-compat=true')
