"""Check the per-head RMSNorm + RoPE fold (--qwen-fuse-head-norm-rope).

The fixture is the prepared form after --qwen-fuse-parallel-rope: the RoPE call
reads W * (X * rsqrt(S / 128 + eps)) where S is the marked ARM-ordered
sum-of-squares loop over X. Unmarked sums and shared intermediates are kept.
"""
import os
import subprocess

import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')
ANY4 = 'memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>>'

SOURCE = '''
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#scalar = affine_map<(d0, d1, d2, d3) -> ()>
#head = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#lane = affine_map<(d0, d1, d2, d3) -> (d3)>
module {
  func.func private @qwen_rope_heads_v1(ANY4, ANY4, ANY4, ANY4) attributes {llvm.emit_c_interface}
  func.func @layer(%x: tensor<1x4x2x128xf32>, %c: ANY4, %s: ANY4, %o: ANY4) -> tensor<1x4x2x128xf32> {
    %w = arith.constant dense<1.5> : tensor<128xf32>
    %eps = arith.constant dense<9.99999997E-7> : tensor<f32>
    %n128 = arith.constant 1.280000e+02 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant ZERO : f32
    %empty = tensor.empty() : tensor<1x4x2x1xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x4x2x1xf32>) -> tensor<1x4x2x1xf32>
    %wide = tensor.empty() : tensor<1x4x2x128xf32>
    %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<1x4x2x1xf32>) {
      %row = tensor.extract_slice %x[0, %i, 0, 0] [1, 1, 2, 128] [1, 1, 1, 1] : tensor<1x4x2x128xf32> to tensor<1x1x2x128xf32>
      %part = tensor.empty() {MARKER} : tensor<1x1x2x1xf32>
      %next = tensor.insert_slice %part into %acc[0, %i, 0, 0] [1, 1, 2, 1] [1, 1, 1, 1] : tensor<1x1x2x1xf32> into tensor<1x4x2x1xf32>
      scf.yield %next : tensor<1x4x2x1xf32>
    }
    %mean = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%sum : tensor<1x4x2x1xf32>) outs(%init : tensor<1x4x2x1xf32>) {
    ^bb0(%a: f32, %out: f32):
      %v = arith.divf %a, %n128 : f32
      linalg.yield %v : f32
    } -> tensor<1x4x2x1xf32>
    %plus = linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mean, %eps : tensor<1x4x2x1xf32>, tensor<f32>) outs(%init : tensor<1x4x2x1xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<1x4x2x1xf32>
    %rs = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%plus : tensor<1x4x2x1xf32>) outs(%init : tensor<1x4x2x1xf32>) {
    ^bb0(%a: f32, %out: f32):
      %v = math.rsqrt %a : f32
      linalg.yield %v : f32
    } -> tensor<1x4x2x1xf32>
    %n = linalg.generic {indexing_maps = [#id, HEADMAP, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x, %rs : tensor<1x4x2x128xf32>, tensor<1x4x2x1xf32>) outs(%wide : tensor<1x4x2x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<1x4x2x128xf32>
    %y = linalg.generic {indexing_maps = [#lane, #id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%w, %n : tensor<128xf32>, tensor<1x4x2x128xf32>) outs(%wide : tensor<1x4x2x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<1x4x2x128xf32>
    %c1d = arith.constant 1 : index
    %rows = tensor.dim %y, %c1d : tensor<1x4x2x128xf32>
    %yb = bufferization.to_buffer %y read_only : tensor<1x4x2x128xf32> to memref<1x4x2x128xf32>
    %yc = memref.cast %yb : memref<1x4x2x128xf32> to ANY4
    call @qwen_rope_heads_v1(%yc, %c, %s, %o) : (ANY4, ANY4, ANY4, ANY4) -> ()
    %keep = tensor.empty(%rows) : tensor<1x?x2x128xf32>
    return RESULT : tensor<1x4x2x128xf32>
  }
}
'''.replace('ANY4', ANY4)


def run(source, flag):
    if not OPT:
        pytest.skip('set QWEN_PACKED_I8_OPT to the built qwen-packed-i8-opt')
    return subprocess.run([OPT, '-', flag, '--verify-each'], input=source, text=True,
                          capture_output=True, timeout=60)


def fixture(marker='hexagon.fp32_arm_rmsnorm_sum', result='%x', zero='0.000000e+00', head='#head'):
    source = SOURCE.replace('MARKER', marker).replace('RESULT', result).replace('ZERO', zero)
    if head != '#head':  # an unbroadcast factor needs a full-width operand
        source = source.replace('ins(%x, %rs : tensor<1x4x2x128xf32>, tensor<1x4x2x1xf32>)',
                                'ins(%x, %x : tensor<1x4x2x128xf32>, tensor<1x4x2x128xf32>)')
    return source.replace('HEADMAP', head)


@pytest.mark.parametrize('marker', ['hexagon.fp32_arm_rmsnorm_sum', 'hexagon.fp32_arm_k_norm_sum'])
def test_norm_feeding_rope_is_folded_into_the_kernel_call(marker):
    result = run(fixture(marker), '--qwen-fuse-head-norm-rope=expected-count=1')
    assert result.returncode == 0, result.stderr
    ir = result.stdout
    assert 'call @qwen_rope_heads_v1(' not in ir
    assert ir.count('call @qwen_rope_heads_norm_v1(') == 1
    assert 'func.func private @qwen_rope_heads_norm_v1(' + ', '.join([ANY4, 'memref<?xf32, strided<[?], offset: ?>>',
                                                                     ANY4, ANY4, ANY4, 'f32']) + ')' in ir
    # The kernel gets X itself, the weight and the graph's epsilon constant.
    assert 'bufferization.to_buffer %arg0 read_only' in ir
    assert 'arith.constant 9.99999997E-7 : f32' in ir
    assert 'fused head norm RoPE: 1' in result.stderr
    # The RoPE fusion's tensor.dim of the normalized value now reads X.
    assert 'tensor.dim %arg0' in ir


def test_unmarked_sum_is_kept():
    result = run(fixture('other.attribute'), '--qwen-fuse-head-norm-rope')
    assert result.returncode == 0, result.stderr
    assert 'call @qwen_rope_heads_v1(' in result.stdout and 'qwen_rope_heads_norm_v1' not in result.stdout
    assert 'fused head norm RoPE: 0' in result.stderr


def test_shared_normalized_value_is_kept():
    result = run(fixture(result='%y'), '--qwen-fuse-head-norm-rope')
    assert result.returncode == 0, result.stderr
    assert 'qwen_rope_heads_norm_v1' not in result.stdout


@pytest.mark.parametrize('kwargs', [dict(zero='1.000000e+00'), dict(head='#id')])
def test_other_normalization_is_kept(kwargs):
    # A sum not started from zero, or a factor that is not the per-head rsqrt
    # broadcast, is not the graph's RMSNorm.
    result = run(fixture(**kwargs), '--qwen-fuse-head-norm-rope')
    assert result.returncode == 0, result.stderr
    assert 'qwen_rope_heads_norm_v1' not in result.stdout and 'fused head norm RoPE: 0' in result.stderr


def test_unexpected_count_is_rejected():
    result = run(fixture('other.attribute'), '--qwen-fuse-head-norm-rope=expected-count=1')
    assert result.returncode != 0 and 'expected 1, got 0' in result.stderr
