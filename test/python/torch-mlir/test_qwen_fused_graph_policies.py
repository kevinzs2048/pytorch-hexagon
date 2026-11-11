"""Check the fused INT8 dequantization graph rewrite.

Each fixture is the replaced graph chain in its prepared form; negative cases
keep the original operations when a matched precondition does not hold.
"""
import os
import subprocess

import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')

MATMUL = ('memref<?x?xi8, strided<[?, ?], offset: ?>>, memref<?x?xi8, strided<[?, ?], offset: ?>>, '
          'memref<?x?xi32, strided<[?, ?], offset: ?>>')

DEQUANT = '''
#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func private @__hexagon_i8_prepacked_v1_matmul(MATMUL) attributes {llvm.emit_c_interface}
  func.func @graph(%a: memref<?x?xi8, strided<[?, ?], offset: ?>>, %w: memref<?x?xi8, strided<[?, ?], offset: ?>>,
                   %s: tensor<?xf32>, %c: tensor<1024xf32>, %r: tensor<1x?x1024xf32>, %n: index) -> tensor<1x?x1024xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %alloc = memref.alloc(%n) : memref<?x1024xi32>
    linalg.fill ins(%zero : i32) outs(%alloc : memref<?x1024xi32>)
    %cast = memref.cast %alloc : memref<?x1024xi32> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    call @__hexagon_i8_prepacked_v1_matmul(%a, %w, %cast) : (MATMUL) -> ()
    %t = bufferization.to_tensor %alloc restrict writable : memref<?x1024xi32> to tensor<?x1024xi32>
    %e = tensor.empty(%n) : tensor<?x1024xf32>
    %f = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%t : tensor<?x1024xi32>) outs(%e : tensor<?x1024xf32>) {
    ^bb0(%i: i32, %o: f32):
      %v = arith.sitofp %i : i32 to f32
      linalg.yield %v : f32
    } -> tensor<?x1024xf32>
    %sx = tensor.expand_shape %s [[0, 1]] output_shape [%n, 1] : tensor<?xf32> into tensor<?x1xf32>
    %b = linalg.generic {indexing_maps = [#id], iterator_types = ["parallel", "parallel"]} outs(%e : tensor<?x1024xf32>) {
    ^bb0(%o: f32):
      %i = linalg.index 0 : index
      %eq = arith.cmpi eq, %n, %c1 : index
      %sel = arith.select %eq, %c0, %i : index
      %x = tensor.extract %sx[%sel, %c0] : tensor<?x1xf32>
      linalg.yield %x : f32
    } -> tensor<?x1024xf32>
    %m1 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%f, %b : tensor<?x1024xf32>, tensor<?x1024xf32>) outs(%e : tensor<?x1024xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.mulf %x, %y FASTMATH : f32
      linalg.yield %v : f32
    } -> tensor<?x1024xf32>
    %m2 = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel", "parallel"]}
        ins(%m1, %c : tensor<?x1024xf32>, tensor<1024xf32>) outs(%e : tensor<?x1024xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.mulf %x, %y : f32
      linalg.yield %v : f32
    } -> tensor<?x1024xf32>
    %x = tensor.expand_shape %m2 [[0, 1], [2]] output_shape [1, %n, 1024] : tensor<?x1024xf32> into tensor<1x?x1024xf32>
    %e3 = tensor.empty(%n) : tensor<1x?x1024xf32>
    %out = linalg.generic {indexing_maps = [#id3, #id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%r, %x : tensor<1x?x1024xf32>, tensor<1x?x1024xf32>) outs(%e3 : tensor<1x?x1024xf32>) {
    ^bb0(%p: f32, %q: f32, %o: f32):
      %v = arith.addf %p, %q : f32
      linalg.yield %v : f32
    } -> tensor<1x?x1024xf32>
    return %out : tensor<1x?x1024xf32>
  }
}
'''.replace('MATMUL', MATMUL)

def run(source, flag):
    if not OPT:
        pytest.skip('set QWEN_PACKED_I8_OPT to the built qwen-packed-i8-opt')
    result = subprocess.run([OPT, '-', flag, '--verify-each'], input=source, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stderr


def test_dequant_and_residual_become_one_runtime_call():
    ir, log = run(DEQUANT.replace('FASTMATH', ''), '--qwen-fuse-i8-dequant=expected-count=1 expected-residual=1')
    assert 'call @__hexagon_i8_prepacked_v1_matmul_dequant_residual(' in ir
    assert 'call @__hexagon_i8_prepacked_v1_matmul(' not in ir
    assert 'arith.sitofp' not in ir and 'arith.addf' not in ir and 'linalg.fill' not in ir
    assert '0 plain, 1 residual, 0 product' in log


def test_dequant_with_fast_math_is_left_unchanged():
    ir, log = run(DEQUANT.replace('FASTMATH', 'fastmath<contract>'), '--qwen-fuse-i8-dequant')
    assert 'call @__hexagon_i8_prepacked_v1_matmul(' in ir
    assert 'dequant' not in ir.replace('qwen-fuse-i8-dequant', '')
    assert '0 plain, 0 residual, 0 product' in log
