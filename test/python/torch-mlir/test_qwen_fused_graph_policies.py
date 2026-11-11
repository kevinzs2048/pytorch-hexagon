"""Check the fused INT8 dequantization, parallel RoPE, trig and attention-mask graph rewrites.

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

ROPE = '''
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#heads = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
module {
  func.func @graph(%y: tensor<1x?x2x128xf32>, %cos: tensor<1x1x?x128xf32>, %sin: tensor<1x1x?x128xf32>, %n: index)
      -> RESULTS {
    %et = tensor.empty(%n) : tensor<1x2x?x128xf32>
    %x = linalg.transpose ins(%y : tensor<1x?x2x128xf32>) outs(%et : tensor<1x2x?x128xf32>) permutation = [0, 2, 1, 3]
    %m1 = linalg.generic {indexing_maps = [#id, #heads, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%x, %cos : tensor<1x2x?x128xf32>, tensor<1x1x?x128xf32>) outs(%et : tensor<1x2x?x128xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<1x2x?x128xf32>
    %lo = tensor.extract_slice %x[0, 0, 0, 0] [1, 2, %n, 64] [1, 1, 1, 1] : tensor<1x2x?x128xf32> to tensor<1x2x?x64xf32>
    %hi = tensor.extract_slice %x[0, 0, 0, 64] [1, 2, %n, 64] [1, 1, 1, 1] : tensor<1x2x?x128xf32> to tensor<1x2x?x64xf32>
    %eh = tensor.empty(%n) : tensor<1x2x?x64xf32>
    %neg = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%hi : tensor<1x2x?x64xf32>) outs(%eh : tensor<1x2x?x64xf32>) {
    ^bb0(%a: f32, %o: f32):
      %v = arith.negf %a : f32
      linalg.yield %v : f32
    } -> tensor<1x2x?x64xf32>
    %cat = tensor.concat dim(3) %neg, %lo : (tensor<1x2x?x64xf32>, tensor<1x2x?x64xf32>) -> tensor<1x2x?x128xf32>
    %m2 = linalg.generic {indexing_maps = [#id, #heads, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%cat, %sin : tensor<1x2x?x128xf32>, tensor<1x1x?x128xf32>) outs(%et : tensor<1x2x?x128xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<1x2x?x128xf32>
    %out = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%m1, %m2 : tensor<1x2x?x128xf32>, tensor<1x2x?x128xf32>) outs(%et : tensor<1x2x?x128xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<1x2x?x128xf32>
    return RETURNS
  }
}
'''

TRIG = '''
module {
  func.func private @__hexagon_arm_cos4(memref<4xf32>, f32, f32, f32, f32)
  func.func @graph(%x: tensor<1x?x128xf32>, %n: index) -> tensor<1x?x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %cs = arith.constant STEP : index
    %init = tensor.empty(%n) : tensor<1x?x128xf32>
    %t = memref.alloca() : memref<4xf32>
    %rows = tensor.dim %x, %c1 : tensor<1x?x128xf32>
    %r = scf.for %i = %c0 to %rows step %c1 iter_args(%a = %init) -> (tensor<1x?x128xf32>) {
      %q = scf.for %j = %c0 to %c64 step %cs iter_args(%b = %a) -> (tensor<1x?x128xf32>) {
        %j0 = arith.addi %j, %c0 : index
        %j1 = arith.addi %j, %c1 : index
        %j2 = arith.addi %j, %c2 : index
        %j3 = arith.addi %j, %c3 : index
        %x0 = tensor.extract %x[%c0, %i, %j0] : tensor<1x?x128xf32>
        %x1 = tensor.extract %x[%c0, %i, %j1] : tensor<1x?x128xf32>
        %x2 = tensor.extract %x[%c0, %i, %j2] : tensor<1x?x128xf32>
        %x3 = tensor.extract %x[%c0, %i, %j3] : tensor<1x?x128xf32>
        func.call @__hexagon_arm_cos4(%t, %x0, %x1, %x2, %x3) : (memref<4xf32>, f32, f32, f32, f32) -> ()
        %y0 = memref.load %t[%c0] : memref<4xf32>
        %y1 = memref.load %t[%c1] : memref<4xf32>
        %y2 = memref.load %t[%c2] : memref<4xf32>
        %y3 = memref.load %t[%c3] : memref<4xf32>
        %k0 = arith.addi %j0, %c64 : index
        %k1 = arith.addi %j1, %c64 : index
        %k2 = arith.addi %j2, %c64 : index
        %k3 = arith.addi %j3, %c64 : index
        %b0 = tensor.insert %y0 into %b[%c0, %i, %j0] : tensor<1x?x128xf32>
        %b1 = tensor.insert %y0 into %b0[%c0, %i, %k0] : tensor<1x?x128xf32>
        %b2 = tensor.insert %y1 into %b1[%c0, %i, %j1] : tensor<1x?x128xf32>
        %b3 = tensor.insert %y1 into %b2[%c0, %i, %k1] : tensor<1x?x128xf32>
        %b4 = tensor.insert %y2 into %b3[%c0, %i, %j2] : tensor<1x?x128xf32>
        %b5 = tensor.insert %y2 into %b4[%c0, %i, %k2] : tensor<1x?x128xf32>
        %b6 = tensor.insert %y3 into %b5[%c0, %i, %j3] : tensor<1x?x128xf32>
        %b7 = tensor.insert %y3 into %b6[%c0, %i, %k3] : tensor<1x?x128xf32>
        scf.yield %b7 : tensor<1x?x128xf32>
      }
      scf.yield %q : tensor<1x?x128xf32>
    } {hexagon.fp32_arm_rope_trig4_batch}
    return %r : tensor<1x?x128xf32>
  }
}
'''

MASK = '''
#m3 = affine_map<(d0) -> (d0)>
#m2 = affine_map<(d0, d1) -> (d0, d1)>
#m4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#m5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#m6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#m7 = affine_map<(d0, d1, d2, d3) -> ()>
#m8 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
#m9 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
module {
  func.func @graph(%pos: tensor<?xi64>, %valid: tensor<1x?xi64>, %q: index, %kv: index) -> tensor<1x1x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant dense<true> : tensor<i1>
    %zero = arith.constant dense<0.000000e+00> : tensor<f32>
    %low = arith.constant dense<FILL> : tensor<f32>
    %width = tensor.dim %valid, %c1 : tensor<1x?xi64>
    %e2 = tensor.empty(%width) : tensor<1x?xi1>
    %3 = linalg.generic {indexing_maps = [#m2, #m2], iterator_types = ["parallel", "parallel"]} ins(%valid : tensor<1x?xi64>) outs(%e2 : tensor<1x?xi1>) {
    ^bb0(%in: i64, %out: i1):
      %r = arith.cmpi ne, %in, %c0_i64 : i64
      linalg.yield %r : i1
    } -> tensor<1x?xi1>
    %e5 = tensor.empty() : tensor<1xi64>
    %6 = linalg.generic {indexing_maps = [#m3], iterator_types = ["parallel"]} outs(%e5 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    %e8 = tensor.empty(%kv) : tensor<?xi64>
    %9 = linalg.generic {indexing_maps = [#m3], iterator_types = ["parallel"]} outs(%e8 : tensor<?xi64>) {
    ^bb0(%out: i64):
      %i = linalg.index 0 : index
      %r = arith.index_cast %i : index to i64
      linalg.yield %r : i64
    } -> tensor<?xi64>
    %x6 = tensor.expand_shape %6 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi64> into tensor<1x1x1x1xi64>
    %n = tensor.dim %pos, %c0 : tensor<?xi64>
    %xp = tensor.expand_shape %pos [[0, 1, 2, 3]] output_shape [1, 1, %n, 1] : tensor<?xi64> into tensor<1x1x?x1xi64>
    %x9 = tensor.expand_shape %9 [[0, 1, 2, 3]] output_shape [1, 1, 1, %kv] : tensor<?xi64> into tensor<1x1x1x?xi64>
    %e10 = tensor.empty(%n, %kv) : tensor<1x1x?x?xi1>
    %11 = linalg.generic {indexing_maps = [#m4, #m5, #m6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x9, %xp : tensor<1x1x1x?xi64>, tensor<1x1x?x1xi64>) outs(%e10 : tensor<1x1x?x?xi1>) {
    ^bb0(%a: i64, %b: i64, %out: i1):
      %r = arith.cmpi sle, %a, %b : i64
      linalg.yield %r : i1
    } -> tensor<1x1x?x?xi1>
    %12 = linalg.generic {indexing_maps = [#m7, #m6, #m6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%true, %11 : tensor<i1>, tensor<1x1x?x?xi1>) outs(%e10 : tensor<1x1x?x?xi1>) {
    ^bb0(%a: i1, %b: i1, %out: i1):
      %r = arith.andi %a, %b : i1
      linalg.yield %r : i1
    } -> tensor<1x1x?x?xi1>
    %e13 = tensor.empty(%kv) : tensor<1x1x1x?xi1>
    %14 = linalg.generic {indexing_maps = [#m8, #m9, #m6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x6, %x9 : tensor<1x1x1x1xi64>, tensor<1x1x1x?xi64>) outs(%e13 : tensor<1x1x1x?xi1>) {
    ^bb0(%a: i64, %b: i64, %out: i1):
      %ra = arith.index_cast %a : i64 to index
      %rb = arith.index_cast %b : i64 to index
      %v = tensor.extract %3[%ra, %rb] : tensor<1x?xi1>
      linalg.yield %v : i1
    } -> tensor<1x1x1x?xi1>
    %16 = linalg.generic {indexing_maps = [#m6, #m4, #m6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %14 : tensor<1x1x?x?xi1>, tensor<1x1x1x?xi1>) outs(%e10 : tensor<1x1x?x?xi1>) {
    ^bb0(%a: i1, %b: i1, %out: i1):
      %r = arith.andi %a, %b : i1
      linalg.yield %r : i1
    } -> tensor<1x1x?x?xi1>
    %e20 = tensor.empty(%q, %kv) : tensor<1x1x?x?xi1>
    %21 = linalg.generic {indexing_maps = [#m6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%e20 : tensor<1x1x?x?xi1>) {
    ^bb0(%out: i1):
      %i = linalg.index 2 : index
      %j = linalg.index 3 : index
      %one = arith.cmpi eq, %n, %c1 : index
      %ii = arith.select %one, %c0, %i : index
      %v = tensor.extract %16[%c0, %c0, %ii, %j] : tensor<1x1x?x?xi1>
      linalg.yield %v : i1
    } -> tensor<1x1x?x?xi1>
    %e22 = tensor.empty(%q, %kv) : tensor<1x1x?x?xf32>
    %23 = linalg.generic {indexing_maps = [#m6, #m7, #m7, #m6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %zero, %low : tensor<1x1x?x?xi1>, tensor<f32>, tensor<f32>) outs(%e22 : tensor<1x1x?x?xf32>) {
    ^bb0(%c: i1, %a: f32, %b: f32, %out: f32):
      %r = arith.select %c, %a, %b : f32
      linalg.yield %r : f32
    } -> tensor<1x1x?x?xf32>
    return %23 : tensor<1x1x?x?xf32>
  }
}
'''


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


def test_rope_becomes_one_parallel_call():
    source = ROPE.replace('RESULTS', 'tensor<1x2x?x128xf32>').replace('RETURNS', '%out : tensor<1x2x?x128xf32>')
    ir, log = run(source, '--qwen-fuse-parallel-rope=expected-count=1')
    assert 'call @qwen_rope_heads_v1(' in ir
    assert 'arith.negf' not in ir and 'tensor.concat' not in ir and 'linalg.transpose' not in ir
    assert 'fused parallel RoPE: 1' in log


def test_rope_is_left_unchanged_when_the_transpose_has_other_users():
    source = ROPE.replace('RESULTS', '(tensor<1x2x?x128xf32>, tensor<1x2x?x128xf32>)').replace(
        'RETURNS', '%out, %x : tensor<1x2x?x128xf32>, tensor<1x2x?x128xf32>')
    ir, log = run(source, '--qwen-fuse-parallel-rope')
    assert 'qwen_rope_heads_v1' not in ir and 'tensor.concat' in ir
    assert 'fused parallel RoPE: 0' in log


def test_trig_table_becomes_one_parallel_call():
    ir, log = run(TRIG.replace('STEP', '4'), '--qwen-fuse-parallel-trig=expected-count=1')
    assert 'call @qwen_arm_trig_rows_v1(' in ir and 'scf.for' not in ir
    assert 'fused parallel trig: 1' in log


def test_trig_table_with_another_step_is_left_unchanged():
    ir, log = run(TRIG.replace('STEP', '2'), '--qwen-fuse-parallel-trig')
    assert 'qwen_arm_trig_rows_v1' not in ir and 'call @__hexagon_arm_cos4(' in ir
    assert 'fused parallel trig: 0' in log


def test_attention_mask_becomes_one_parallel_call():
    ir, log = run(MASK.replace('FILL', '-3.40282347E+38'), '--qwen-fuse-causal-mask=expected-count=1')
    assert 'call @qwen_causal_mask_v1(' in ir  # dead producers are left to --canonicalize
    assert 'fused causal mask: 1' in log


def test_attention_mask_with_another_fill_is_left_unchanged():
    ir, log = run(MASK.replace('FILL', '-1.000000e+04'), '--qwen-fuse-causal-mask')
    assert 'qwen_causal_mask_v1' not in ir and 'arith.cmpi sle' in ir
    assert 'fused causal mask: 0' in log
