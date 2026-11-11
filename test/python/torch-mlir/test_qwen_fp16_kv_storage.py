"""Check the FP16 resident KV storage rewrite (--qwen-fp16-kv-storage).

Fixtures are the prepared forms after qwen-resident-kv and attention fusion:
a resident plane append feeding the fused attention call, and attention over
fresh in-graph K/V (initial domain). Any other plane use must be rejected.
"""
import os
import subprocess

import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')

ATTENTION = ('memref<1x16x1x128xf32, strided<[?, ?, ?, ?], offset: ?>>, '
             'memref<1x8x?x128xf32, strided<[?, ?, ?, ?], offset: ?>>, '
             'memref<1x8x?x128xf32, strided<[?, ?, ?, ?], offset: ?>>, '
             'memref<1x1x1x?xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<1x1x16x128xf32>')

PLANE = 'tensor<1x8x?x128xf32> {bufferization.buffer_layout = strided<[?, 128, 1024, 1]>, bufferization.writable = true, hexagon.kv_plane = INDEX : i64}'

RESIDENT = '''
module {
  func.func private @qwen_ordered_fp32_attention_v1(ATTENTION) -> i32 attributes {llvm.emit_c_interface}
  func.func @decode(%k: PLANE_K, %v: PLANE_V, %newk: tensor<1x8x1x128xf32>, %newv: tensor<1x8x1x128xf32>,
                    %q: tensor<1x16x1x128xf32>, %mask: tensor<1x1x1x?xf32>) -> (tensor<1x8x1x128xf32>, i32)
      attributes {hexagon.resident_kv} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %n = tensor.dim %k, %c2 : tensor<1x8x?x128xf32>
    %total = arith.addi %n, %c1 : index
    %stride = arith.muli %total, %c1024 : index
    %kb = bufferization.to_buffer %k : tensor<1x8x?x128xf32> to memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>>
    %kr = memref.reinterpret_cast %kb to offset: [0], sizes: [1, 8, %total, 128], strides: [%stride, 128, 1024, 1] : memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>> to memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>>
    %kt = memref.subview %kr[0, 0, %n, 0] [1, 8, 1, 128] [1, 1, 1, 1] : memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>> to memref<1x8x1x128xf32, strided<[?, 128, 1024, 1], offset: ?>>
    %ks = bufferization.to_buffer %newk read_only : tensor<1x8x1x128xf32> to memref<1x8x1x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    memref.copy %ks, %kt : memref<1x8x1x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x8x1x128xf32, strided<[?, 128, 1024, 1], offset: ?>>
    %vb = bufferization.to_buffer %v : tensor<1x8x?x128xf32> to memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>>
    %vr = memref.reinterpret_cast %vb to offset: [0], sizes: [1, 8, %total, 128], strides: [%stride, 128, 1024, 1] : memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>> to memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>>
    %vt = memref.subview %vr[0, 0, %n, 0] [1, 8, 1, 128] [1, 1, 1, 1] : memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>> to memref<1x8x1x128xf32, strided<[?, 128, 1024, 1], offset: ?>>
    %vs = bufferization.to_buffer %newv read_only : tensor<1x8x1x128xf32> to memref<1x8x1x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    memref.copy %vs, %vt : memref<1x8x1x128xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x8x1x128xf32, strided<[?, 128, 1024, 1], offset: ?>>
    %qb = bufferization.to_buffer %q read_only : tensor<1x16x1x128xf32> to memref<1x16x1x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %kc = memref.cast %kr : memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>> to memref<1x8x?x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %vc = memref.cast %vr : memref<1x8x?x128xf32, strided<[?, 128, 1024, 1]>> to memref<1x8x?x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %mb = bufferization.to_buffer %mask read_only : tensor<1x1x1x?xf32> to memref<1x1x1x?xf32, strided<[?, ?, ?, ?], offset: ?>>
    %out = memref.alloc() : memref<1x1x16x128xf32>
    %status = call @qwen_ordered_fp32_attention_v1(%qb, %kc, %vc, %mb, %out) : (ATTENTION) -> i32
    EXTRA
    return %newk, %status : tensor<1x8x1x128xf32>, i32
  }
}
'''.replace('ATTENTION', ATTENTION).replace('PLANE_K', PLANE.replace('INDEX', '0')).replace(
    'PLANE_V', PLANE.replace('INDEX', '1'))

FRESH = '''
module {
  func.func private @qwen_ordered_fp32_attention_v1(ATTENTION) -> i32 attributes {llvm.emit_c_interface}
  func.func @initial(%k: tensor<1x8x?x128xf32>, %v: tensor<1x8x?x128xf32>, %q: tensor<1x16x1x128xf32>,
                     %mask: tensor<1x1x1x?xf32>) -> i32 {
    %qb = bufferization.to_buffer %q read_only : tensor<1x16x1x128xf32> to memref<1x16x1x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %kb = bufferization.to_buffer %k read_only : tensor<1x8x?x128xf32> to memref<1x8x?x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %vb = bufferization.to_buffer %v read_only : tensor<1x8x?x128xf32> to memref<1x8x?x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %mb = bufferization.to_buffer %mask read_only : tensor<1x1x1x?xf32> to memref<1x1x1x?xf32, strided<[?, ?, ?, ?], offset: ?>>
    %out = memref.alloc() : memref<1x1x16x128xf32>
    %status = call @qwen_ordered_fp32_attention_v1(%qb, %kb, %vb, %mb, %out) : (ATTENTION) -> i32
    return %status : i32
  }
}
'''.replace('ATTENTION', ATTENTION)


def run(source, flag):
    if not OPT:
        pytest.skip('set QWEN_PACKED_I8_OPT to the built qwen-packed-i8-opt')
    return subprocess.run([OPT, '-', flag, '--verify-each'], input=source, text=True,
                          capture_output=True, timeout=60)


def test_resident_planes_become_fp16_and_feed_fp16_attention():
    result = run(RESIDENT.replace('EXTRA', ''), '--qwen-fp16-kv-storage=expected-planes=2 expected-calls=1')
    assert result.returncode == 0, result.stderr
    ir = result.stdout
    assert 'tensor<1x8x?x128xf32> {bufferization' not in ir
    assert ir.count('tensor<1x8x?x128xf16> {bufferization.buffer_layout') == 2
    assert ir.count('arith.truncf') == 2 and 'memref<1x8x1x128xf16' in ir
    assert 'call @qwen_ordered_fp16kv_attention_v1(' in ir
    assert 'qwen_ordered_fp32_attention_v1' not in ir
    # Returned deltas stay FP32; the session rounds them when it owns the copy.
    assert '-> (tensor<1x8x1x128xf32>, i32)' in ir
    assert 'planes: 2 attention calls: 1' in result.stderr


def test_fresh_attention_operands_are_rounded_by_one_generic():
    result = run(FRESH, '--qwen-fp16-kv-storage=expected-calls=1')
    assert result.returncode == 0, result.stderr
    ir = result.stdout
    # One two-result generic: separate generics let the pipeline's empty-tensor
    # CSE give K and V one buffer, so V's conversion overwrote K after to_buffer.
    assert ir.count('linalg.generic') == 1 and ir.count('arith.truncf') == 2
    assert '-> (tensor<1x8x?x128xf16>, tensor<1x8x?x128xf16>)' in ir
    assert ir.count('bufferization.to_buffer') == 4  # Q, mask and the two FP16 views
    assert 'call @qwen_ordered_fp16kv_attention_v1(' in ir
    assert 'planes: 0 attention calls: 1' in result.stderr


def test_other_plane_use_is_rejected():
    extra = '%peek = tensor.extract_slice %k[0, 0, 0, 0] [1, 8, 1, 128] [1, 1, 1, 1] : tensor<1x8x?x128xf32> to tensor<1x8x1x128xf32>'
    result = run(RESIDENT.replace('EXTRA', extra), '--qwen-fp16-kv-storage')
    assert result.returncode != 0 and 'FP16 KV: unsupported plane use' in result.stderr


def test_unexpected_count_is_rejected():
    result = run(RESIDENT.replace('EXTRA', ''), '--qwen-fp16-kv-storage=expected-planes=4')
    assert result.returncode != 0 and 'expected 4' in result.stderr
