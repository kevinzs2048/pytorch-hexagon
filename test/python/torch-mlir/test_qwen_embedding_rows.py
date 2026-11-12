"""Check the embedding row copy (--qwen-embedding-rows).

The fixture is the captured token embedding gather: out[0, t, j] =
table[ids[0, t], j] with the two bounds asserts per element. It becomes one
call that checks each id once and copies whole rows.
"""
import os
import subprocess

import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')

SOURCE = '''
#ids = affine_map<(d0, d1, d2) -> (d0, d1)>
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @embed(%ids: tensor<1x?xi64>) -> tensor<1x?x8xf32> {
    %table = arith.constant dense<1.5> : tensor<16x8xf32>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0_i64 = arith.constant 0 : i64
    %n = tensor.dim %ids, %c1 : tensor<1x?xi64>
    %init = tensor.empty(%n) : tensor<1x?x8xf32>
    %out = linalg.generic {indexing_maps = [#ids, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%ids : tensor<1x?xi64>) outs(%init : tensor<1x?x8xf32>) {
    ^bb0(%in: i64, %o: f32):
      %row = arith.index_cast %in : i64 to index
      %col = linalg.index 2 : index
      %lt = arith.cmpi slt, %row, %c16 : index
      cf.assert %lt, "index must be smaller than dim size"
      %ge = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %ge, "index must be larger or equal to 0"
      %v = tensor.extract %table[%row, COLUMN] : tensor<16x8xf32>
      linalg.yield %v : f32
    } -> tensor<1x?x8xf32>
    return %out : tensor<1x?x8xf32>
  }
}
'''


def run(source, flag):
    if not OPT:
        pytest.skip('set QWEN_PACKED_I8_OPT to the built qwen-packed-i8-opt')
    return subprocess.run([OPT, '-', flag, '--verify-each'], input=source, text=True,
                          capture_output=True, timeout=60)


def test_gather_becomes_one_row_copy_call():
    result = run(SOURCE.replace('COLUMN', '%col'), '--qwen-embedding-rows=expected-count=1')
    assert result.returncode == 0, result.stderr
    ir = result.stdout
    assert 'linalg.generic' not in ir and 'cf.assert' not in ir
    assert ir.count('call @qwen_embedding_rows_v1(') == 1
    assert ('func.func private @qwen_embedding_rows_v1(memref<?x?xi64, strided<[?, ?], offset: ?>>, '
            'memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) '
            'attributes {llvm.emit_c_interface}') in ir
    assert 'embedding rows: 1' in result.stderr


def test_other_gather_is_kept():
    # Reading a different column than the output column is not a row copy.
    result = run(SOURCE.replace('COLUMN', '%c1'), '--qwen-embedding-rows')
    assert result.returncode == 0, result.stderr
    assert 'qwen_embedding_rows_v1' not in result.stdout and 'embedding rows: 0' in result.stderr


def test_unexpected_count_is_rejected():
    result = run(SOURCE.replace('COLUMN', '%c1'), '--qwen-embedding-rows=expected-count=1')
    assert result.returncode != 0 and 'expected 1, got 0' in result.stderr
