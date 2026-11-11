"""Check the Q8_0 projection rewrite (--qwen-q8-0-projections).

Each fixture is one fused INT8 projection in its prepared form after
--qwen-fuse-i8-dequant: the per-token activation quantization of x
(x * (1 / S) -> i8, S guarded by torch-mlir zero checks) feeding
__hexagon_i8_prepacked_v1_matmul_dequant[_residual]. The rewrite must call the
Q8_0 runtime on x, remove the quantization and its zero checks, and dump the
INT8 weight and column scale bytes used to identify the Q8_0 weight offline.
"""
import json
import os
import subprocess

import numpy as np
import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')
I8 = 'memref<?x?xi8, strided<[?, ?], offset: ?>>'
F2 = 'memref<?x?xf32, strided<[?, ?], offset: ?>>'
F1 = 'memref<?xf32, strided<[?], offset: ?>>'


def fixture(n, k, rows='?', mode='plain', factor='inverse', extra_scale_use=False, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.integers(-127, 128, size=(k, n), dtype=np.int8)
    column = rng.random(n, dtype=np.float32) + 0.5
    suffix = {'plain': '', 'residual': '_residual'}[mode]
    operands = [I8, I8, F2, F1] + ([F2] if mode == 'residual' else []) + [F2]
    groups = '[[0, 1], [2]]' if rows == '?' else '[[0], [1, 2]]'
    flat = f'tensor<?x{k}xi8>' if rows == '?' else f'tensor<1x{k}xi8>'
    flat_static = f'memref<?x{k}xi8>' if rows == '?' else f'memref<1x{k}xi8>'
    out = f'memref<?x{n}xf32>' if rows == '?' else f'memref<1x{n}xf32>'
    alloc = 'memref.alloc(%m)' if rows == '?' else 'memref.alloc()'
    empty = '(%m)' if rows == '?' else '()'
    inverse = ('%r = arith.divf %one, %a : f32' if factor == 'inverse' else '%r = arith.addf %one, %a : f32')
    residual = (f'''
    %rb = bufferization.to_buffer %res read_only : tensor<{rows}x{n}xf32> to memref<{rows}x{n}xf32>
    %rc = memref.cast %rb : memref<{rows}x{n}xf32> to {F2}''' if mode == 'residual' else '')
    call_args = '%qc, %wc, %sc2, %cc' + (', %rc' if mode == 'residual' else '') + ', %oc'
    extra = (f'''
    %keep = tensor.extract %s[%c0, %c0, %c0] : tensor<1x{rows}x1xf32>
    "test.use"(%keep) : (f32) -> ()''' if extra_scale_use else '')
    ir = f'''
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#bc = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module {{
  func.func private @__hexagon_i8_prepacked_v1_matmul_dequant{suffix}({", ".join(operands)}) attributes {{llvm.emit_c_interface}}
  func.func @graph(%x: tensor<1x{rows}x{k}xf32>, %s: tensor<1x{rows}x1xf32>, %rs: tensor<{rows}x1xf32>,
                   %res: tensor<{rows}x{n}xf32>) -> tensor<{rows}x{n}xf32> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %m = tensor.dim %x, %c1 : tensor<1x{rows}x{k}xf32>
    %ms = tensor.dim %s, %c1 : tensor<1x{rows}x1xf32>
    scf.for %i = %c0 to %ms step %c1 {{
      %v = tensor.extract %s[%c0, %i, %c0] : tensor<1x{rows}x1xf32>
      %ne = arith.cmpf one, %v, %zero : f32
      cf.assert %ne, "unimplemented: tensor with zero element"
    }}{extra}
    %e1 = tensor.empty{empty} : tensor<1x{rows}x1xf32>
    %inv = linalg.generic {{indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}}
        ins(%s : tensor<1x{rows}x1xf32>) outs(%e1 : tensor<1x{rows}x1xf32>) {{
    ^bb0(%a: f32, %o: f32):
      {inverse}
      linalg.yield %r : f32
    }} -> tensor<1x{rows}x1xf32>
    %e2 = tensor.empty{empty} : tensor<1x{rows}x{k}xf32>
    %scaled = linalg.generic {{indexing_maps = [#id3, #bc, #id3], iterator_types = ["parallel", "parallel", "parallel"]}}
        ins(%x, %inv : tensor<1x{rows}x{k}xf32>, tensor<1x{rows}x1xf32>) outs(%e2 : tensor<1x{rows}x{k}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %r = arith.mulf %a, %b : f32
      linalg.yield %r : f32
    }} -> tensor<1x{rows}x{k}xf32>
    %e3 = tensor.empty{empty} : tensor<1x{rows}x{k}xi8>
    %q = linalg.generic {{indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}}
        ins(%scaled : tensor<1x{rows}x{k}xf32>) outs(%e3 : tensor<1x{rows}x{k}xi8>) {{
    ^bb0(%a: f32, %o: i8):
      %i = arith.fptosi %a : f32 to i32
      %t = arith.trunci %i : i32 to i8
      linalg.yield %t : i8
    }} -> tensor<1x{rows}x{k}xi8>
    %flat = tensor.collapse_shape %q {groups} : tensor<1x{rows}x{k}xi8> into {flat}
    %qb = bufferization.to_buffer %flat read_only : {flat} to {flat_static}
    %qc = memref.cast %qb : {flat_static} to {I8}
    %w = arith.constant dense<"0x{weight.tobytes().hex().upper()}"> : tensor<{k}x{n}xi8>
    %wb = bufferization.to_buffer %w read_only : tensor<{k}x{n}xi8> to memref<{k}x{n}xi8>
    %wc = memref.cast %wb : memref<{k}x{n}xi8> to {I8}
    %sb = bufferization.to_buffer %rs read_only : tensor<{rows}x1xf32> to memref<{rows}x1xf32>
    %sc2 = memref.cast %sb : memref<{rows}x1xf32> to {F2}
    %col = arith.constant dense<"0x{column.tobytes().hex().upper()}"> : tensor<{n}xf32>
    %cb = bufferization.to_buffer %col read_only : tensor<{n}xf32> to memref<{n}xf32>
    %cc = memref.cast %cb : memref<{n}xf32> to {F1}{residual}
    %ob = {alloc} : {out}
    %oc = memref.cast %ob : {out} to {F2}
    call @__hexagon_i8_prepacked_v1_matmul_dequant{suffix}({call_args}) : ({", ".join(operands)}) -> ()
    %t = bufferization.to_tensor %ob restrict writable : {out} to tensor<{rows}x{n}xf32>
    return %t : tensor<{rows}x{n}xf32>
  }}
}}
'''
    return ir, weight, column


def run(source, options, tmp_path):
    if not OPT:
        pytest.skip('QWEN_PACKED_I8_OPT is not set')
    dump = tmp_path / 'dump'
    dump.mkdir()
    flag = f'--qwen-q8-0-projections={options} dump-dir={dump}'
    result = subprocess.run([OPT, '-', '--allow-unregistered-dialect', flag, '--canonicalize', '--verify-each'],
                            input=source, capture_output=True, text=True)
    return result, dump


def test_projection_calls_q8_0_runtime_on_the_fp32_activation(tmp_path):
    ir, weight, column = fixture(64, 32)
    result, dump = run(ir, 'expected-count=1', tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert 'call @__hexagon_q8_0_matmul(' in out and 'func.func private @__hexagon_q8_0_matmul(' in out
    assert 'i8_prepacked' not in out and 'arith.trunci' not in out
    assert 'cf.assert' not in out and 'scf.for' not in out
    assert 'Q8_0 projections: 1, activation quantizations removed: 1' in result.stderr
    assert json.loads((dump / 'manifest.json').read_text()) == [{'id': 0, 'n': 64, 'k': 32, 'mode': 'plain'}]
    assert (dump / '0.w').read_bytes() == weight.tobytes()
    assert (dump / '0.s').read_bytes() == column.tobytes()


def test_residual_projection_keeps_the_residual_operand(tmp_path):
    ir, _, _ = fixture(64, 32, mode='residual')
    result, dump = run(ir, 'expected-count=1', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'call @__hexagon_q8_0_matmul_residual(' in result.stdout
    assert json.loads((dump / 'manifest.json').read_text())[0]['mode'] == 'residual'


def test_single_row_projection_uses_the_q8_x_q8_row_kernel(tmp_path):
    ir, _, _ = fixture(64, 32, rows='1')
    result, _ = run(ir, 'expected-count=1 single-row=true', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'call @__hexagon_q8_0_matvec_q8(' in result.stdout and '__hexagon_q8_0_matmul' not in result.stdout


def test_single_row_option_rejects_dynamic_rows(tmp_path):
    ir, _, _ = fixture(64, 32)
    result, _ = run(ir, 'single-row=true', tmp_path)
    assert result.returncode != 0 and 'needs one static row' in result.stderr


def test_lm_head_uses_the_q8_x_q8_row_kernel(tmp_path):
    ir, _, _ = fixture(3104, 32, rows='1')
    result, _ = run(ir, 'expected-count=1', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'call @__hexagon_q8_0_matvec_q8(' in result.stdout


def test_other_activation_scaling_is_rejected(tmp_path):
    ir, _, _ = fixture(64, 32, factor='other')
    result, _ = run(ir, 'expected-count=1', tmp_path)
    assert result.returncode != 0 and 'unmatched Q8_0 projection' in result.stderr


def test_zero_check_of_a_scale_with_other_users_is_kept_and_reported(tmp_path):
    ir, _, _ = fixture(64, 32, extra_scale_use=True)
    result, _ = run(ir, 'expected-count=1', tmp_path)
    assert result.returncode != 0 and 'dead scale checks erased' in result.stderr


def test_coverage_mismatch_fails(tmp_path):
    ir, _, _ = fixture(64, 32)
    result, _ = run(ir, 'expected-count=2', tmp_path)
    assert result.returncode != 0 and 'expected 2, got 1' in result.stderr
