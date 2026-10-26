"""Check the actual QK copy origin and preserve conflicting tensor uses."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from qwen3.compiler_dependency import compiler_root


def lower(source, cse):
    opt = os.environ.get('QWEN_MLIR_OPT') or shutil.which('mlir-opt')
    if not opt:
        pytest.skip('set QWEN_MLIR_OPT to the pinned compiler mlir-opt')
    command = [opt, '-', '--verify-each']
    if cse:
        command += ['--cse']
    command += [
        '--one-shot-bufferize=bufferize-function-boundaries '
        'allow-return-allocs-from-loops '
        'function-boundary-type-conversion=identity-layout-map',
        '--cse', '--canonicalize',
    ]
    result = subprocess.run(command, input=source, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout


def fixture():
    return (compiler_root() / 'test/python/torch-mlir/qwen3_qk_bufferization.mlir').read_text()


def test_duplicate_slices_cause_per_tile_copy():
    before, after = lower(fixture(), False), lower(fixture(), True)
    assert 'memref.alloc(%' in before
    assert ': memref<?xf32>' in before
    assert 'memref.alloc(%' not in after
    assert 'memref.copy' not in after
    for ir in (before, after):
        assert ir.count('math.fma') == 1
        assert ir.count('scf.for') == 3
        assert 'step %c128' in ir
        assert 'fastmath' not in ir


def test_old_scores_are_retained_when_the_caller_reads_them():
    source = fixture().replace(
        ') -> tensor<16x?xf32> {',
        ') -> (tensor<16x?xf32>, tensor<16x?xf32>) {', 1).replace(
        'return %result : tensor<16x?xf32>',
        'return %result, %scores : tensor<16x?xf32>, tensor<16x?xf32>')
    after = lower(source, True)
    # The old tensor must remain observable. One-Shot Bufferize retains a
    # whole-score copy before entering the loops, while eliminating tile copies.
    assert 'memref.alloc(%' in after
    assert 'memref.copy' in after
    assert after.index('memref.copy') < after.index('scf.for')
    assert ': memref<?xf32>' not in after
    assert after.count('math.fma') == 1
