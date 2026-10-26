"""Reproduce late clone allocation and reject naive post-deallocation hoisting.

This diagnostic is not evidence that every Qwen allocation has this origin.
The executable pass ordering is useful when investigating the accepted q1
trace: allocation reuse must move or recompute ownership and deallocation too.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


IR = """module {
  func.func @late_clone(%input: memref<128xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0.0 : f32
    %result = scf.for %k = %c0 to %c1024 step %c1
        iter_args(%sum = %zero) -> f32 {
      %copy = bufferization.clone %input
          : memref<128xf32> to memref<128xf32>
      %x = memref.load %copy[%c0] : memref<128xf32>
      %add = arith.addf %sum, %x : f32
      memref.dealloc %copy : memref<128xf32>
      scf.yield %add : f32
    }
    return %result : f32
  }
}
"""


def test_late_clone_requires_ownership_safe_reuse():
    opt = os.environ.get('QWEN_MLIR_OPT') or shutil.which('mlir-opt')
    if not opt:
        pytest.skip('set QWEN_MLIR_OPT to the pinned compiler mlir-opt')
    assert Path(opt).is_file()

    def lower(pipeline):
        result = subprocess.run(
            [opt, '-', '--verify-each', '--pass-pipeline=' + pipeline],
            input=IR, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('memref.alloc()') == 1
        assert result.stdout.count('memref.dealloc') == 1
        assert result.stdout.count('arith.addf') == 1
        return result.stdout

    current_order = lower(
        'builtin.module(func.func(buffer-loop-hoisting),'
        'convert-bufferization-to-memref)'
    )
    assert current_order.index('scf.for') < current_order.index('memref.alloc()')

    naive_late_hoisting = lower(
        'builtin.module(convert-bufferization-to-memref,'
        'func.func(buffer-loop-hoisting))'
    )
    assert naive_late_hoisting.index('memref.alloc()') < naive_late_hoisting.index('scf.for')
    # This is a counterexample, not an admissible optimization: the reused
    # allocation is still freed on each iteration. A production fix must
    # validate ownership and both allocation/deallocation placement together.
    assert naive_late_hoisting.index('scf.for') < naive_late_hoisting.index('memref.dealloc')
    assert naive_late_hoisting.index('memref.dealloc') < naive_late_hoisting.index('scf.yield')
