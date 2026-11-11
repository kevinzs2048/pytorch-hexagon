"""Check that one unknown-origin buffer does not force quadratic deallocation."""
import os
from pathlib import Path
import subprocess

import pytest

OPT = os.environ.get('QWEN_LINALG_HEXAGON_OPT')


def lower():
    if not OPT:
        pytest.skip('set QWEN_LINALG_HEXAGON_OPT to the pinned linalg-hexagon-opt')
    source = Path(__file__).with_name('qwen_dealloc_unknown_origin.mlir')
    result = subprocess.run([OPT, str(source), '--linalg-to-llvm=fusion=false'],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_private_call_result_does_not_select_generic_dealloc_helper():
    ir = lower()
    # The generic helper compares every released buffer with every other one.
    assert 'dealloc_helper' not in ir
    # Each graph temporary and the call result are still released.
    assert ir.count('llvm.call @free') >= 3
