"""Fuse only the proven zero-point-zero symmetric INT8 quantization chain."""
import os
from pathlib import Path
import subprocess
import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')
pytestmark = pytest.mark.skipif(not OPT, reason='QWEN_PACKED_I8_OPT required')
SOURCE = Path(__file__).resolve().parents[3] / 'models/qwen3/native/policy/test/bounded-int8-quant.mlir'


def run(text):
    return subprocess.run([OPT, '-', '--qwen-vectorize-activation=expected-count=1',
                           '--canonicalize', '--cse', '--verify-each'], input=text,
                          text=True, capture_output=True, timeout=30)


def test_exact_chain_fuses_and_preserves_scale_check():
    result = run(SOURCE.read_text())
    assert result.returncode == 0, result.stderr
    assert 'qwen bounded quant fusion: 1' in result.stderr
    assert 'arith.fptosi' not in result.stdout
    assert result.stdout.count('cf.assert') == result.stdout.count('arith.divf') == 1
    assert 'arith.minui' in result.stdout and 'arith.trunci' in result.stdout


@pytest.mark.parametrize('old,new', [
    ('dense<0.0>', 'dense<1.0>'),
    ('constant -127.0', 'constant -128.0'),
    ('constant 127.0', 'constant 126.0'),
    ('cmpf ult', 'cmpf olt'),
    ('cmpf ugt', 'cmpf ogt'),
    ('f32 to i8', 'f32 to i16'),
])
def test_other_contracts_are_not_fused(old, new):
    text = SOURCE.read_text().replace(old, new)
    if 'i16' in new:
        text = text.replace('xi8', 'xi16').replace('%unused:i8', '%unused:i16').replace('%v:i8','%v:i16')
    result = run(text)
    assert result.returncode == 0, result.stderr
    assert 'qwen bounded quant fusion: 0' in result.stderr
    assert 'arith.fptosi' in result.stdout
