"""Validate the packed byte permutation and reject nonconstant weights."""
import os
import re
import subprocess
import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')
pytestmark = pytest.mark.skipif(not OPT, reason='QWEN_PACKED_I8_OPT required')


def fixture(dynamic=False):
    data = bytes((i * 37 + i // 11) % 256 for i in range(128 * 8))
    dt = 'memref<?x?xi8, strided<[?, ?], offset: ?>>'
    ct = 'memref<?x?xi32, strided<[?, ?], offset: ?>>'
    weight = '%warg' if dynamic else '%w'
    const = '' if dynamic else f'%w = arith.constant dense<"0x{data.hex()}"> : tensor<128x8xi8>'
    text = f'''module {{
      func.func private @__qwen_i8_multi_matmul({dt}, {dt}, {ct})
      func.func @test(%a: {dt}, %c: {ct}, %warg: tensor<128x8xi8>) {{
        {const}
        %empty = tensor.empty() : tensor<8x128xi8>
        %t = linalg.transpose ins({weight} : tensor<128x8xi8>) outs(%empty : tensor<8x128xi8>) permutation = [1, 0]
        %b = bufferization.to_buffer %t read_only : tensor<8x128xi8> to memref<8x128xi8>
        %cast = memref.cast %b : memref<8x128xi8> to {dt}
        func.call @__qwen_i8_multi_matmul(%a, %cast, %c) : ({dt}, {dt}, {ct}) -> ()
        return
      }}
    }}'''
    return data, text


def run(text):
    return subprocess.run([OPT, '-', '--hexagon-prepack-i8-weights=expected-count=1',
                           '--canonicalize', '--verify-each'], input=text,
                          text=True, capture_output=True, timeout=30)


def test_exact_byte_permutation():
    original, text = fixture()
    result = run(text)
    assert result.returncode == 0, result.stderr
    assert 'call @__hexagon_i8_prepacked_v1_matmul' in result.stdout
    assert 'linalg.transpose' not in result.stdout
    match = re.search(r'dense<"0x([0-9A-Fa-f]+)"> : tensor<8x128xi8>', result.stdout)
    assert match, result.stdout
    packed = bytes.fromhex(match[1])
    # Independent inverse: every logical B[k,n] must retain its signed byte.
    for n in range(128):
        for k in range(8):
            assert packed[(k // 4) * 512 + n * 4 + k % 4] == original[n * 8 + k]


def test_mutable_weight_rejected():
    _, text = fixture(dynamic=True)
    result = run(text)
    assert result.returncode != 0
    assert 'expected 1, got 0' in result.stderr
