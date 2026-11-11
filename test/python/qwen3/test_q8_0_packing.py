"""Q8_0 tile packing and the numpy references of runtime/q8_0/q8_0_hmx.cpp."""
import os
import sys

import numpy as np
import pytest

from models.qwen3.q8_0 import quantize as qz
from models.qwen3.q8_0.weights import unpack_prepacked_int8


def blocks(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(-127, 128, size=(n, k // 32, 32), dtype=np.int8),
            (rng.random((n, k // 32)) * 0.01 + 1e-4).astype(np.float16))


@pytest.mark.parametrize("vrmpy", [False, True])
def test_pack_round_trip(vrmpy):
    q, d = blocks(64, 96)
    blob = (qz.pack_vrmpy if vrmpy else qz.pack)(q, d)
    assert blob.size == 64 // 32 * 96 // 32 * qz.TILE_BYTES
    q2, d2 = qz.unpack(blob, 64, 96, vrmpy=vrmpy)
    assert np.array_equal(q, q2) and np.array_equal(d, d2)


def test_pack_layouts_by_element():
    q, d = blocks(64, 64, seed=1)
    pair = qz.pack(q, d).reshape(2, 2, qz.TILE_BYTES)
    vrmpy = qz.pack_vrmpy(q, d).reshape(2, 2, qz.TILE_BYTES)
    nt, kt = 1, 1
    for c in (0, 7, 31):
        for k in (0, 5, 30, 31):
            value = q[nt * 32 + c, kt, k].view(np.uint8)
            assert pair[nt, kt, (k // 2) * 64 + c * 2 + k % 2] == value          # [p][c][r]
            assert vrmpy[nt, kt, (k // 4) * 128 + c * 4 + k % 4] == value        # [g][c][j]
        scales = pair[nt, kt, 1024:].view(np.float16)
        assert scales[2 * c] == d[nt * 32 + c, kt] and scales[2 * c + 1] == d[nt * 32 + c, kt]


def test_activation_quantization_against_hand_computed_values():
    # Block 0: amax 127, so d = 1 and x * (1 / d) = x exactly: ties at .5 must
    # round half to even (llama.cpp NEON vcvtnq), not away from zero.
    ties = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -126.5], np.float32)
    block0 = np.concatenate([ties, np.full(23, 3.0, np.float32), [127.0]]).astype(np.float32)
    # Block 1: all zero -> d = 0, q = 0. Block 2: amax 12.7 -> d = fp32(12.7 / 127).
    block2 = np.zeros(32, np.float32); block2[0] = 12.7; block2[1] = -6.35
    q, d = qz.quantize_activation(np.concatenate([block0, np.zeros(32, np.float32), block2]))
    assert q[0, :8].tolist() == [0, 2, 2, 0, -2, -2, 126, -126]
    assert q[0, 8:31].tolist() == [3] * 23 and q[0, 31] == 127 and d[0] == np.float16(1.0)
    assert not q[1].any() and d[1] == 0
    d2 = np.float32(np.float32(12.7) / np.float32(127))
    assert d[2] == np.float16(d2) and q[2, 0] == 127 and q[2, 1] == -64  # -6.35 / 0.1 = -63.5 -> -64 (even)


def test_q8_q8_row_is_close_to_the_dequantized_product():
    q, d = blocks(64, 128, seed=3)
    x = np.random.default_rng(4).standard_normal(128).astype(np.float32)
    y = qz.q8_q8_matvec(x, q, d)
    exact = (q.astype(np.float32) * d.astype(np.float32)[..., None]).reshape(64, 128) @ x
    assert y.dtype == np.float32 and np.abs(y - exact).max() <= 0.02 * np.abs(exact).max()


def test_prepacked_int8_unpack_inverts_the_prepack_formula():
    n, k = 256, 8
    q = np.random.default_rng(5).integers(-127, 128, size=(n, k), dtype=np.int8)
    packed = np.empty(n * k, np.int8)
    for c in range(0, n, 128):
        for r in range(0, k, 4):
            for j in range(128):
                for t in range(4):
                    packed[(c // 128) * k * 128 + (r // 4) * 512 + j * 4 + t] = q[c + j, r + t]
    assert np.array_equal(unpack_prepacked_int8(packed, n, k), q)


@pytest.mark.skipif(not os.environ.get("LLAMA_GGUF_PY"), reason="LLAMA_GGUF_PY (llama.cpp gguf-py) not set")
def test_gguf_py_blocks_dequantize_to_the_weight():
    sys.path.insert(0, os.environ["LLAMA_GGUF_PY"])
    import gguf
    w = (np.random.default_rng(6).standard_normal((32, 64)) * 0.05).astype(np.float32)
    raw = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q8_0)
    q, d = qz.gguf_blocks(raw, 32, 64)
    approx = q.astype(np.float32) * d.astype(np.float32)[..., None]
    assert np.abs(approx.reshape(32, 64) - w).max() <= np.abs(w).max() / 127
