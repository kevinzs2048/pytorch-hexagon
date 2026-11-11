# SPDX-License-Identifier: BSD-3-Clause
"""Q8_0 tile packing for runtime/q8_0/q8_0_hmx.cpp and its numpy references.

Packed tile (1152 bytes) for column tile nt (32 outputs) and reduction tile kt
(32 inputs): 1024 int8 values, then the 32 FP16 block scales expanded to the
same 64 lanes (lane 2c + r holds d[c]). Value order:
  pack        (HMX row-pair layout)  [p][c][r] = q[c][32 kt + 2p + r]
  pack_vrmpy  (LM head, HVX vrmpy)   [g][c][j] = q[c][32 kt + 4g + j]
Tiles are column-tile major, reduction tiles contiguous.
"""
import numpy as np

QK = 32
TILE_BYTES = 1024 + 128


def _scales(d, nt, kt_count):
    ds = d[nt * 32:(nt + 1) * 32]  # [c, kt]
    return np.repeat(ds.T[:, :, None], 2, axis=2).reshape(kt_count, 64).astype(np.float16)


def pack(q, d):
    """q int8 [N, K/32, 32], d fp16 [N, K/32] -> HMX row-pair tiles (uint8)."""
    n, kt_count, _ = q.shape
    out = np.zeros((n // 32, kt_count, TILE_BYTES), np.uint8)
    for nt in range(n // 32):
        v = q[nt * 32:(nt + 1) * 32].reshape(32, kt_count, 16, 2).transpose(1, 2, 0, 3)
        out[nt, :, :1024] = v.reshape(kt_count, 1024).view(np.uint8)
        out[nt, :, 1024:] = _scales(d, nt, kt_count).view(np.uint8).reshape(kt_count, 128)
    return out.reshape(-1)


def pack_vrmpy(q, d):
    """q int8 [N, K/32, 32], d fp16 [N, K/32] -> HVX vrmpy tiles (uint8)."""
    n, kt_count, _ = q.shape
    out = np.zeros((n // 32, kt_count, TILE_BYTES), np.uint8)
    for nt in range(n // 32):
        v = q[nt * 32:(nt + 1) * 32].reshape(32, kt_count, 8, 4).transpose(1, 2, 0, 3)
        out[nt, :, :1024] = v.reshape(kt_count, 1024).view(np.uint8)
        out[nt, :, 1024:] = _scales(d, nt, kt_count).view(np.uint8).reshape(kt_count, 128)
    return out.reshape(-1)


def unpack(blob, n, k, vrmpy=False):
    """Inverse of pack / pack_vrmpy -> q int8 [N, K/32, 32], d fp16 [N, K/32]."""
    t = np.asarray(blob, np.uint8).reshape(n // 32, k // 32, TILE_BYTES)
    shape = (8, 32, 4) if vrmpy else (16, 32, 2)
    q = t[..., :1024].view(np.int8).reshape(n // 32, k // 32, *shape).transpose(0, 3, 1, 2, 4).reshape(n, k // 32, 32)
    d = t[..., 1024:].view(np.float16).reshape(n // 32, k // 32, 32, 2)[..., 0].transpose(0, 2, 1).reshape(n, k // 32)
    return q, d


def gguf_blocks(raw, n, k):
    """gguf-py Q8_0 bytes (34-byte blocks: fp16 d, 32 int8) -> q int8 [N, K/32, 32], d fp16 [N, K/32]."""
    b = np.asarray(raw).reshape(-1).view(np.uint8).reshape(n, k // QK, 34)
    return b[:, :, 2:].copy().view(np.int8), b[:, :, :2].copy().view(np.float16)[..., 0]


def dequant(q, d):
    """FP16 weights [N, K] as the HMX path dequantizes them: fp16(fp16(q) * d)."""
    n, kt, _ = q.shape
    return (q.astype(np.float16) * d[..., None]).astype(np.float16).reshape(n, kt * QK)


def hmx_matmul(x, q, d):
    """HMX path reference: fp16(fp16(x) @ dequant(q, d)^T) as FP32 (FP16 accumulator read)."""
    w = dequant(q, d).astype(np.float32)
    return (x.astype(np.float16).astype(np.float32) @ w.T).astype(np.float16).astype(np.float32)


def quantize_activation(x):
    """llama.cpp quantize_row_q8_0: d = amax / 127 (FP32), q = round-half-even(x * (1 / d)),
    stored as fp16(d). x [K] float32 -> q int8 [K/32, 32], d fp16 [K/32]."""
    b = np.asarray(x, np.float32).reshape(-1, QK)
    d = (np.abs(b).max(axis=1) / np.float32(127)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, 1), 0).astype(np.float32)
    return np.rint(b * inv[:, None]).astype(np.int8), d.astype(np.float16)


def q8_q8_matvec(x, q, d):
    """Q8_0 x Q8_0 row as the DSP runtime computes it (bit-exact):
    sum_b float(int32 block dot) * (fp32(d_w) * fp32(d_x)), FP32 accumulation in block order."""
    qa, da = quantize_activation(x)
    isum = np.einsum("bj,nbj->nb", qa.astype(np.int64), q.astype(np.int64)).astype(np.float32)
    s = (d.astype(np.float32) * da.astype(np.float32)[None, :]).astype(np.float32)
    acc = np.zeros(q.shape[0], np.float32)
    for b in range(q.shape[1]):
        acc = (acc + (isum[:, b] * s[:, b]).astype(np.float32)).astype(np.float32)
    return acc
