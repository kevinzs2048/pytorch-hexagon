# SPDX-License-Identifier: BSD-3-Clause
"""q1 HMX attention with two adjacent KV groups per program.

Four logical query rows share one padded HMX tile. Block-diagonal Q keeps the
heads independent; output slicing selects each head's own V features. Persistent
KV remains FP32. Nonadjacent head layouts use the original one-group algorithm.
"""
import triton
import triton.language as tl
from models.qwen3.kernels.attention import _softmax_exp, _attention_tile


@triton.jit
def _attention_two_groups(
    Q, K, V, Out, query_length, kv_length,
    q_head_stride, q_token_stride, k_head_stride, k_token_stride,
    v_head_stride, v_token_stride, BLOCK_N: tl.constexpr = 128,
):
    tl.static_assert(BLOCK_N == 128)
    group = tl.program_id(1)
    dims = tl.arange(0, 256)
    cols = tl.arange(0, 128)
    d128 = tl.arange(0, 128)
    # Contiguous loads plus explicit zero blocks avoid scalar masked Q loads.
    q0 = tl.load(Q + group * 4 * q_head_stride + d128)
    q1 = tl.load(Q + (group * 4 + 1) * q_head_stride + d128)
    q2 = tl.load(Q + (group * 4 + 2) * q_head_stride + d128)
    q3 = tl.load(Q + (group * 4 + 3) * q_head_stride + d128)
    qa = tl.permute(tl.join(q0, q1), (1, 0))
    qb = tl.permute(tl.join(q2, q3), (1, 0))
    zeros = tl.zeros((2, 128), tl.float32)
    top = tl.reshape(tl.permute(tl.join(qa, zeros), (0, 2, 1)), (2, 256))
    bottom = tl.reshape(tl.permute(tl.join(zeros, qb), (0, 2, 1)), (2, 256))
    q = tl.reshape(tl.permute(tl.join(top, bottom), (2, 0, 1)), (4, 256)).to(tl.float16)
    maximum = tl.full((4,), -float('inf'), tl.float32)
    denominator = tl.zeros((4,), tl.float32)
    accumulator = tl.zeros((4, 256), tl.float32)
    for start in range(0, kv_length, 128):
        positions = start + cols
        k = tl.load(K + group * 2 * k_head_stride
                    + positions[:, None] * k_token_stride + dims[None, :],
                    positions[:, None] < kv_length, other=0).to(tl.float16)
        v = tl.load(V + group * 2 * v_head_stride
                    + positions[:, None] * v_token_stride + dims[None, :],
                    positions[:, None] < kv_length, other=0).to(tl.float16)
        scores = tl.dot(q, tl.trans(k), input_precision='ieee', out_dtype=tl.float16)
        scores = scores.to(tl.float32) * 0.08838834764831845
        scores = tl.where(positions[None, :] < kv_length, scores, -float('inf'))
        maxima = scores
        for level in tl.static_range(7):
            low, high = tl.split(tl.permute(
                tl.reshape(maxima, (4, 2, 64 >> level)), (0, 2, 1)))
            maxima = tl.maximum(low, high)
        next_maximum = tl.maximum(maximum, tl.reshape(maxima, (4,)))
        probability = _softmax_exp(scores - next_maximum[:, None])
        correction = _softmax_exp(maximum - next_maximum)
        sums = probability
        for level in tl.static_range(7):
            low, high = tl.split(tl.permute(
                tl.reshape(sums, (4, 2, 64 >> level)), (0, 2, 1)))
            sums = low + high
        denominator = denominator * correction + tl.reshape(sums, (4,))
        accumulator = accumulator * correction[:, None]
        product = tl.dot(probability.to(tl.float16), v,
                         input_precision='ieee', out_dtype=tl.float16)
        accumulator = accumulator + product.to(tl.float32)
        maximum = next_maximum
    result = accumulator * (1.0 / denominator)[:, None]
    # Select each query pair's own 128 features; avoid duplicate masked stores.
    a, b = tl.split(tl.permute(tl.reshape(result, (2, 2, 2, 128)), (1, 2, 3, 0)))
    outa, unuseda = tl.split(tl.permute(a, (0, 2, 1)))
    unusedb, outb = tl.split(tl.permute(b, (0, 2, 1)))
    pair = tl.arange(0, 2)
    tl.store(Out + (group * 4 + pair[:, None]) * 128 + d128[None, :], outa)
    tl.store(Out + (group * 4 + 2 + pair[:, None]) * 128 + d128[None, :], outb)


@triton.jit
def attention_forward_q1_hmx_g2(
    Q, K, V, Out, query_length, kv_length,
    q_head_stride, q_token_stride, k_head_stride, k_token_stride,
    v_head_stride, v_token_stride, BLOCK_N: tl.constexpr = 128,
):
    if k_head_stride == 128 and v_head_stride == 128:
        _attention_two_groups(Q, K, V, Out, query_length, kv_length,
                              q_head_stride, q_token_stride, k_head_stride,
                              k_token_stride, v_head_stride, v_token_stride,
                              BLOCK_N)
    else:
        # Four programs still cover all eight KV heads for admitted strided
        # inputs. Preserve the established arithmetic outside the packed layout.
        # A single call site retains load-to-packing fusion for both heads.
        for head_offset in range(2):
            _attention_tile(Q, K, V, Out, 1, kv_length,
                            q_head_stride, q_token_stride, k_head_stride,
                            k_token_stride, v_head_stride, v_token_stride,
                            2, BLOCK_N, True, True, 2, head_offset)
