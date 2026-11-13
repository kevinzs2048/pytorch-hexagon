# SPDX-License-Identifier: BSD-3-Clause
"""Ordered FP32 prefill stages for q16..32 and effective KV length up to 256.

The adapter validates causal masks and admits the fast path only when packing
reports |Q|, |K|, |V| < 2^40. Otherwise it calls the strict reference runtime.
Masked scores then round exactly to -FLT_MAX, their probabilities are zero,
and pruning preserves the original PV block partition and FMA order.
"""
import triton
import triton.language as tl
from models.qwen3.kernels.strict_decode import _exp_compat

SCORE_ELEMENTS = tl.constexpr(16 * 32 * 256)
SCRATCH_FP32_ELEMENTS = 16 * 32 * 256 + 32

# Keep the independent accumulators visible to the caller. The target-aware
# compiler preserves scalar FP32 FMAs without a forced call boundary.
@triton.jit
def _dot_pair(A, B, a_base, b_base, a_stride, length):
    accum = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for z in range(0, length):
        a = tl.load(A + a_base + z)
        a1 = tl.load(A + a_base + a_stride + z)
        for i in tl.static_range(0, 8):
            b = tl.load(B + b_base + z * 8 + i)
            value = tl.fma(a, b, accum[i])
            accum = accum[:i] + (value,) + accum[i + 1:]
            value1 = tl.fma(a1, b, accum[8 + i])
            accum = accum[:8 + i] + (value1,) + accum[9 + i:]
    return accum

@triton.jit
def _maximum_key(values):
    bits = values.to(tl.int32, bitcast=True)
    nan = (bits & 2147483647) > 2139095040
    mask = (bits >> 31) | -2147483648
    keys = tl.where(nan, 4294967295, (bits ^ mask).to(tl.uint32))
    selected = tl.max(keys, 0)
    inverse = tl.where(selected >= 2147483648, 2147483648, 4294967295).to(tl.uint32)
    result = selected ^ inverse
    result = tl.where(selected == 4294967295, 2143289344, result).to(tl.uint32)
    return result.to(tl.float32, bitcast=True)

@triton.jit
def qwen_prefill_pack(Scores, K, V, PackedK, PackedV, Q, query_length, kv_length, q_head_stride, q_token_stride, k_head_stride, k_token_stride, v_head_stride, v_token_stride, heads_per_program, SOFTMAX_BLOCK: tl.constexpr=32):
    score_stride = (kv_length + 31) // 32 * 32
    rows = tl.arange(0, 8)
    cols = tl.arange(0, 32)
    four = tl.arange(0, 4)
    lane = tl.arange(0, 8)
    kpad = (kv_length + 7) // 8 * 8
    vpad = (kv_length + 3) // 4 * 4
    for head in range(tl.program_id(1) * 2, tl.program_id(1) * 2 + 2):
        key_max = 0
        value_max = 0
        for start in range(0, kpad, 8):
            for feature in range(0, 128, 32):
                values = tl.load(K + head * k_head_stride + (start + rows[:,None]) * k_token_stride + feature + cols[None,:], start + rows[:,None] < kv_length, other=0)
                key_max = tl.maximum(key_max, tl.max(tl.reshape(values.to(tl.int32, bitcast=True) & 2147483647, (256,)), 0))
                tl.store(PackedK + (head * kpad + start) * 128 + (feature + cols[:,None]) * 8 + rows[None,:], tl.trans(values))
        for start in range(0, vpad, 4):
            for feature in range(0, 128, 32):
                values = tl.load(V + head * v_head_stride + (start + four[:,None]) * v_token_stride + feature + cols[None,:], start + four[:,None] < kv_length, other=0)
                value_max = tl.maximum(value_max, tl.max(tl.reshape(values.to(tl.int32, bitcast=True) & 2147483647, (128,)), 0))
                packed = tl.trans(tl.reshape(values, (4,4,8)), (1,0,2))
                tl.store(PackedV + ((head * 16 + feature // 8 + four[:,None,None]) * vpad + start + four[None,:,None]) * 8 + lane[None,None,:], packed)
        for qhead in range(head * 2, head * 2 + 2):
            query_max = 0
            for row in range(0, query_length):
                values = tl.load(Q + qhead * q_head_stride + row * q_token_stride + tl.arange(0, 128))
                query_max = tl.maximum(query_max, tl.max(values.to(tl.int32, bitcast=True) & 2147483647, 0))
            # |Q|,|K| < 2^40 gives |scaled dot| << half ULP(FLT_MAX).
            # Masked scores therefore round to exactly -FLT_MAX. Bounded V
            # makes the omitted zero-probability suffix preserve FMA results.
            safe = (query_max < 1400897536) & (key_max < 1400897536) & (value_max < 1400897536)
            tl.store(Scores + SCORE_ELEMENTS + qhead, safe.to(tl.float32))
    # Scores is the shared score workspace for this packing stage. Initialize the
    # final full32-lane block before QK overwrites its valid positions.
    if kv_length % 32:
        tail = kv_length // 32 * 32 + tl.arange(0, 32)
        for head in range(tl.program_id(1) * 4, tl.program_id(1) * 4 + 4):
            for row in range(0, query_length):
                tl.store(Scores + (head * query_length + row) * score_stride + tail,
                         tl.full((32,), -float('inf'), tl.float32))


@triton.jit
def qwen_prefill_qk(Q, K, V, Out, Scratch, query_length, kv_length, q_head_stride, q_token_stride, k_head_stride, k_token_stride, v_head_stride, v_token_stride, heads_per_program, SOFTMAX_BLOCK: tl.constexpr=32):
    score_stride = (kv_length + 31) // 32 * 32
    first_head = tl.program_id(1) * heads_per_program
    softmax_lane = tl.arange(0, SOFTMAX_BLOCK)
    kpad = (kv_length + 7) // 8 * 8
    vpad = (kv_length + 3) // 4 * 4
    for head in range(first_head, first_head + heads_per_program):
        kv_head = head // 2
        score_base = head * query_length * score_stride
        Scores = Scratch + score_base
        for start in range(0, kpad, 8):
            for qr in range(0, query_length, 2):
                stride = tl.where(qr + 1 < query_length, q_token_stride, 0)
                pair = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                if start <= kv_length - query_length + tl.minimum(qr + 1, query_length - 1):
                    pair = _dot_pair(Q, K, head * q_head_stride + qr * q_token_stride,
                                     (kv_head * kpad + start) * 128, stride, 128)
                for i in tl.static_range(0, 8):
                    if start + i < kv_length:
                        tl.store(Scores + qr * score_stride + start + i, pair[i] * 0.08838834764831845 + tl.where(start + i <= kv_length - query_length + qr, 0.0, -3.4028234663852886e38))
                        if qr + 1 < query_length:
                            tl.store(Scores + (qr + 1) * score_stride + start + i, pair[8 + i] * 0.08838834764831845 + tl.where(start + i <= kv_length - query_length + qr + 1, 0.0, -3.4028234663852886e38))


@triton.jit
def qwen_prefill_softmax(Q, K, V, Out, Scratch, query_length, kv_length, q_head_stride, q_token_stride, k_head_stride, k_token_stride, v_head_stride, v_token_stride, heads_per_program, SOFTMAX_BLOCK: tl.constexpr=32):
    score_stride = (kv_length + 31) // 32 * 32
    first_head = tl.program_id(1) * heads_per_program
    softmax_lane = tl.arange(0, SOFTMAX_BLOCK)
    kpad = (kv_length + 7) // 8 * 8
    vpad = (kv_length + 3) // 4 * 4
    for head in range(first_head, first_head + heads_per_program):
        score_base = head * query_length * score_stride
        Scores = Scratch + score_base
        for current_head in range(0, query_length):
            maximum = -float('inf')
            for start in range(0, kv_length, 32):
                positions = start + tl.arange(0, 32)
                values = tl.load(Scores + current_head * score_stride + positions)
                block_maximum = _maximum_key(values)
                maximum = tl.where((block_maximum > maximum) | (block_maximum != block_maximum), block_maximum, maximum)
            for start in range(0, kv_length, SOFTMAX_BLOCK):
                positions = start + softmax_lane
                scores = tl.load(Scores + current_head * score_stride + positions)
                probability = _exp_compat(scores - maximum)
                tl.store(Scores + current_head * score_stride + positions, probability)
            sums = (0.0, 0.0, 0.0, 0.0)
            for col in range(0, kv_length // 4 * 4, 4):
                for i in tl.static_range(0, 4):
                    value = sums[i] + tl.load(Scores + current_head * score_stride + col + i)
                    sums = sums[:i] + (value,) + sums[i + 1:]
            for i in tl.static_range(0, 4):
                if kv_length // 4 * 4 + i < kv_length:
                    value = sums[i] + tl.load(Scores + current_head * score_stride + kv_length // 4 * 4 + i)
                    sums = sums[:i] + (value,) + sums[i + 1:]
            total = sums[0] + sums[2] + (sums[1] + sums[3])
            inverse = 1.0 / total
            for start in range(0, kv_length, SOFTMAX_BLOCK):
                positions = start + softmax_lane
                probability = tl.load(Scores + current_head * score_stride + positions) * inverse
                tl.store(Scores + current_head * score_stride + positions, probability)


@triton.jit
def qwen_prefill_pv(Q, K, V, Out, Scratch, query_length, kv_length, q_head_stride, q_token_stride, k_head_stride, k_token_stride, v_head_stride, v_token_stride, heads_per_program, SOFTMAX_BLOCK: tl.constexpr=32):
    score_stride = (kv_length + 31) // 32 * 32
    first_head = tl.program_id(1) * heads_per_program
    softmax_lane = tl.arange(0, SOFTMAX_BLOCK)
    kpad = (kv_length + 7) // 8 * 8
    vpad = (kv_length + 3) // 4 * 4
    for head in range(first_head, first_head + heads_per_program):
        kv_head = head // 2
        score_base = head * query_length * score_stride
        Scores = Scratch + score_base
        for col in range(0, 128, 8):
            for qr in range(0, query_length, 2):
                # The adapter admits only q16..32 and KV<=256, so the reference
                # uses one PV accumulation block. Its final +0 stays explicit.
                stride = tl.where(qr + 1 < query_length, score_stride, 0)
                active_width = kv_length - query_length + tl.minimum(qr + 1, query_length - 1) + 1
                accum = _dot_pair(Scratch, V, score_base + qr * score_stride,
                                  (kv_head * 16 + col // 8) * vpad * 8, stride, active_width)
                for i in tl.static_range(0, 8):
                    tl.store(Out + (qr * 16 + head) * 128 + col + i, (0.0 + accum[i]))
                    if qr + 1 < query_length:
                        tl.store(Out + ((qr + 1) * 16 + head) * 128 + col + i, (0.0 + accum[8 + i]))


