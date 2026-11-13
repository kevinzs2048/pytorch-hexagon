# SPDX-License-Identifier: BSD-3-Clause
"""Experimental Triton online attention; not yet selected by the model runtime.

Q: [16, query, 128], K/V: [8, capacity, 128], Out: [query, 16, 128].
Strides are in elements, so token-major resident KV needs no layout conversion.
Query positions are the final `query` positions of the effective KV sequence.
EXPERIMENTAL precision: FP16 Q/K/V/probabilities and dot outputs,
FP32 online state, pairwise sums, approximate exponential and reciprocal scale.
This does not preserve the accepted W8A8 + FP32 model numerical policy.
"""
import triton
import triton.language as tl


@triton.jit
def _softmax_exp(x):
    tl.static_assert(x.dtype == tl.float32)
    # Explicit approximation for nonpositive softmax arguments. Terms below
    # exp(-80) are discarded. This policy must be validated at model level.
    # Keep range reduction finite even for NaN/Inf or masked lanes.
    # Classify FP32 bits once. Signed integer <= 0 includes both zeros
    # and all negative values; the magnitude guard excludes NaN/Inf.
    bits = x.to(tl.int32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    nonpositive = bits <= 0
    in_range = nonpositive & (magnitude <= 0x42A00000)  # abs(x) <= 80
    bounded = tl.where(in_range, x, 0.0)
    scaled = bounded * 1.4426950408889634 + 0.5
    # bounded is in [-80, 0], so scaled is in [-115, 0.5]. Adding
    # 1.5 * 2**23 rounds to an integer in FP32 without a numeric int cast.
    # Correct a rounded-up value to floor, keeping both integer and FP32
    # representations. All integer-valued subtractions below are exact.
    # Preserve FP32 rounding at each step: do not enable reassociation.
    biased = scaled + 12582912.0
    rounded = biased - 12582912.0
    round_up = rounded > scaled
    exponent_int = (biased.to(tl.int32, bitcast=True) - 0x4B400000
                    - round_up.to(tl.int32))
    exponent = rounded - tl.where(round_up, 1.0, 0.0)
    remainder = (bounded - exponent * 0.693145751953125) - exponent * 1.428606765330187e-6
    polynomial = 1.0 + remainder * (1.0 + remainder * (0.5 + remainder * (
        0.16666666666666666 + remainder * (0.041666666666666664 + remainder * (
            0.008333333333333333 + remainder * (0.001388888888888889 + remainder * 0.0001984126984126984))))))
    scale = ((exponent_int + 127) << 23).to(tl.float32, bitcast=True)
    value = tl.where((bits < 0) & (magnitude > 0x42A00000), 0.0, polynomial * scale)
    # This helper is defined only for nonpositive softmax arguments.
    return tl.where((~nonpositive) | (magnitude > 0x7F800000), float("nan"), value)


@triton.jit
def _attention_tile(
    Q, K, V, Out,
    query_length, kv_length,
    q_head_stride, q_token_stride,
    k_head_stride, k_token_stride,
    v_head_stride, v_token_stride,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 128,
    FP16_DOT: tl.constexpr = True,
    APPROX_EXP: tl.constexpr = True,
    HEAD_MULTIPLIER: tl.constexpr = 1, HEAD_OFFSET=0,
):
    tl.static_assert((BLOCK_M == 2 or BLOCK_M == 32 or BLOCK_M == 64) and BLOCK_N == 128)
    tl.static_assert(FP16_DOT and APPROX_EXP)
    # One program handles both query heads sharing one KV head.
    rows_per_head: tl.constexpr = BLOCK_M // 2
    tile = 0 if BLOCK_M == 2 else tl.program_id(0)
    kv_head = tl.program_id(1) * HEAD_MULTIPLIER + HEAD_OFFSET
    lanes = tl.arange(0, BLOCK_M)
    head = kv_head * 2 + lanes // rows_per_head
    rows = tile * rows_per_head + lanes % rows_per_head
    dims = tl.arange(0, 128)
    cols = tl.arange(0, BLOCK_N)
    qrows = tile * rows_per_head + tl.arange(0, rows_per_head)
    q0 = tl.load(Q + (kv_head * 2) * q_head_stride + qrows[:, None] * q_token_stride
                 + dims[None, :], qrows[:, None] < query_length, other=0)
    q1 = tl.load(Q + (kv_head * 2 + 1) * q_head_stride + qrows[:, None] * q_token_stride
                 + dims[None, :], qrows[:, None] < query_length, other=0)
    q = tl.reshape(tl.permute(tl.join(q0, q1), (2, 0, 1)), (BLOCK_M, 128))
    if FP16_DOT:
        q = q.to(tl.float16)
    maximum = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, 128), tl.float32)
    # Runtime bound, independent of allocated capacity and prompt contents.
    for start in range(0, kv_length, BLOCK_N):
        positions = start + cols
        k = tl.load(K + kv_head * k_head_stride
                    + positions[:, None] * k_token_stride + dims[None, :],
                    positions[:, None] < kv_length, other=0)
        v = tl.load(V + kv_head * v_head_stride
                    + positions[:, None] * v_token_stride + dims[None, :],
                    positions[:, None] < kv_length, other=0)
        if FP16_DOT:
            k = k.to(tl.float16)
            v = v.to(tl.float16)
        scores = tl.dot(q, tl.trans(k), input_precision='ieee', out_dtype=tl.float16).to(tl.float32) * 0.08838834764831845
        visible = (positions[None, :] < kv_length) & (
            positions[None, :] <= kv_length - query_length + rows[:, None])
        scores = tl.where(visible, scores, -float('inf'))
        maxima = scores
        for level in tl.static_range(7):
            low, high = tl.split(tl.permute(
                tl.reshape(maxima, (BLOCK_M, 2, 64 >> level)), (0, 2, 1)))
            maxima = tl.maximum(low, high)
        next_maximum = tl.maximum(maximum, tl.reshape(maxima, (BLOCK_M,)))
        if APPROX_EXP:
            probability = _softmax_exp(scores - next_maximum[:, None])
            correction = _softmax_exp(maximum - next_maximum)
        else:
            probability = tl.exp(scores - next_maximum[:, None])
            correction = tl.exp(maximum - next_maximum)
        sums = probability
        for level in tl.static_range(7):
            low, high = tl.split(tl.permute(
                tl.reshape(sums, (BLOCK_M, 2, 64 >> level)), (0, 2, 1)))
            sums = low + high
        denominator = denominator * correction + tl.reshape(sums, (BLOCK_M,))
        accumulator = accumulator * correction[:, None]
        if FP16_DOT:
            probability = probability.to(tl.float16)
        product = tl.dot(probability, v, input_precision='ieee', out_dtype=tl.float16)
        accumulator = accumulator + product.to(tl.float32)
        maximum = next_maximum
    inverse = 1.0 / denominator
    result = accumulator * inverse[:, None]
    out0, out1 = tl.split(tl.permute(tl.reshape(result, (2, rows_per_head, 128)), (1, 2, 0)))
    tl.store(Out + qrows[:, None] * 2048 + (kv_head * 2) * 128 + dims[None, :],
             out0, qrows[:, None] < query_length)
    tl.store(Out + qrows[:, None] * 2048 + (kv_head * 2 + 1) * 128 + dims[None, :],
             out1, qrows[:, None] < query_length)


@triton.jit
def attention_forward(
    Q, K, V, Out, query_length, kv_length,
    q_head_stride, q_token_stride, k_head_stride, k_token_stride,
    v_head_stride, v_token_stride,
    BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 128,
    FP16_DOT: tl.constexpr = True, APPROX_EXP: tl.constexpr = True,
):
    tl.static_assert(BLOCK_M == 64 and BLOCK_N == 128)
    # Runtime shape selects the tile; both branches are in one compiled object.
    if query_length <= 16:
        _attention_tile(Q, K, V, Out, query_length, kv_length,
                        q_head_stride, q_token_stride, k_head_stride, k_token_stride,
                        v_head_stride, v_token_stride,
                        32, BLOCK_N, FP16_DOT, APPROX_EXP)
    else:
        _attention_tile(Q, K, V, Out, query_length, kv_length,
                        q_head_stride, q_token_stride, k_head_stride, k_token_stride,
                        v_head_stride, v_token_stride,
                        64, BLOCK_N, FP16_DOT, APPROX_EXP)
