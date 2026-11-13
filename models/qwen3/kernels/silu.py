# SPDX-License-Identifier: BSD-3-Clause
"""FP32 SiLU preserving the existing compatibility exponent and division.

Four programs partition each 3072-element row into 768 elements. The compiler
must admit 32-lane parallel mixed FP32/predicate operations for HVX vectorization.
"""
import triton
import triton.language as tl
from models.qwen3.kernels.strict_decode import _exp_compat


@triton.jit
def _divide_positive_normal(x, denominator):
    """Exact FP32 division using integer mantissas, with an IEEE fallback.

    The fast path admits a positive normal denominator and a normal quotient
    (or a signed-zero numerator). Each restoring step preserves
    numerator_prefix = quotient * divisor + remainder, 0 <= remainder < divisor.
    After 23 steps, round the exact remainder to nearest, ties to even. No
    reciprocal approximation or floating-point reassociation is involved.
    """
    xb = x.to(tl.uint32, bitcast=True)
    db = denominator.to(tl.uint32, bitcast=True)
    magnitude = xb & 0x7fffffff
    xe = (magnitude >> 23).to(tl.int32)
    de = (db >> 23).to(tl.int32)  # Sign bit makes negative denominators fail.
    xm = (magnitude & 0x7fffff) | 0x800000
    dm = (db & 0x7fffff) | 0x800000
    below = ((xm.to(tl.int32) - dm.to(tl.int32)) >> 31) & 1
    exponent = xe - de + 127 - below
    bad_x = tl.maximum(1 - xe, xe - 254)
    bad_d = tl.maximum(1 - de, de - 254)
    bad_e = tl.maximum(1 - exponent, exponent - 254)
    bad = tl.maximum(bad_d, tl.minimum(magnitude.to(tl.int32),
                                      tl.maximum(bad_x, bad_e)))
    if tl.max(bad, 0) > 0:
        result = tl.div_rn(x, denominator)
    else:
        remainder = (xm << below.to(tl.uint32)) - dm
        quotient = tl.full((32,), 1, tl.uint32)
        for step in range(11):
            for digit in tl.static_range(2):
                remainder = remainder << 1
                bit = remainder >= dm
                remainder = tl.where(bit, remainder - dm, remainder)
                quotient = (quotient << 1) + bit.to(tl.uint32)
        remainder = remainder << 1
        bit = remainder >= dm
        remainder = tl.where(bit, remainder - dm, remainder)
        quotient = (quotient << 1) + bit.to(tl.uint32)
        twice = remainder << 1
        rounding = (twice > dm) | ((twice == dm) & ((quotient & 1) != 0))
        quotient = quotient + rounding.to(tl.uint32)
        bits = ((exponent - 1).to(tl.uint32) << 23) + quotient
        bits = bits | (xb & 0x80000000)
        zero_mask = ((magnitude | (0 - magnitude)) >> 31) - 1
        bits = (bits & ~zero_mask) | (xb & zero_mask)
        result = bits.to(tl.float32, bitcast=True)
    return result


@triton.jit
def silu_forward(X, Y, rows, stride, BLOCK: tl.constexpr = 32):
    tl.static_assert(BLOCK == 32)
    lane = tl.arange(0, BLOCK)
    for row in range(rows):
        for col in range(tl.program_id(0) * 768,
                         (tl.program_id(0) + 1) * 768, BLOCK):
            x = tl.load(X + row * stride + col + lane)
            y = _divide_positive_normal(x, 1.0 + _exp_compat(-x))
            tl.store(Y + row * 3072 + col + lane, y)
