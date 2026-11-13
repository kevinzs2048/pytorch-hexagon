# SPDX-License-Identifier: BSD-3-Clause
"""Vector FP32 SiLU with a bounded approximation and strict exceptional path.

The ordinary finite domain uses non-fused FP32 exponent evaluation and three
reciprocal refinements. It is not a bitwise substitute for the strict kernel.
Zeros or 2^-64 <= abs(x) <= 16 are admitted per vector; any other lane takes
the existing strict exponent/division path for the entire vector. The input
and output must not overlap because exceptional vectors are repaired after
the ordinary pass.
"""
import triton
import triton.language as tl
from models.qwen3.kernels.quantized_silu_up import _reciprocal_guarded_domain
from models.qwen3.kernels.silu import _divide_positive_normal
from models.qwen3.kernels.strict_decode import _exp_compat


@triton.jit(noinline=True)
def _strict_vector(X, Y, input_offset, output_offset):
    # Keep the cold path's constants and accumulators out of the hot loop's
    # register allocation. An exceptional vector still uses the exact kernel.
    lane = tl.arange(0, 32)
    x = tl.load(X + input_offset + lane)
    y = _divide_positive_normal(x, 1.0 + _exp_compat(-x))
    tl.store(Y + output_offset + lane, y)


@triton.jit
def _exp_silu_domain(x):
    # Range reduction for [-16,16]; the reduced interval lies in [-.347,.347].
    # Degree-five Chebyshev interpolation of exp on that interval. Its real
    # arithmetic remainder is < 1.1e-7; device FP32 error is checked separately.
    rounded = x * 1.4426950408889634 + 12582912.0
    exponent = rounded.to(tl.int32, bitcast=True) - 0x4b400000
    nearest = rounded - 12582912.0
    remainder = (x - nearest * 0.693145751953125) - nearest * 1.428606765330187e-6
    polynomial = (((((0.008369236841999675 * remainder + 0.04191812637141698) * remainder
                    + 0.16666504463628357) * remainder + 0.49998863795080173) * remainder
                    + 1.0000000108513993) * remainder + 1.0000000760142347)
    return polynomial * ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _domain_violation(x):
    magnitude = x.to(tl.int32, bitcast=True) & 0x7fffffff
    return tl.maximum(tl.minimum(0x1f800000 - magnitude, magnitude),
                      magnitude - 0x41800000)


@triton.jit
def silu_forward(X, Y, rows, stride, BLOCK: tl.constexpr = 32):
    tl.static_assert(BLOCK == 32)
    lane = tl.arange(0, BLOCK)
    for row in range(rows):
        bad = tl.full((BLOCK,), 0, tl.int32)
        for col in range(tl.program_id(0) * 768,
                         (tl.program_id(0) + 1) * 768, 2 * BLOCK):
            x0 = tl.load(X + row * stride + col + lane)
            x1 = tl.load(X + row * stride + col + BLOCK + lane)
            y0 = x0 * _reciprocal_guarded_domain(1.0 + _exp_silu_domain(-x0))
            y1 = x1 * _reciprocal_guarded_domain(1.0 + _exp_silu_domain(-x1))
            tl.store(Y + row * 3072 + col + lane, y0)
            tl.store(Y + row * 3072 + col + BLOCK + lane, y1)
            bad = tl.maximum(bad, tl.maximum(_domain_violation(x0), _domain_violation(x1)))
        # A single scalar guard per row/worker keeps the ordinary loop free
        # of control-flow boundaries. Repair exceptional vectors before return.
        if tl.max(bad, 0) > 0:
            for col in range(tl.program_id(0) * 768,
                             (tl.program_id(0) + 1) * 768, BLOCK):
                x = tl.load(X + row * stride + col + lane)
                if tl.max(_domain_violation(x), 0) > 0:
                    _strict_vector(X, Y, row * stride + col, row * 3072 + col)
