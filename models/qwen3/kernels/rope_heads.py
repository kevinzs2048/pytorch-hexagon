# SPDX-License-Identifier: BSD-3-Clause
"""Rotary embedding fused with the head-major transpose that feeds attention.

For units [start, start + count) of (head, row), head-major, it reads the
row-major projection Y[row, head, 0:128] and writes Out[head, row, 0:128]:
    out[d]      = y[d] * cos[d] + (-y[d + 64]) * sin[d]          d < 64
    out[d + 64] = y[d + 64] * cos[d + 64] + y[d] * sin[d + 64]
which is the graph's x * cos + concat(-x_hi, x_lo) * sin in the same
operation order (negate, two multiplies, one addition; no contraction).
Negation flips the sign bit like arith.negf; Triton's unary minus would be
0.0 - x, which differs for +0.0.
"""
import triton
import triton.language as tl


@triton.jit
def qwen_rope_heads(Y, Cos, Sin, Out, start, count, rows, y_row, y_head,
                    c_row, o_head, o_row):
    d = tl.arange(0, 64)
    # Walk (head, row) incrementally: one division per call, not per unit.
    head = start // rows
    row = start - head * rows
    for unit in range(0, count):
        source = Y + row * y_row + head * y_head
        low = tl.load(source + d)
        high = tl.load(source + 64 + d)
        cos_low = tl.load(Cos + row * c_row + d)
        cos_high = tl.load(Cos + row * c_row + 64 + d)
        sin_low = tl.load(Sin + row * c_row + d)
        sin_high = tl.load(Sin + row * c_row + 64 + d)
        negated = (high.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
        target = Out + head * o_head + row * o_row
        tl.store(target + d, low * cos_low + negated * sin_low)
        tl.store(target + 64 + d, high * cos_high + low * sin_high)
        row += 1
        wrap = row == rows
        head += wrap.to(tl.int32)
        row = tl.where(wrap, 0, row)
