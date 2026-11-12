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

qwen_rope_heads_norm first applies the per-head RMSNorm that feeds RoPE in
Qwen3 (q_norm / k_norm), with the graph's ARM-ordered sum of squares: lanes
l = 0..15 accumulate the 8 chunks y[16k + l]^2 in order (each chunk loaded
separately; deriving them from the 64-wide loads by permute/split was slower),
lanes fold 16 -> 4 as ((a[j] + a[4+j]) + a[8+j]) + a[12+j], the four fold as
((s0 + s1) + s2) + s3; mean = sum / 128, r = rsqrt(mean + eps) and
x = W * (y * r). The rotary result of x is written as above.
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


@triton.jit
def qwen_rope_heads_norm(Y, W, Cos, Sin, Out, start, count, rows, y_row, y_head,
                    c_row, o_head, o_row, eps):
    d = tl.arange(0, 64)
    lane = tl.arange(0, 16)
    w_low = tl.load(W + d)
    w_high = tl.load(W + 64 + d)
    head = start // rows
    row = start - head * rows
    for unit in range(0, count):
        source = Y + row * y_row + head * y_head
        c = tl.load(source + lane)
        acc = c * c
        c = tl.load(source + 16 + lane)
        acc = acc + c * c
        c = tl.load(source + 32 + lane)
        acc = acc + c * c
        c = tl.load(source + 48 + lane)
        acc = acc + c * c
        c = tl.load(source + 64 + lane)
        acc = acc + c * c
        c = tl.load(source + 80 + lane)
        acc = acc + c * c
        c = tl.load(source + 96 + lane)
        acc = acc + c * c
        c = tl.load(source + 112 + lane)
        acc = acc + c * c
        # acc[4q + j] -> [j, a, b] with q = 2a + b.
        quarter = tl.permute(tl.reshape(acc, (2, 2, 4)), (2, 0, 1))
        b0, b1 = tl.split(quarter)
        q0, q2 = tl.split(b0)
        q1, q3 = tl.split(b1)
        s = ((q0 + q1) + q2) + q3
        e0, e1 = tl.split(tl.reshape(s, (2, 2)))
        s0, s2 = tl.split(e0)
        s1, s3 = tl.split(e1)
        total = (((s0 + s1) + s2) + s3)
        r = tl.math.rsqrt(total / 128.0 + eps)
        low = w_low * (tl.load(source + d) * r)
        high = w_high * (tl.load(source + 64 + d) * r)
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
