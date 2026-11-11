# SPDX-License-Identifier: BSD-3-Clause
"""Additive causal/padding attention mask, one query row per unit.

For rows [start, start + count) and columns j < kv:
    Out[row, j] = 0.0       if j <= Pos[row * pos_step] and Valid[j] != 0
                  -FLT_MAX  otherwise
which is the graph's select(causal & padding, 0.0, -3.40282347e+38). The
result has only those two exact values, so it is independent of evaluation order.
Full column blocks use unmasked vector loads/stores; only the final partial
block takes the masked (copying) path.
"""
import triton
import triton.language as tl


@triton.jit
def qwen_causal_mask(Pos, Valid, Out, start, count, kv, pos_step, BLOCK: tl.constexpr):
    j = tl.arange(0, BLOCK)
    full = kv - kv % BLOCK
    for unit in range(0, count):
        row = start + unit
        limit = tl.load(Pos + row * pos_step)
        for base in range(0, full, BLOCK):
            column = base + j
            valid = tl.load(Valid + column)
            keep = (column.to(tl.int64) <= limit) & (valid != 0)
            tl.store(Out + row * kv + column, tl.where(keep, 0.0, -3.4028234663852886e38))
        if full < kv:
            column = full + j
            inside = column < kv
            valid = tl.load(Valid + column, mask=inside, other=0)
            keep = (column.to(tl.int64) <= limit) & (valid != 0)
            tl.store(Out + row * kv + column, tl.where(keep, 0.0, -3.4028234663852886e38), mask=inside)
