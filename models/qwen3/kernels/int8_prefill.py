# SPDX-License-Identifier: BSD-3-Clause
"""Exact INT8 HMX tile conversion; padded rows supply the zero-point bias.

The kernels accept 1..32 query rows. Rows 32..63 contain unsigned 128, so
HMX output row 32 is exactly 128 * sum_K(W). Subtracting it from real
rows recovers signed INT8 matmul without a separate weight reduction.
The runtime retains HVX for small queries where padding and HMX setup cost more.
"""
import triton
import triton.language as tl

@triton.jit
def qwen_int8_pack_activation(A, Packed, rows, width):
    r = tl.arange(0, 4)
    c = tl.arange(0, 128)
    b = tl.arange(0, 32)
    for tile in range(0, width // 128):
        for start in range(0, 64, 4):
            rr = start + r
            x = tl.load(A + rr[:, None] * width + tile * 128 + c[None, :],
                        rr[:, None] < rows, other=0)
            u = x.to(tl.uint8) ^ 128
            tiled = tl.trans(tl.reshape(u, (4, 4, 32)), (1, 0, 2))
            tl.store(Packed + (tile * 4 + r[:, None, None]) * 2048 +
                     (start + r[None, :, None]) * 32 + b[None, None, :], tiled)
    for tile in range(width // 128 * 4, width // 32):
        for start in range(0, 64, 4):
            rr = start + r
            x = tl.load(A + rr[:, None] * width + tile * 32 + b[None, :],
                        rr[:, None] < rows, other=0)
            tl.store(Packed + tile * 2048 + rr[:, None] * 32 + b[None, :],
                     x.to(tl.uint8) ^ 128)


@triton.jit
def qwen_int8_accumulate_result(Acc, Out, rows, columns, column):
    c = tl.arange(0, 32)
    bias = tl.load(Acc + 32 * 32 + c)
    for row in range(0, rows):
        value = tl.load(Acc + row * 32 + c) - bias
        old = tl.load(Out + row * columns + column + c)
        tl.store(Out + row * columns + column + c, old + value)


# Fused output stage for dequantized projections. Per output element it
# evaluates the replaced graph in its original order:
#   ((float(acc - bias) * RowScale[row]) * ColScale[column]) [+ Residual | * Factor]
# over `tiles` 32-column tiles starting at `column`. Accumulator rows are
# `acc_stride` apart; tile t's bias row is at Bias + t * bias_step (HMX tiles
# carry it as row 32; a plain INT32 product passes a zero row and step 0).
# The caller passes RowScale stride 0 when one scale broadcasts to every row.
@triton.jit
def _dequant_tiles(Acc, Bias, Out, RowScale, ColScale, Other, rows, columns, column,
                   tiles, acc_stride, bias_step, scale_stride, other_stride, MODE: tl.constexpr):
    c = tl.arange(0, 32)
    for t in range(0, tiles):
        col = column + t * 32
        bias = tl.load(Bias + t * bias_step + c)
        weight = tl.load(ColScale + col + c)
        for row in range(0, rows):
            value = (tl.load(Acc + row * acc_stride + t * 32 + c) - bias).to(tl.float32)
            scale = tl.load(RowScale + row * scale_stride)
            result = (value * scale) * weight
            if MODE == 1:
                result = tl.load(Other + row * other_stride + col + c) + result
            if MODE == 2:
                result = tl.load(Other + row * other_stride + col + c) * result
            tl.store(Out + row * columns + col + c, result)


@triton.jit
def qwen_int8_dequant_result(Acc, Bias, Out, RowScale, ColScale, rows, columns, column,
                             tiles, acc_stride, bias_step, scale_stride):
    _dequant_tiles(Acc, Bias, Out, RowScale, ColScale, ColScale, rows, columns, column,
                   tiles, acc_stride, bias_step, scale_stride, 0, 0)


@triton.jit
def qwen_int8_dequant_residual_result(Acc, Bias, Out, RowScale, ColScale, Residual, rows,
                                      columns, column, tiles, acc_stride, bias_step,
                                      scale_stride, residual_stride):
    _dequant_tiles(Acc, Bias, Out, RowScale, ColScale, Residual, rows, columns, column,
                   tiles, acc_stride, bias_step, scale_stride, residual_stride, 1)


@triton.jit
def qwen_int8_dequant_product_result(Acc, Bias, Out, RowScale, ColScale, Factor, rows,
                                     columns, column, tiles, acc_stride, bias_step,
                                     scale_stride, factor_stride):
    _dequant_tiles(Acc, Bias, Out, RowScale, ColScale, Factor, rows, columns, column,
                   tiles, acc_stride, bias_step, scale_stride, factor_stride, 2)
