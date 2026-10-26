"""Opt-in FP32 SiLU decomposition using the native CPU operation order.

ATen's CPU SiLU computes x / (1 + exp(-x)). The generic Python and torch-mlir
C++ decompositions instead compute x * sigmoid(x), adding a rounded reciprocal
before multiplication. In dynamically quantized graphs that small difference
can change subsequent INT8 codes. Keep a direct division for native FP32 SiLU.

This is not a promise of CPU/DSP bitwise transcendental equality: exp routines
may still differ. Full model outputs must retain their original acceptance
reference and tolerance. This helper is opt-in and changes neither explicit
x*sigmoid(x) expressions nor non-FP32 SiLU decompositions.
"""
import torch


def with_fp32_silu_division(table):
    """Copy a decomposition table, overriding only aten.silu.default in FP32.

The original entry is required and retained for every other dtype. The caller's
table and PyTorch's process-wide decomposition registrations remain untouched.
"""
    op = torch.ops.aten.silu.default
    original = table.get(op)
    if original is None:
        raise ValueError("A standard aten.silu.default decomposition is required")

    def silu(value):
        if value.dtype != torch.float32:
            return original(value)
        return value / (1.0 + torch.exp(-value))

    result = dict(table)
    result[op] = silu
    return result
