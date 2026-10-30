"""Keep native ATen SiLU in the framework graph.

Numerical lowering is owned by the compiler policy stage.
"""
import torch


def preserve_native_silu(table):
    op=torch.ops.aten.silu.default
    if op not in table:
        raise ValueError('Expected a standard SiLU decomposition entry')
    selected=dict(table)
    del selected[op]
    return selected
