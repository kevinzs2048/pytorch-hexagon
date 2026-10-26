"""Adapt native Qwen3 AOT q1/prefill inputs to one bounded Hexagon ABI.

PyTorch specializes the single-token graph enough to omit the leading query
SymInt, while its q>=2 symbolic graph retains that scalar. The downstream
buffer contract still validates every tensor, scalar and byte count before
any RPC; this adapter only restores the omitted q=1 scalar.
"""

import torch


def normalize_qwen3_runtime_arguments(values):
    values = tuple(values)
    if len(values) == 62:
        if (not isinstance(values[0], torch.Tensor)
                or tuple(values[0].shape) != (1, 1)
                or not isinstance(values[1], torch.Tensor)
                or tuple(values[1].shape) != (1,)
                or not isinstance(values[2], torch.Tensor)
                or tuple(values[2].shape) != (1, 1)
                or type(values[3]) is not int or values[3] != 1024
                or type(values[61]) is not int or values[61] != 1):
            raise ValueError('Native q1 AOT runtime prefix/suffix differs')
        return (1, *values)
    if len(values) == 63:
        query = values[0]
        if (type(query) is not int or not 2 <= query <= 32
                or not isinstance(values[1], torch.Tensor)
                or tuple(values[1].shape) != (1, query)
                or not isinstance(values[2], torch.Tensor)
                or tuple(values[2].shape) != (query,)
                or not isinstance(values[3], torch.Tensor)
                or tuple(values[3].shape) != (1, query)
                or type(values[4]) is not int or values[4] != 1024
                or type(values[62]) is not int or values[62] != 1):
            raise ValueError('Native bounded-prefill AOT runtime prefix/suffix differs')
        return values
    raise ValueError('Only native q1 or bounded-prefill AOT inputs are admitted')
