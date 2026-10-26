"""Build the original dynamic request in one owned immutable byte allocation.

Only host serialization changes. Shape admission, tensor order, padding and
every tensor byte retain the existing HXDY ABI. Borrowed NumPy views live only
until bytes.join finishes; no returned packet aliases a tensor or shared map.
"""
import struct
import sys

from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import MAGIC, layout_for


def pack_torch_request_joined(contract, arguments):
    import torch

    if sys.byteorder != 'little':
        raise ValueError('The buffer ABI requires a little-endian host')
    if len(arguments) != len(contract['inputs']):
        raise ValueError('Wrong model argument count')
    values, tensors = {}, []

    def bind(expected, actual):
        if type(actual) is not int or actual <= 0:
            raise ValueError('Runtime tensor dimensions must be positive integers')
        if type(expected) is int:
            if actual != expected:
                raise ValueError('Runtime tensor shape differs from compiled contract')
        elif expected in values:
            if actual != values[expected]:
                raise ValueError('Inconsistent symbolic dimension')
        else:
            values[expected] = actual

    for row, argument in zip(contract['inputs'], arguments, strict=True):
        if row['kind'] == 'scalar':
            if type(argument) is not int:
                raise ValueError('Runtime scalar must be a concrete Python integer')
            expected = row['value']
            if type(expected) is int:
                if argument != expected:
                    raise ValueError('Runtime scalar differs from compiled contract')
            elif expected in values:
                if argument != values[expected]:
                    raise ValueError('Runtime scalar disagrees with tensor shape')
            else:
                values[expected] = argument
            continue
        dtype = {'float32': torch.float32, 'int64': torch.int64}[row['dtype']]
        if (not isinstance(argument, torch.Tensor) or argument.dtype != dtype
                or argument.device.type != 'cpu' or argument.layout != torch.strided
                or not argument.is_contiguous() or argument.requires_grad):
            raise ValueError('Expected contiguous, inference-only CPU tensor with compiled dtype')
        if argument.ndim != len(row['shape']):
            raise ValueError('Runtime tensor rank differs from compiled contract')
        for expected, actual in zip(row['shape'], argument.shape, strict=True):
            bind(expected, int(actual))
        tensors.append(argument)
    for name, (lo, hi) in contract['variables'].items():
        if name not in values and lo == hi:
            values[name] = lo
    if set(values) != set(contract['variables']):
        raise ValueError('Cannot infer every runtime shape variable from model inputs')
    layout = layout_for(contract, values)
    names = tuple(contract['variables'])
    parts = [MAGIC + struct.pack('<'+'q'*len(names), *(values[name] for name in names))]
    views, cursor = [], 0
    try:
        for row, tensor in zip(layout['inputs'], tensors, strict=True):
            view = memoryview(tensor.numpy()).cast('B')
            views.append(view)
            if len(view) != row['bytes']:
                raise ValueError('Wrong input tensor byte count')
            gap = row['offset'] - cursor
            if gap < 0:
                raise ValueError('Canonical nonoverlapping tensor layout required')
            if gap:
                parts.append(bytes(gap))
            parts.append(view)
            cursor = row['offset'] + row['bytes']
        tail = layout['input_bytes'] - cursor
        if tail:
            parts.append(bytes(tail))
        packet = b''.join(parts)
        if len(packet) != len(parts[0]) + layout['input_bytes']:
            raise ValueError('Complete immutable request length differs')
        return packet, layout, values
    finally:
        for view in views:
            view.release()
