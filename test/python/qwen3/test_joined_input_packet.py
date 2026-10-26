"""Check ABI bytes, float bit patterns, ownership and rejection boundaries."""
import copy
import sys
from pathlib import Path

import pytest
import torch

from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import pack_torch_request
from qwen3.runtime.support.joined_input_packet import pack_torch_request_joined


def fixture(q=3):
    contract = dict(variables={'query': [1, 32]}, inputs=[
        dict(kind='scalar', value='query'),
        dict(kind='tensor', dtype='int64', shape=[1, 'query']),
        dict(kind='scalar', value=1024),
        dict(kind='tensor', dtype='float32', shape=[1, 2, 'query', 7]),
        dict(kind='tensor', dtype='float32', shape=[1, 2, 11, 7]),
    ], outputs=[dict(kind='tensor', dtype='float32', shape=[1, 'query', 7])])
    args = [q, torch.arange(q).reshape(1, q), 1024,
            torch.arange(2*q*7, dtype=torch.float32).reshape(1, 2, q, 7),
            torch.arange(154, dtype=torch.float32).reshape(1, 2, 11, 7)]
    return contract, args


@pytest.mark.parametrize('q', [1, 2, 3, 13, 17, 31, 32])
def test_complete_packet_and_all_padding(q):
    c, args = fixture(q)
    assert pack_torch_request_joined(c, args) == pack_torch_request(c, args)


def test_signed_zero_nonfinite_and_subnormal_bits():
    c, args = fixture(3)
    bits = torch.tensor([0, -2147483648, 1, 2139095040, -8388608,
                         2143363909, 2139169605], dtype=torch.int32)
    args[3].view(torch.int32).reshape(-1)[:] = bits.repeat(6)
    assert pack_torch_request_joined(c, args) == pack_torch_request(c, args)


def test_packet_owns_snapshot_and_repeated_alias_inputs():
    c, args = fixture(3)
    c['inputs'].append(copy.deepcopy(c['inputs'][3]))
    args.append(args[3])
    packet = pack_torch_request_joined(c, args)[0]
    expected = pack_torch_request(c, args)[0]
    args[3].fill_(42)
    assert type(packet) is bytes and packet == expected
    assert pack_torch_request_joined(c, args)[0] != packet


def test_fixed_query_without_runtime_scalar():
    c, args = fixture(1)
    c['variables']['query'] = [1, 1]
    c['inputs'] = c['inputs'][1:]
    for row in c['inputs']:
        if row['kind'] == 'tensor':
            row['shape'] = [1 if x == 'query' else x for x in row['shape']]
    assert pack_torch_request_joined(c, args[1:]) == pack_torch_request(c, args[1:])


@pytest.mark.parametrize('bad', ['count', 'bool', 'scalar', 'bounds', 'dtype',
    'rank', 'shape', 'symbol', 'noncontiguous', 'grad', 'empty'])
def test_same_invalid_inputs_rejected(bad):
    c, args = fixture(3)
    if bad == 'count': args.pop()
    elif bad == 'bool': args[0] = True
    elif bad == 'scalar': args[2] = 512
    elif bad == 'bounds': c['variables']['query'] = [1, 2]
    elif bad == 'dtype': args[3] = args[3].to(torch.float64)
    elif bad == 'rank': args[3] = args[3].reshape(-1)
    elif bad == 'shape': args[4] = args[4][:, :, :10, :]
    elif bad == 'symbol': args[0] = 4
    elif bad == 'noncontiguous': args[3] = args[3].transpose(1, 2)
    elif bad == 'grad': args[3].requires_grad_(True)
    elif bad == 'empty': args[1] = torch.empty((1, 0), dtype=torch.int64)
    for fn in (pack_torch_request, pack_torch_request_joined):
        with pytest.raises(ValueError): fn(c, args)
