"""Canonical diagnostics agree with the established byte ABI and reject damage."""
import hashlib
import numpy as np
import pytest
from tools.qwen3.packet_trace import describe_packet,describe_bytes
from qcom_hexagon_backend.backend.hexagon_buffer_io import make_layout,pack_inputs
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import MAGIC
import struct


def fixture():
    values=[np.array([[0.0,-0.0,2.5]],dtype='<f4'),np.array([4,8],dtype='<i8')]
    layout=make_layout([dict(dtype='float32',shape=[1,3]),dict(dtype='int64',shape=[2])],
                       [dict(dtype='float32',shape=[1,3])])
    return layout,values


def test_streamed_digest_and_snapshot_equal_original_pack_inputs(tmp_path):
    layout,values=fixture()
    payload=MAGIC+struct.pack('<q',2)+pack_inputs(layout,[v.tobytes() for v in values])
    target=tmp_path/'native.bin'
    row=describe_packet(layout,'inputs',values,query=2,save=target)
    assert target.read_bytes()==payload
    assert row['sha256']==hashlib.sha256(payload).hexdigest()
    parsed,arrays=describe_bytes(layout,'inputs',payload,query=2)
    assert parsed==row
    assert all(np.array_equal(a.view('u1'),b.view('u1')) for a,b in zip(values,arrays,strict=True))


def test_zero_sign_is_distinguished_without_changing_numeric_tolerance():
    layout,values=fixture()
    original=describe_packet(layout,'outputs',values[:1])
    changed=[np.array([[0.0,0.0,2.5]],dtype='<f4')]
    assert np.array_equal(values[0],changed[0])
    assert original['sha256']!=describe_packet(layout,'outputs',changed)['sha256']


@pytest.mark.parametrize('fault',['padding','header','length'])
def test_corrupted_transport_is_rejected(fault):
    layout,values=fixture()
    payload=bytearray(MAGIC+struct.pack('<q',2)+pack_inputs(layout,[v.tobytes() for v in values]))
    if fault=='padding':payload[16+12]=1
    elif fault=='header':payload[0]^=1
    else:payload.pop()
    with pytest.raises(ValueError):describe_bytes(layout,'inputs',bytes(payload),query=2)


@pytest.mark.parametrize('fault',['dtype','shape','count'])
def test_wrong_reference_representation_is_rejected(fault):
    layout,values=fixture()
    if fault=='dtype':values[0]=values[0].astype('f8')
    elif fault=='shape':values[0]=values[0].reshape(3)
    else:values=values[:1]
    with pytest.raises(ValueError):describe_packet(layout,'inputs',values,query=2)
