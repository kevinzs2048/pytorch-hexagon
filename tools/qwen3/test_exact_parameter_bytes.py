"""Complete-byte parameter admission, including mutations bypassing versions."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen3/runtime/arm'))
from hexagon_dynamo_parameters import spec
from hexagon_dynamic_module_constants import dynamic_tensor_digest
from exact_parameter_bytes import ExactParameterBytePool,raw_view


def binding(value):
    return SimpleNamespace(input_count=2,retained=[1],records=[dict(input_index=0,
        spec=spec(value),sha256=dynamic_tensor_digest(value),bytes=value.numel()*value.element_size())])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.float16,torch.bfloat16,
    torch.int8,torch.int32,torch.int64,torch.bool])
def test_all_logical_bytes_and_compiler_sha_identity(dtype):
    value=torch.arange(257).to(dtype).reshape(1,257)
    raw=raw_view(value)
    assert hashlib.sha256(raw.tobytes()).hexdigest()==dynamic_tensor_digest(value)
    bound=binding(value);before=repr(bound.records);pool=ExactParameterBytePool();guard=pool.bind(bound)
    sentinel=object();assert guard.check([value,sentinel])==[sentinel]
    assert repr(bound.records)==before
    assert pool.snapshot_bytes==raw.nbytes
    # Sharing a second real graph does not duplicate private byte snapshots.
    assert pool.bind(binding(value)).check([value,sentinel])==[sentinel]
    assert pool.snapshot_bytes==raw.nbytes


@pytest.mark.parametrize('where',[0,128,256])
def test_data_writes_without_version_change_rejected_before_execution(where):
    value=torch.arange(257,dtype=torch.int32);original=value.clone()
    pool=ExactParameterBytePool();guard=pool.bind(binding(value))
    guard.check([value,None]);version=value._version
    value.data[where]^=1
    assert value._version==version
    with pytest.raises(ValueError,match='Compiled parameter changed'):guard.check([value,None])
    value.data.copy_(original);assert guard.check([value,None])==[None]


def test_initial_mutation_cannot_establish_a_new_reference():
    value=torch.arange(64,dtype=torch.uint8);bound=binding(value)
    value.data[32]=200
    pool=ExactParameterBytePool();guard=pool.bind(bound)
    with pytest.raises(ValueError,match='Compiled parameter changed'):guard.check([value,None])
    assert pool.snapshot_bytes==0


def test_rank_zero_signed_zero_and_nan_payloads_are_exact_bytes():
    value=torch.tensor([0x80000000],dtype=torch.uint32).view(torch.float32).reshape(())
    pool=ExactParameterBytePool();guard=pool.bind(binding(value));guard.check([value,None])
    value.data.view(torch.int32).fill_(0)
    with pytest.raises(ValueError):guard.check([value,None])
    nan=torch.tensor([0x7fc00001],dtype=torch.int32).view(torch.float32)
    guard=pool.bind(binding(nan));guard.check([nan,None])
    nan.data.view(torch.int32).fill_(0x7fc00002)
    with pytest.raises(ValueError):guard.check([nan,None])


def test_strided_alias_checks_logical_bytes_and_storage_offset():
    storage=torch.arange(128,dtype=torch.int32);value=storage[5:125:3]
    assert not value.is_contiguous() and value.storage_offset()==5
    pool=ExactParameterBytePool();guard=pool.bind(binding(value));guard.check([value,None])
    storage.data[6]^=1  # Outside the parameter's logical values, as in old SHA.
    guard.check([value,None])
    storage.data[65]^=1
    with pytest.raises(ValueError):guard.check([value,None])


def test_embedded_zero_bytes_tail_and_reference_immutability():
    value=torch.zeros(259,dtype=torch.uint8);value[-1]=23
    pool=ExactParameterBytePool();guard=pool.bind(binding(value));guard.check([value,None])
    value.data[255]=42
    with pytest.raises(ValueError):guard.check([value,None])
    value.data[255]=0;guard.check([value,None])
    assert all(type(row) is bytes for row in pool._bytes.values())


def test_changed_shape_stride_dtype_and_arity_rejected():
    value=torch.arange(64,dtype=torch.int32).reshape(8,8)
    guard=ExactParameterBytePool().bind(binding(value));guard.check([value,None])
    for changed in (value.reshape(64),value.T,value.float()):
        with pytest.raises(ValueError,match='metadata changed'):guard.check([changed,None])
    with pytest.raises(ValueError,match='arity'):guard.check([value])


def test_mutable_binding_records_cannot_change_the_expected_reference():
    value=torch.arange(32,dtype=torch.int32);bound=binding(value)
    guard=ExactParameterBytePool().bind(bound);guard.check([value,None])
    value.data.add_(7);bound.records[0]['sha256']=dynamic_tensor_digest(value)
    with pytest.raises(ValueError):guard.check([value,None])


def test_empty_byte_comparison_with_explicitly_pinned_empty_metadata():
    value=torch.empty((0,3),dtype=torch.float32)
    bound=SimpleNamespace(input_count=2,retained=[1],records=[dict(input_index=0,
        spec=spec(value),sha256=hashlib.sha256(b'').hexdigest(),bytes=0)])
    guard=ExactParameterBytePool().bind(bound);assert guard.check([value,None])==[None]
