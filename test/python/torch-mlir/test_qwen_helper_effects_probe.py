import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

TOOLS=Path(__file__).resolve().parents[3]/'tools/qwen3'
spec=importlib.util.spec_from_file_location('qwen_helper_effects_probe',TOOLS/'probe_helper_effects_arm.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def response(rows=1):
    data=bytearray(128+4*3072*rows)
    struct.pack_into('<QI4xQQ',data,0,123,128,456789,0)
    data[32:40]=bytes.fromhex('82c100efc03f2050')
    return data


def test_response_carries_complete_actual_evidence():
    assert module.parse_response(response(),128,0)['processor_cycles']==456789


@pytest.mark.parametrize('offset,value',[(0,0),(8,0),(12,1),(16,0),(24,1),(32,0),(40,1),(127,1)])
def test_response_rejects_changed_extent_timing_counter_code_and_reserved(offset,value):
    data=response()
    if offset in (0,8,16):data[offset:offset+(4 if offset==8 else 8)]=bytes(4 if offset==8 else 8)
    else:data[offset]=value
    with pytest.raises(ValueError):module.parse_response(data,128,0)


@pytest.mark.parametrize('delta',[-1,1])
def test_response_rejects_partial_or_extra_FP_word(delta):
    data=bytes(response())
    data=data[:delta] if delta<0 else data+bytes(delta)
    with pytest.raises(ValueError):module.parse_response(data,128,0)


@pytest.mark.parametrize('rows',[2,13,32])
def test_dynamic_response_binds_all_rows_to_exact_output_extent(rows):
    data=response(rows)
    assert module.parse_response(data,128,0,rows)['processor_cycles']==456789
    with pytest.raises(ValueError):module.parse_response(data,128,0,rows-1)
    with pytest.raises(ValueError):module.parse_response(data[:-4],128,0,rows)


@pytest.mark.parametrize('rows',[0,33,True,2.0])
def test_dynamic_response_rejects_out_of_contract_rows(rows):
    with pytest.raises(ValueError):module.parse_response(response(),128,0,rows)


def test_patterns_separate_finite_oracle_from_original_special_value_contract():
    rows=list(module.patterns(np))
    assert len(rows)==6 and len({r[0] for r in rows})==6
    for _,x,other,finite in rows:
        assert x.shape==other.shape==(3072,) and x.dtype==other.dtype==np.float32
        assert bool(np.isfinite(x).all() and np.isfinite(other).all())==finite
    special=rows[-1][1].view(np.uint32)
    assert {0x7f800000,0xff800000,0x7fc00001,0xffc00123,0x7f800001,0xff800001}<=set(map(int,special))


@pytest.mark.parametrize('extent',[2,13,32])
def test_dynamic_patterns_expose_cross_row_stride_and_reuse_errors(extent):
    for name,x,other,finite in module.patterns(np,extent):
        assert x.shape==other.shape==(3072*extent,) and x.flags.c_contiguous and other.flags.c_contiguous
        assert bool(np.isfinite(x).all() and np.isfinite(other).all())==finite
        assert not np.array_equal(x[:3072].view(np.uint32),x[3072:6144].view(np.uint32))
        if name in ('finite-range','finite-adjacent','finite-seeded'):
            assert len({row.tobytes() for row in x.reshape(extent,3072)})==extent
            assert len({row.tobytes() for row in other.reshape(extent,3072)})==extent
