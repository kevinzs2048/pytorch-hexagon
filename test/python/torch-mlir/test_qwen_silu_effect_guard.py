from qwen3.compiler_dependency import compiler_root
import copy
from pathlib import Path

import pytest

from hexagon_compiled_artifact_cache import digest_file
from hexagon_silu_helper_effects import guard_wrapper,guard_caller,ORIGINAL_HEADER_SHA256,CONTRACT

RUNTIME=compiler_root()/'tools/qwen3/runtime'


def fixture_caller(tmp_path):
    wrapper=tmp_path/'qwen_captured_q1_wrapper.cpp';wrapper.write_bytes((RUNTIME/'reference-wrapper.cpp').read_bytes())
    header=tmp_path/'qwen3_sleef_exp_candidate.h';header.write_bytes((RUNTIME/header.name).read_bytes())
    return dict(query_specialization=1,input_bytes=234885520,output_bytes=235488768,
        wrapper=dict(path=str(wrapper),sha256=digest_file(wrapper)),
        runtime_header=dict(path=str(header),sha256=digest_file(header)))


def test_guard_precedes_original_model_call_and_preserves_all_original_source():
    original=(RUNTIME/'reference-wrapper.cpp').read_text();changed=guard_wrapper(original)
    include='#include "silu_effects_guard.h"\n'
    check='  if (!qwen3_silu_effects_runtime_matches()) return 6;\n'
    assert changed.count(include)==changed.count(check)==1
    assert changed.replace(include,'',1).replace(check,'',1)==original
    entry=changed.index('extern "C" int hexagon_forward_buffers(')
    assert entry<changed.index(check)<changed.index('_mlir_ciface_Qwen3DynamicBoundProbe(',entry)


def test_caller_binds_guard_bytes_and_exact_original_header(tmp_path):
    caller=fixture_caller(tmp_path);old=caller['wrapper']['sha256']
    assert caller['runtime_header']['sha256']==ORIGINAL_HEADER_SHA256
    guard_caller(caller,RUNTIME/'silu_effects_guard.h')
    assert caller['wrapper']['sha256']!=old and caller['silu_effects']['contract']==CONTRACT
    assert caller['silu_effects']['guard_header']['sha256']==digest_file(RUNTIME/'silu_effects_guard.h')
    with pytest.raises(ValueError,match='Fresh guard destination'):
        guard_caller(caller,RUNTIME/'silu_effects_guard.h')


@pytest.mark.parametrize('key,value',[('query_specialization',None),('query_specialization',2),
    ('input_bytes',234885568),('output_bytes',235488764)])
def test_guard_rejects_other_shapes_packets_or_scalar_representation(tmp_path,key,value):
    caller=fixture_caller(tmp_path);caller[key]=value
    with pytest.raises(ValueError,match='Exact original query1'):
        guard_caller(caller,RUNTIME/'silu_effects_guard.h')


@pytest.mark.parametrize('name',['wrapper','runtime_header'])
def test_guard_rejects_modified_original_math_or_source(tmp_path,name):
    caller=fixture_caller(tmp_path);Path(caller[name]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        guard_caller(caller,RUNTIME/'silu_effects_guard.h')


def test_wrapper_rejects_second_model_call_or_preexisting_guard():
    text=(RUNTIME/'reference-wrapper.cpp').read_text()
    with pytest.raises(ValueError,match='unique guarded'):
        guard_wrapper(guard_wrapper(text))
    text=text.replace('  *kernel_us = 0;\n','  *kernel_us = 0;\n_mlir_ciface_Qwen3DynamicBoundProbe();\n',1)
    with pytest.raises(ValueError,match='One original model'):
        guard_wrapper(text)


def test_multi_caller_binds_same_original_math_and_guard_with_a_distinct_contract(tmp_path):
    from hexagon_silu_helper_effects import DYNAMIC_CONTRACT
    caller=fixture_caller(tmp_path)
    caller.update(query_specialization=None,input_bytes=235012880,contract=dict(variables={'query':[2,32]}))
    original=Path(caller['wrapper']['path']).read_text()
    guard_caller(caller,RUNTIME/'silu_effects_guard.h',multi=True)
    assert caller['silu_effects']['contract']==DYNAMIC_CONTRACT
    assert caller['silu_effects']['unchanged_FP_helper_sha256']==ORIGINAL_HEADER_SHA256
    assert Path(caller['wrapper']['path']).read_text()==guard_wrapper(original)
    assert caller['silu_effects']['guard_header']['sha256']==digest_file(RUNTIME/'silu_effects_guard.h')


@pytest.mark.parametrize('bounds',[[1,1],[2,2],[3,13],[2,33]])
def test_multi_guard_rejects_other_domains_even_with_matching_packet_capacity(tmp_path,bounds):
    caller=fixture_caller(tmp_path)
    caller.update(query_specialization=None,input_bytes=235012880,contract=dict(variables={'query':bounds}))
    original=Path(caller['wrapper']['path']).read_bytes()
    with pytest.raises(ValueError):guard_caller(caller,RUNTIME/'silu_effects_guard.h',multi=True)
    assert Path(caller['wrapper']['path']).read_bytes()==original and not (tmp_path/'silu_effects_guard.h').exists()
