"""Bind scoped SiLU effects to the unchanged helper and guarded q1 caller."""
from pathlib import Path

from hexagon_runtime.hexagon_compiled_artifact_cache import digest_file

ORIGINAL_HEADER_SHA256='0e4c1bcc6df416dc3d26329308a9074aa1ddb741a01d5d50a5e95d8ec9892e29'
CONTRACT='original_Qwen_query1_silu_readnone_nounwind_willreturn_v1'
DYNAMIC_CONTRACT='original_Qwen_query2_to32_silu_readnone_nounwind_willreturn_v1'


def guard_wrapper(text):
    entry='extern "C" int hexagon_forward_buffers('
    guard='  *kernel_us = 0;\n'
    if (text.count(entry)!=1 or text.count('#include "qwen3_sleef_exp_candidate.h"')!=1
            or text.count('#pragma clang fp contract(off)')!=1
            or 'qwen3_silu_effects_runtime_matches' in text):
        raise ValueError('Original unique guarded buffer caller and FP helper required')
    pos=text.index(entry)
    if text[pos:].count(guard)!=1 or text[pos:].count('_mlir_ciface_Qwen3DynamicBoundProbe(')!=1:
        raise ValueError('One original model call and kernel initialization required')
    tail=text[pos:].replace(guard,guard+'  if (!qwen3_silu_effects_runtime_matches()) return 6;\n',1)
    return text[:pos]+'#include "silu_effects_guard.h"\n'+tail


def guard_caller(caller,guard_header,*,multi=False):
    if multi:
        from hexagon_qwen_shape_policy import contract_query_bounds
        if contract_query_bounds(caller['contract'])!=(2,32):
            raise ValueError('Exact original query2..32 captured shape contract required')
    if (caller['query_specialization']!=(None if multi else 1)
            or caller['input_bytes']!=(235012880 if multi else 234885520)
            or caller['output_bytes']!=235488768
            or caller['runtime_header']['sha256']!=ORIGINAL_HEADER_SHA256):
        raise ValueError('Exact original '+('query2..32' if multi else 'query1')+' packet and original FP helper required')
    wrapper=Path(caller['wrapper']['path']);original=wrapper.read_text()
    if digest_file(wrapper)!=caller['wrapper']['sha256'] or digest_file(caller['runtime_header']['path'])!=ORIGINAL_HEADER_SHA256:
        raise ValueError('Original caller/helper changed')
    target=wrapper.parent/'silu_effects_guard.h'
    if target.exists():raise ValueError('Fresh guard destination required')
    changed=guard_wrapper(original)
    target.write_bytes(Path(guard_header).read_bytes());wrapper.write_text(changed)
    caller['wrapper']['sha256']=digest_file(wrapper)
    caller['silu_effects']=dict(contract=DYNAMIC_CONTRACT if multi else CONTRACT,unchanged_FP_helper_sha256=ORIGINAL_HEADER_SHA256,
        guard_header=dict(path=str(target),sha256=digest_file(target)),
        actual_fmaf_entire_code_bytes='82c100efc03f2050',runtime_mismatch_returncode=6,
        FP_contract='original_unconstrained_no_errno_or_exception_status_observation')
    return caller
