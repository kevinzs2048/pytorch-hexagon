"""Authenticate the same actual partial-order PV compiler for native reuse."""
import json
from pathlib import Path
from run_regression import sha

CONTRACT='original_Qwen_query2to32_PV_partial352_336_336_ordered_register8_v1'


def checked(ref):
    path=Path(ref['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or sha(path)!=ref['sha256']:
        raise ValueError('Original retained dynamic PV evidence changed')
    return path


def checked_references(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and 'sha256' in value:
            path=checked(value)
            if 'bytes' in value and path.stat().st_size!=value['bytes']:
                raise ValueError('Retained dynamic PV evidence size changed')
        for item in value.values():checked_references(item)
    elif isinstance(value,list):
        for item in value:checked_references(item)


def require_compiler(sdk,diagnostics):
    if (sdk['status']!='isolated_current_dynamic_QK_PV_register_model_candidate_SDK_built_NOT_DSP_or_model_acceptance'
            or sdk['query_bounds']!=[2,32] or sdk['original_dynamic_model_arguments']!=65
            or sdk['DSP_calls'] or sdk['model_inference_acceptance'] or sdk['production_model_acceptance']
            or sdk['diagnostic_instrumentation'] or sdk['only_change']!='post_original_dynamic_QK_per_partial_PV_register8_model_candidate_NOT_accepted'
            or not all(sdk[k] for k in ('unmodified_SDK_replay_byte_identical','original_link_inputs_unchanged','original_SDK_unchanged'))
            or len(sdk['commands'])!=4 or any(type(row['returncode']) is not int or row['returncode'] for row in sdk['commands'])):
        raise ValueError('Exact original QK plus unaccepted ordered per-partial PV compiler required')
    for key in ('source','dynamic_QK_invariant_source','loop_profile_source','candidate','operator_proof','original_QK_sdk_receipt'):
        checked(sdk[key])
    original=json.loads(checked(sdk['original_QK_sdk_receipt']).read_text())
    if original['query_bounds']!=[2,32] or original['status']!='isolated_dynamic_QK_invariant_vector_and_CSE_SDK_built_NOT_model_or_DSP':
        raise ValueError('Original already accepted dynamic QK compiler is mandatory')
    proof=json.loads(checked(sdk['operator_proof']).read_text())
    if (proof['status']!='actual_dynamic_PV_ABBA120_all_FP32_words_full_FNV_and_normal_owners_independently_revisited'
            or type(proof['DSP_calls']) is not int or proof['DSP_calls']!=120 or proof['new_DSP_calls']
            or proof['model_inference_acceptance'] or proof['performance_acceptance'] or len(proof['cases'])!=30):
        raise ValueError('Actual independently reviewed operator120 is mandatory before native candidate reuse')
    required=['QWEN_DYNAMIC_QK_INVARIANT_VECTOR matched=28',
        'QWEN_DYNAMIC_PV_REGISTER verified=84 lanes=8 per_partial_k_order=original partial_merges=original no_fast_math=1',
        'QWEN_DYNAMIC_PV_REGISTER_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE']
    if any(diagnostics.count(marker)!=1 for marker in required) or 'QWEN_DYNAMIC_PV_STOP_NO_MODEL_CODE' in diagnostics:
        raise ValueError('All28 QK and all84 original partial-order PV proofs must survive actual model codegen')
