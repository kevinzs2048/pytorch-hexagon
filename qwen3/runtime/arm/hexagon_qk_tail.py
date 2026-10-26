"""Reuse only target-proved q1 QK code above the authenticated original PV code."""
import copy
import json
from pathlib import Path
import shutil
from hexagon_compiled_artifact_cache import canonical,digest_file
from hexagon_zero_tail import checked,normal_owner

CONTRACT='original_q1_QK_actual_positive_zero_keys_and_seed_identical_column_recurrence_v1'
INPUTS={'qk_tail_code','qk_tail_code_owner','qk_tail_link','qk_tail_link_owner',
        'qk_tail_model_review','qk_tail_model_review_owner',
        'qk_tail_native_review','qk_tail_native_review_owner','qk_tail_runtime'}


def validate_link_runtime(runtime,link_row,base_link_row):
    link=json.loads(checked(link_row).read_text())
    code=json.loads(checked(link['model_code']).read_text())
    checked(runtime);checked(link['runtime_source']);checked(code['primary'])
    if (link['status']!='actual_q1_runtime_extent_model_linked_DIAGNOSTIC_NOT_model_acceptance'
            or link.get('QK_guard_candidate') is not True or link['query_bounds']!=[1,1]
            or runtime['sha256']!=link['runtime']['sha256']
            or link['QK_current_accepted_baseline_link']!=base_link_row
            or code['status']!='one_existing_q1_QK_PRIMARY_compiled_constants_identical_NOT_model_acceptance'
            or code.get('QK_guard_candidate') is not True
            or code['model_PRIMARY_compilations']!=1 or code['query_bounds']!=[1,1]):
        raise ValueError('Exact target-proved q1 QK runtime above this accepted PV baseline required')
    return code,link


def authenticate(inputs):
    for key in INPUTS:checked(inputs[key])
    code,link=validate_link_runtime(inputs['qk_tail_runtime'],inputs['qk_tail_link'],inputs['zero_tail_link'])
    if link['model_code']!=inputs['qk_tail_code']:raise ValueError('QK link belongs to another primary producer')
    normal_owner(inputs['qk_tail_code_owner'],code['source_revision'])
    normal_owner(inputs['qk_tail_link_owner'],link['source_revision'])
    sdk=json.loads(checked(code['SDK']).read_text())
    for name in ('candidate','CPP_source','model_hook','integer_guard','physical_review','all28_review'):
        checked(sdk[name])
    if (sdk['status']!='q1_QK_candidate_SDK_built_NOT_model_acceptance' or sdk['preflight_only']
            or not sdk['base_replay_byte_identical'] or not sdk['original_PV_guard_objects_retained']):
        raise ValueError('Same target/all28-proved QK compiler and unchanged original PV objects required')
    if link['runtime_source']!=sdk['integer_guard']:raise ValueError('Runtime must be the exact target-proved integer guard')
    model=json.loads(checked(inputs['qk_tail_model_review']).read_text())
    native=json.loads(checked(inputs['qk_tail_native_review']).read_text())
    for review,key in ((model,'model'),(native,'native')):
        normal_owner(inputs['qk_tail_'+key+'_review_owner'],review['source_revision'])
        if review['new_DSP_calls']!=0 or review['model_acceptance'] or review['performance_acceptance']:
            raise ValueError('Independent normally closed diagnostic evidence required, never borrowed model acceptance')
    if (model['status']!='actual_q1_zero_tail_original24_full_words_raw_RPC_FNV_ABBA_and_owner_independently_revisited_DIAGNOSTIC'
            or model['DSP_calls']!=24 or model['build']!=inputs['qk_tail_link']
            or native['status']!='actual_q1_runtime_extent_native16_all_full_words_tokens_KV_3FNV_raw_RPC_and_owner_independently_revisited_DIAGNOSTIC'
            or native['DSP_calls']!=16 or native['mode']!='candidate'
            or native['q1_review']!=inputs['qk_tail_model_review'] or native['generation_regression'] is not True):
        raise ValueError('Full model24 and native16 must bind this exact QK primary and library')
    diagnostics=checked(code['compiler_diagnostics'])
    return code,sdk,diagnostics


def stage(inputs,objects,bounds):
    if tuple(bounds)!=(1,1):raise ValueError('QK-tail reuse applies only to the existing q1 category')
    code,sdk,diagnostics=authenticate(inputs)
    base=json.loads(checked(inputs['zero_tail_code']).read_text())
    folder=Path(objects);old=json.loads((folder/'result.json').read_text())
    if (old.get('zero_tail_producer')!=inputs['zero_tail_code']
            or old['objects'][0]['sha256']!=base['primary']['sha256']
            or digest_file(old['objects'][0]['path'])!=base['primary']['sha256']
            or old['objects'][1:]!=code['unchanged_constant_objects']
            or old['inputs']['linalg']['sha256']!=code['actual_linalg']['sha256']):
        raise ValueError('Authenticated original PV primary, full preparation and identical constants must precede QK reuse')
    producer=json.loads(checked(code['original_model']).read_text())
    for name in ('overrides','option_reconciliation','expected_objects_result','triton_python'):
        if old[name]!=producer[name]:raise ValueError('Original compiler options or constant reference changed')
    (folder/'before-qk-tail-result.json').write_bytes((folder/'result.json').read_bytes())
    shutil.copyfile(checked(code['primary']),folder/'dynamic-hexagon-0.o')
    shutil.copyfile(diagnostics,folder/'compiler-diagnostics.txt')
    result=copy.deepcopy(old)
    result['objects'][0].update(sha256=code['primary']['sha256'],bytes=(folder/'dynamic-hexagon-0.o').stat().st_size)
    result['inputs']['compiler']=sdk['candidate']
    result['compiler_diagnostics']['sha256']=code['compiler_diagnostics']['sha256']
    result.update(qk_tail_producer=inputs['qk_tail_code'],qk_tail_base_producer=inputs['zero_tail_code'],
        new_model_object_codegen_invocations=0)
    (folder/'result.json').write_bytes(canonical(result)+b'\n')
    (folder/'qk-tail-reuse-result.json').write_bytes(canonical(dict(
        status='exact_q1_QK_tail_object_staged_NOT_model_or_delivery_acceptance',contract=CONTRACT,
        producer=inputs['qk_tail_code'],base_producer=inputs['zero_tail_code'],
        model_review=inputs['qk_tail_model_review'],native_review=inputs['qk_tail_native_review'],
        new_model_object_codegen_invocations=0,DSP_calls=0))+b'\n')
