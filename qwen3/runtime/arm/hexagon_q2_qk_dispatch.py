"""Explicit reuse of the target-checked Q2 QK runtime dispatch primary.

This is an experimental object-reuse gate. Model packet diagnostics do not
replace fresh ARM controls, native generation or full release acceptance.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from hexagon_compiled_artifact_cache import canonical, digest_file
from hexagon_zero_tail import checked, normal_owner

CONTRACT = 'original_Qwen_q2to32_QK_runtime_M8to32_exact_register_else_original_FMA_v1'
INPUTS = {'q2_qk_dispatch_code','q2_qk_dispatch_code_owner',
          'q2_qk_dispatch_link','q2_qk_dispatch_link_owner',
          'q2_qk_dispatch_model_review','q2_qk_dispatch_model_review_owner'}


def authenticate(inputs):
    for name in INPUTS:
        checked(inputs[name])
    code = json.loads(checked(inputs['q2_qk_dispatch_code']).read_text())
    link = json.loads(checked(inputs['q2_qk_dispatch_link']).read_text())
    review = json.loads(checked(inputs['q2_qk_dispatch_model_review']).read_text())
    for name, proof in (('code',code),('link',link),('model_review',review)):
        normal_owner(inputs['q2_qk_dispatch_'+name+'_owner'],proof['source_revision'])
        owner = json.loads(checked(inputs['q2_qk_dispatch_'+name+'_owner']).read_text())
        source = checked(proof['source'])
        root = Path(proof['source_root'])
        output = (Path(inputs['q2_qk_dispatch_'+name]['path']) if name=='model_review' else
                  Path(inputs['q2_qk_dispatch_'+name]['path']).parent)
        if (owner['source_root'] != str(root) or owner['command'][1] != str(source)
                or owner['command'][owner['command'].index('--output')+1] != str(output)):
            raise ValueError('Exact normal producer source and output binding required')
        blob = subprocess.check_output(['git','-C',str(root),'show',
                proof['source_revision']+':'+str(source.relative_to(root))])
        if hashlib.sha256(blob).hexdigest() != proof['source']['sha256']:
            raise ValueError('Recorded Q2 producer must match its actual committed source')
    if (code['status'] != 'one_existing_Q2_runtime_dispatch_PRIMARY_compiled_constants_identical_NOT_model_acceptance'
            or code['model_PRIMARY_compilations'] != 1 or code['query_bounds'] != [2,32]
            or code['DSP_calls'] or code['model_acceptance'] or code['performance_acceptance']
            or code['options']['addFastMath'] != 'False'
            or link['status'] != 'actual_Q2_runtime_dispatch_model_linked_NOT_device_or_model_acceptance'
            or link['query_bounds'] != [2,32] or link['DSP_calls']
            or link['model_inference_acceptance'] or link['performance_acceptance']
            or link['code'] != inputs['q2_qk_dispatch_code']
            or link['code_controller'] != inputs['q2_qk_dispatch_code_owner']
            or link['candidate_primary'] != code['primary']
            or not link['accepted_q2_replay_byte_identical']
            or not link['original_caller_and_runtime_objects_unchanged']):
        raise ValueError('Exact normally compiled and linked one-primary strict Q2 candidate required')
    if (review['status'] != 'actual_Q2_model_ABBA20_complete_logits_56KV_3FNV_and_normal_owners_independently_revisited_NOT_generation_acceptance'
            or review['link'] != inputs['q2_qk_dispatch_link']
            or review['DSP_calls'] != 20 or review['new_DSP_calls'] or review['model_PRIMARY_compilations']
            or review['complete_model_packet_numerical_comparison'] is not True
            or review['original_ARM_W8A8_queries'] != [8,7,6]
            or review['derived_DSP_reference_queries'] != [2,32]
            or [r['query'] for r in review['cases']] != [8,7,6,2,32]
            or review['unchanged_q1_library'] != link['unchanged_q1_library']
            or review['model_inference_acceptance'] or review['generation_regression'] or review['performance_acceptance']):
        raise ValueError('Independent full logits/56KV ABBA20 must bind this exact candidate without borrowing model acceptance')
    manifest = json.loads(checked(review['manifest']).read_text())
    if (manifest['libraries'] != dict(baseline=link['original_library'],candidate=link['library'])
            or manifest['link_controller'] != inputs['q2_qk_dispatch_link_owner']):
        raise ValueError('Actually tested libraries and normal linker owner must match')
    checked(code['primary']); checked(code['actual_linalg']); checked(link['library'])
    checked(link['original_library']); checked(link['original_objects'])
    sdk = json.loads(checked(code['SDK']).read_text())
    physical = json.loads(checked(code['physical_review']).read_text())
    if (sdk['status'] != 'Q2_dispatch_target_proved_model_SDK_built_NOT_model_acceptance'
            or not sdk['unmodified_SDK_replay_byte_identical'] or not sdk['original_PV_objects_unchanged']
            or sdk['physical_review'] != code['physical_review']
            or physical['status'] != 'actual_Q2_runtime_dispatch_ABBA64_all_FP32_words_full_FNV_and_normal_owners_independently_revisited_NOT_model_acceptance'
            or physical['DSP_calls'] != 64 or physical['new_DSP_calls'] or physical['model_inference_acceptance']
            or link['unchanged_constant_objects'] != code['unchanged_constant_objects']):
        raise ValueError('Original target64 dispatch proof, unchanged PV and constants required')
    checked(sdk['candidate']); checked(sdk['dispatch']); checked(sdk['proven_recurrence'])
    diagnostics = checked(code['compiler_diagnostics'])
    return code, link, sdk, diagnostics


def stage(inputs, objects, bounds):
    if tuple(bounds) != (2,32):
        raise ValueError('Q2 runtime dispatch applies only to the existing q2..32 category')
    code, link, sdk, diagnostics = authenticate(inputs)
    folder = Path(objects)
    old = json.loads((folder/'result.json').read_text())
    base = json.loads(checked(link['original_objects']).read_text())
    reuse = json.loads((folder/'reuse-result.json').read_text())
    if (reuse['status'] != 'authenticated_exact_compiler_stage_objects_reused_NOT_model_acceptance'
            or reuse['query_bounds'] != [2,32] or reuse['new_machine_code_compilations']
            or old['objects'][0]['sha256'] != base['objects'][0]['sha256']
            or digest_file(old['objects'][0]['path']) != base['objects'][0]['sha256']
            or old['objects'][1:] != code['unchanged_constant_objects']
            or old['inputs']['linalg']['sha256'] != code['actual_linalg']['sha256']):
        raise ValueError('Successful exact current accepted PV q2 stage must precede QK dispatch reuse')
    for name in ('overrides','option_reconciliation','expected_objects_result','triton_python'):
        if old[name] != base[name]:
            raise ValueError('Original q2 compiler options, constant references or Python changed')
    (folder/'before-q2-qk-dispatch-result.json').write_bytes((folder/'result.json').read_bytes())
    shutil.copyfile(checked(code['primary']),folder/'dynamic-hexagon-0.o')
    shutil.copyfile(diagnostics,folder/'compiler-diagnostics.txt')
    result = copy.deepcopy(old)
    result['objects'][0].update(sha256=code['primary']['sha256'],bytes=(folder/'dynamic-hexagon-0.o').stat().st_size)
    result['inputs']['compiler'] = sdk['candidate']
    result['compiler_diagnostics']['sha256'] = code['compiler_diagnostics']['sha256']
    result.update(q2_qk_dispatch_producer=inputs['q2_qk_dispatch_code'],
                  q2_qk_dispatch_tested_link=inputs['q2_qk_dispatch_link'],new_model_object_codegen_invocations=0)
    (folder/'result.json').write_bytes(canonical(result)+b'\n')
    (folder/'q2-qk-dispatch-reuse-result.json').write_bytes(canonical(dict(
        status='exact_Q2_QK_runtime_dispatch_object_staged_NOT_model_or_delivery_acceptance',
        contract=CONTRACT,producer=inputs['q2_qk_dispatch_code'],tested_link=inputs['q2_qk_dispatch_link'],
        model_review=inputs['q2_qk_dispatch_model_review'],runtime_query_bounds=[8,32],
        original_fallback_query_bounds=[2,7],unchanged_q1_library=link['unchanged_q1_library'],
        new_model_object_codegen_invocations=0,DSP_calls=0))+b'\n')
