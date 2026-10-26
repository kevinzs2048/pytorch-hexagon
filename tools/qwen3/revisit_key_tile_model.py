"""Bind a completed q1 runtime ABBA8 to its original complete ARM packet."""
import argparse
import json
from pathlib import Path
import statistics

from build import sha, write, entry
from run_integer_regressions import candidate_libraries
from key_tile_model_contract import KIND, MODEL_STATUS, RUN_STATUS, CHANGE, validate_key_tile
from key_forward_model_contract import KIND as FORWARD_KIND, MODEL_STATUS as FORWARD_MODEL_STATUS, RUN_STATUS as FORWARD_RUN_STATUS, CHANGE as FORWARD_CHANGE, validate_key_forward
CONTRACTS = {KIND: dict(model_status=MODEL_STATUS, run_status=RUN_STATUS, change=CHANGE),
    FORWARD_KIND: dict(model_status=FORWARD_MODEL_STATUS, run_status=FORWARD_RUN_STATUS, change=FORWARD_CHANGE)}
from ordered_pv_model_contract import KIND as PV_KIND, MODEL_STATUS as PV_MODEL_STATUS, RUN_STATUS as PV_RUN_STATUS, CHANGE as PV_CHANGE, require_reference as require_PV_reference, require_compiler as require_PV_compiler
CONTRACTS[PV_KIND]=dict(model_status=PV_MODEL_STATUS,run_status=PV_RUN_STATUS,change=PV_CHANGE)
from dynamic_qk_invariant_contract import KIND as INVARIANT_KIND, SDK_STATUS as INVARIANT_SDK_STATUS, MODEL_STATUS as INVARIANT_MODEL_STATUS, RUN_STATUS as INVARIANT_RUN_STATUS, CHANGE as INVARIANT_CHANGE, require_reference, require_compiler
from build_model_empty_tensor_cse import same_linalg_input
from bounded_loader_policy import verify_native_layout, KIND as LOADER_KIND, MODE

KINDS = {
    'fixed512-copy-q1': ('fixed512_query1_copy_model_built_NOT_device_or_state_acceptance',
                        'actual_same_input_q1_fixed512_copy_model_full_packet_gate_passed'),
    'integer-tiles-q1': ('exact_query1_integer_tiles_model_built_NOT_device_or_state_acceptance',
                       'actual_same_input_q1_integer_tiles_model_full_packet_gate_passed'),
    'q1-runtime-combined': ('exact_query1_integer_tiles_and_fixed512_model_built_NOT_device_or_state_acceptance',
                           'actual_same_input_q1_combined_runtime_model_full_packet_gate_passed'),
    'empty-tensor-cse-q1': ('empty_tensor_CSE_query1_model_built_NOT_device_or_state_acceptance',
                          'actual_same_input_q1_empty_tensor_CSE_model_full_packet_gate_passed'),
    'gqa-value-view-q1': ('GQA_value_view_query1_model_built_NOT_device_or_state_acceptance',
                         'actual_same_input_q1_GQA_value_view_model_full_packet_gate_passed')}

KINDS['empty-tensor-cse-q2to32']=('empty_tensor_CSE_query2to32_model_built_NOT_device_or_state_acceptance','actual_same_input_dynamic_empty_tensor_CSE_model_full_packet_gate_passed')

KINDS.update({k:(c['model_status'],c['run_status']) for k,c in CONTRACTS.items()})
KINDS[INVARIANT_KIND]=(INVARIANT_MODEL_STATUS,INVARIANT_RUN_STATUS)


def require_loader_control(run,plan,command):
    flags=('--constant-load-split-proofs','--constant-loader-mode')
    present=[flag in command for flag in flags]
    if any(present):
        policy=plan.get('bounded_constant_loader',{})
        value=lambda flag:command[command.index(flag)+1]
        if (not all(present) or any(command.count(flag)!=1 or command.index(flag)+1==len(command) for flag in flags) or
                policy.get('kind')!=LOADER_KIND or value(flags[0])!=policy.get('proofs') or value(flags[1])!=MODE or
                [list(row) for row in verify_native_layout(run)]!=plan.get('expected_loader_layout_fingerprint')):
            raise ValueError('Both early packet timing arms require the same accepted constant payload layout')
    elif 'constant_loader_layout' in run or 'bounded_constant_loader' in plan:
        raise ValueError('Unrequested constant layout cannot silently change the packet timing control')

def revisit(path, plan_path, cold):
    plan = json.loads(plan_path.read_text()); run = json.loads(path.read_text())
    command = plan['command']; value = lambda flag: command[command.index(flag)+1]
    kind = value('--candidate-kind')
    if kind not in (KIND,FORWARD_KIND,PV_KIND,'empty-tensor-cse-q2to32',INVARIANT_KIND): raise ValueError('Distinct proved K schedule or scalar-source early gate only')
    pv=kind==PV_KIND
    scalar_forward=kind==FORWARD_KIND
    invariant=kind==INVARIANT_KIND
    dynamic_cse=kind in ('empty-tensor-cse-q2to32',INVARIANT_KIND)
    build_status, run_status = KINDS[kind]
    cse = kind in ('empty-tensor-cse-q1','empty-tensor-cse-q2to32',INVARIANT_KIND)
    shared = kind in CONTRACTS
    contract = CONTRACTS.get(kind)
    view = kind == 'gqa-value-view-q1' or shared
    compiled_q1 = cse or view
    require_loader_control(run,plan,command)
    files = {row['name']: row for row in plan['files']}
    for row in files.values():
        if sha(row['source']) != row['sha256']: raise ValueError('Original staged input changed')
    build_path = Path(files['result.json']['source']); build = json.loads(build_path.read_text())
    native_path = cold/'native-result.json'; native = json.loads(native_path.read_text())
    _, libraries = candidate_libraries(cold)
    forward = next(row for row in native['forwards'] if row['query'] == (8 if dynamic_cse else 1))
    baseline_sha=libraries[(2,32) if dynamic_cse else (1,1)]
    from profile_model_memory import checked
    sdk=json.loads(checked(build['compiler_receipt']).read_text())
    code=json.loads(checked(build['objects_receipt']).read_text())
    if dynamic_cse:
        binding=build['original_accepted_cold_packet']
        closure=json.loads(checked(build['accepted_q2_baseline393']).read_text())
        owner=json.loads(checked(build['code_process']).read_text())
        original=json.loads((Path(build['original_model_object']['path']).parent/'result.json').read_text())
        expected=dict(native_sha256=sha(native_path),audit_sha256=files['accepted-audit.json']['sha256'],
            input_sha256=forward['input_sha256'],output_sha256=forward['output_sha256'],DSO_sha256=baseline_sha,query=8)
        if (binding!=expected or closure['DSP_calls']!=393 or closure['model_inference_acceptance'] is not True
                or closure['candidate_libraries']['(2, 32)']!=baseline_sha
                or build['candidate_kind']!=kind or build['query1_model_unchanged'] is not True
                or owner['status']!='completed' or owner['returncode']!=0
                or owner['completed_unix']>=owner['authorized_deadline_unix']
                or any(Path('/proc',str(owner[k])).exists() for k in ('controller_pid','child_pid'))
                or sdk['status']!=(INVARIANT_SDK_STATUS if invariant else 'isolated_empty_tensor_CSE_SDK_built_NOT_model_or_DSP')
                or not all(sdk[k] is True for k in ('unmodified_SDK_replay_byte_identical','original_link_inputs_unchanged','original_SDK_unchanged'))
                or not invariant and sdk['accepted_sdk']['sha256']!=original['inputs']['compiler']['sha256']
                or code['inputs']['compiler']['sha256']!=sdk['candidate']['sha256']
                or code['status']!=original['status'] or code['machine_code_compilations']!=1 or code['DSP_calls']!=0
                or code['model_execution'] is not False or not same_linalg_input(code['inputs']['linalg'],original['inputs']['linalg'],relocated=invariant)
                or any(code[k]!=original[k] for k in ('overrides','option_reconciliation','expected_objects_result','triton_python'))
                or code['inputs']['options_result']!=original['inputs']['options_result']
                or len(code['objects'])!=4 or any(code['objects'][i]['sha256']!=original['objects'][i]['sha256'] for i in (1,2,3))
                or code['objects'][0]['sha256']==original['objects'][0]['sha256']):
            raise ValueError('Exact original dynamic graph, isolated CSE SDK, unchanged constants and normal owner required')
        argv=owner['command'];value_arg=lambda flag:argv[argv.index(flag)+1]
        if (value_arg('--compiler-sha256')!=sdk['candidate']['sha256']
                or value_arg('--linalg-sha256')!=code['inputs']['linalg']['sha256']
                or Path(value_arg('--output'))!=Path(build['objects_receipt']['path']).parent):
            raise ValueError('Actual dynamic CSE compilation command differs')
        for ref in (sdk['candidate'],sdk['accepted_sdk'],sdk['source'],code['objects'][0],code['compiler_diagnostics']):checked(ref)
        if invariant:
            full_audit=json.loads(checked(build['accepted_model_audit']).read_text())
            require_reference(closure,full_audit,build['accepted_q2_baseline393'])
            require_compiler(sdk,checked(code['compiler_diagnostics']).read_text())
            checked(sdk['dynamic_QK_invariant_source'])
            early=json.loads(checked(closure['early_model_revisit']).read_text())
            base_model=json.loads(checked(early['build']).read_text())
            base_sdk=json.loads(checked(base_model['compiler_receipt']).read_text())
            if (sdk['accepted_sdk']!=base_sdk['accepted_sdk'] or sdk['link_inputs']!=base_sdk['link_inputs']
                    or build.get('transformed_QK_regions')!=28 or build.get('vector_columns')!=128
                    or build.get('per_output_FMA_values_and_order_retained') is not True
                    or build.get('independent_column_vectorization') is not True):
                raise ValueError('Actual original SDK, guarded all28 scalar hoists and per-output FMA order required')
        else:
            checked(sdk['only_added_translation_unit'])
    else:
        binding=build['original_N32_cold_packet' if scalar_forward else 'original_R_cold_packet']
        if (binding != dict(native_sha256=sha(native_path), audit_sha256=files['accepted-audit.json']['sha256'],
                input_sha256=forward['input_sha256'], output_sha256=forward['output_sha256'], DSO_sha256=baseline_sha)
                or build['accepted_baseline393_calls' if scalar_forward else 'accepted_original_R393_calls'] != 393):
            raise ValueError('Exact current independently accepted R393 packet required')
        (require_PV_compiler if pv else validate_key_forward if scalar_forward else validate_key_tile)(sdk,checked(code['compiler_diagnostics']).read_text())
        if pv:
            require_PV_reference(json.loads(checked(build['accepted_baseline_closure']).read_text()),json.loads(checked(build['accepted_model_audit']).read_text()),build['accepted_baseline_closure'])
    if (run['status'] != run_status or run['DSP_calls'] != 8 or len(run['rows']) != 8
            or run['process_exit_code'] != 0 or run['rounds'] != 2 or run['candidate_kind'] != kind
            or value('--rounds') != '2' or (not compiled_q1 and value('--query') != '1')
            or run['receipt_sha256'] != sha(build_path) or value('--receipt-sha256') != sha(build_path)
            or run['source_sha256'] != files['probe_model_helper_effects_arm.py']['sha256']
            or run['accepted_run_sha256'] != sha(native_path) or value('--accepted-run-sha256') != sha(native_path)
            or run['accepted_audit_sha256'] != files['accepted-audit.json']['sha256']
            or build['status'] != build_status or build['actual_model_compilations'] != (1 if compiled_q1 else 0) or build['link_returncode'] != 0
            or build['query_bounds'] != ([2,32] if dynamic_cse else [1,1]) or build['input_bytes_are_maximum_capacity'] is not dynamic_cse
            or dynamic_cse and (run['query']!=8 or value('--query')!='8')
            or run['input_sha256'] != forward['input_sha256']
            or run['original_complete_output_sha256'] != forward['output_sha256']
            or run['baseline_library_sha256'] != baseline_sha
            or build['original_model_library']['sha256'] != baseline_sha
            or run['candidate_library_sha256'] != build['library']['sha256']
            or run['inline_CPU_observer'] is not False or run['diagnostic_instrumentation'] is not False
            or build['model_object_unchanged'] is not (False if compiled_q1 else True)
            or any(build[key] is not True for key in ('model_graph_unchanged',
                'original_caller_unchanged','original_math_header_unchanged','constant_objects_identical',
                'original_guarded_copy_runtime_unchanged','test_hooks_absent_from_production_runtime'))
            or build['FP_evaluation_order_changed'] is not False or build['power_votes_changed'] is not False):
        raise ValueError('Actual scoped q1 original ARM packet and production runtime gate required')
    extra=[build['fixed512_runtime']] if kind=='q1-runtime-combined' else []
    runtime=[] if compiled_q1 else [build['runtime']]
    if compiled_q1:
        runtime=[build['compiler_receipt'],build['objects_receipt']]
        change=(INVARIANT_CHANGE if invariant else contract['change'] if shared else 'proven identity V layout copy to tensor shape view after original CSE' if view
                else 'CSE after original empty-tensor elimination')
        if (build['SDK_unmodified_replay_byte_identical'] is not True
                or build['original_runtime_objects_unchanged'] is not True
                or build['only_compiler_change'] != change):
            raise ValueError('Exactly the isolated CSE compiler with all original runtimes required')
        raw=[json.loads(line) for line in (path.parent/'stream.stdout.txt').read_text().splitlines()]
        invokes=[event for event in raw if event.get('stage')=='invoke']
        if (raw != run['stream_events'] or not raw or raw[-1].get('stage')!='close'
                or invokes != [row['actual_invoke'] for row in run['rows']]
                or len([event for event in raw if event.get('stage')=='close']) != 2
                or any(event.get('rpc_result',0) or event.get('loader_result',0) for event in raw)):
            raise ValueError('Complete original raw runner invocations and normal close required')
    for row in build['original_link_inputs']+runtime+[build['library'],build['original_model_library']]+extra:
        if sha(row['path']) != row['sha256']: raise ValueError('Original model or linked runtime changed')
    for index, row in enumerate(run['rows']):
        if ((row['round'],row['index'],row['library']) != (index//4,index%4,('baseline','candidate','candidate','baseline')[index%4])
                or row['full_output_sha256'] != forward['output_sha256'] or row['all_logits_and_56_KV_words_bitexact'] is not True):
            raise ValueError('Original ABBA ordering or complete logits/56KV changed')
        event = row['actual_invoke']
        if (any(event[key] != forward['invoke'][key] for key in ('input_bytes','output_bytes',
                'AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'))
                or any(event[key] != 0 for key in ('rpc_result','DSP_result','output_result'))):
            raise ValueError('Complete original packet transport identity changed')
    measurements = {}
    for name, event_key in (('kernel_us','kernel_us'),('callback_seconds','AP_total_seconds'),('rpc_seconds','seconds')):
        old = statistics.mean((row['actual_invoke']['profile'][event_key] if event_key=='AP_total_seconds' else row['actual_invoke'][event_key]) for row in run['rows'] if row['library']=='baseline')
        new = statistics.mean((row['actual_invoke']['profile'][event_key] if event_key=='AP_total_seconds' else row['actual_invoke'][event_key]) for row in run['rows'] if row['library']=='candidate')
        measurements[name] = dict(baseline=old,candidate=new,ratio=new/old)
    result=dict(status=('actual_q1_original_ARM_complete_packet_unwrapped_GQA_value_view_ABBA8_revisited' if view else
                        'actual_q1_original_ARM_complete_packet_unwrapped_CSE_ABBA8_revisited' if cse else
                        'actual_q1_original_ARM_complete_packet_unwrapped_runtime_ABBA8_revisited'),
        reviewer_source=entry(__file__),candidate_kind=kind,run=dict(path=str(path),sha256=sha(path)),plan=dict(path=str(plan_path),sha256=sha(plan_path)),
        build=dict(path=str(build_path),sha256=sha(build_path)),accepted_cold=dict(path=str(cold/'acceptance.json'),sha256=sha(cold/'acceptance.json')),
        measurements=measurements,DSP_calls=8,positive_early_model_gain=all(measurements[key]['ratio']<.995 for key in ('kernel_us','callback_seconds')),
        model_inference_acceptance=False)
    if dynamic_cse:
        result['status']=('actual_original_dynamic_ARM_complete_packet_unwrapped_QK_invariant_vector_ABBA8_revisited' if invariant else 'actual_original_dynamic_ARM_complete_packet_unwrapped_CSE_ABBA8_revisited')
    if invariant or pv:
        complete=path.parent/'complete-candidate-output.bin'
        if (sha(complete)!=forward['output_sha256'] or complete.stat().st_size!=forward['invoke']['output_bytes']
                or run.get('preserved_complete_candidate_output',{}).get('sha256')!=sha(complete)):
            raise ValueError('Preserved full235MB candidate output must match the original independently accepted packet')
        result['complete_candidate_output']=entry(complete)
        result['accepted_model_audit']=build['accepted_model_audit']
    if shared:
        result['status']='actual_q1_original_ARM_complete_packet_unwrapped_'+kind+'_ABBA8_revisited'
        proof=build['primitive_revisit']
        if sha(proof['path'])!=proof['sha256']:raise ValueError('Actual standalone proof changed')
    if compiled_q1:result['full_original_logits_and_56_KV_byteexact']=True
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','plan','accepted-cold','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not a.output.is_absolute() or a.output.exists(): p.error('Fresh absolute q1 revisit required')
    result=revisit(a.run,a.plan,a.accepted_cold);write(a.output,result);print(result['status'])


if __name__=='__main__':main()
