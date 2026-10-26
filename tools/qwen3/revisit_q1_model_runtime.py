"""Bind a completed q1 runtime ABBA8 to its original complete ARM packet."""
import argparse
import json
from pathlib import Path
import statistics

from build import sha, write
from run_integer_regressions import candidate_libraries
from gqa_shared_copy_contract import CONTRACTS

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

KINDS.update({k:(c['model_status'],c['run_status']) for k,c in CONTRACTS.items()})

def revisit(path, plan_path, cold):
    plan = json.loads(plan_path.read_text()); run = json.loads(path.read_text())
    command = plan['command']; value = lambda flag: command[command.index(flag)+1]
    kind = value('--candidate-kind'); build_status, run_status = KINDS[kind]
    cse = kind == 'empty-tensor-cse-q1'
    shared = kind in CONTRACTS
    contract = CONTRACTS.get(kind)
    view = kind == 'gqa-value-view-q1' or shared
    compiled_q1 = cse or view
    files = {row['name']: row for row in plan['files']}
    for row in files.values():
        if sha(row['source']) != row['sha256']: raise ValueError('Original staged input changed')
    build_path = Path(files['result.json']['source']); build = json.loads(build_path.read_text())
    native_path = cold/'native-result.json'; native = json.loads(native_path.read_text())
    _, libraries = candidate_libraries(cold)
    forward = next(row for row in native['forwards'] if row['query'] == 1)
    baseline_sha=libraries[(1,1)]
    if view:
        gate=build['baseline_model_gate']
        if sha(gate['path'])!=gate['sha256']:
            raise ValueError('Actual original CSE baseline model gate changed')
        from probe_model_memory_arm import accepted_baseline_sha
        baseline_sha=accepted_baseline_sha(json.loads(Path(gate['path']).read_text()),1,forward,
            sha(native_path),files['accepted-audit.json']['sha256'],allow_query1_cse=not shared,allow_query1_gqa_value_view=shared)
    if (run['status'] != run_status or run['DSP_calls'] != 8 or len(run['rows']) != 8
            or run['process_exit_code'] != 0 or run['rounds'] != 2 or run['candidate_kind'] != kind
            or value('--rounds') != '2' or (not compiled_q1 and value('--query') != '1')
            or run['receipt_sha256'] != sha(build_path) or value('--receipt-sha256') != sha(build_path)
            or run['source_sha256'] != files['probe_model_helper_effects_arm.py']['sha256']
            or run['accepted_run_sha256'] != sha(native_path) or value('--accepted-run-sha256') != sha(native_path)
            or run['accepted_audit_sha256'] != files['accepted-audit.json']['sha256']
            or build['status'] != build_status or build['actual_model_compilations'] != (1 if compiled_q1 else 0) or build['link_returncode'] != 0
            or build['query_bounds'] != [1,1] or build['input_bytes_are_maximum_capacity'] is not False
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
        change=(contract['change'] if shared else 'proven identity V layout copy to tensor shape view after original CSE' if view
                else 'CSE after original empty-tensor elimination')
        if (build['SDK_unmodified_replay_byte_identical'] is not True
                or build['original_runtime_objects_unchanged'] is not True
                or build['only_compiler_change'] != change):
            raise ValueError('Exactly the isolated CSE compiler with all original runtimes required')
        raw=[json.loads(line) for line in (path.parent/'stream.stdout.txt').read_text().splitlines()]
        invokes=[event for event in raw if event.get('stage')=='invoke']
        if (invokes != [row['actual_invoke'] for row in run['rows']]
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
    for name, event_key in (('kernel_us','kernel_us'),('callback_seconds','seconds')):
        old = statistics.mean(row['actual_invoke'][event_key] for row in run['rows'] if row['library']=='baseline')
        new = statistics.mean(row['actual_invoke'][event_key] for row in run['rows'] if row['library']=='candidate')
        measurements[name] = dict(baseline=old,candidate=new,ratio=new/old)
    result=dict(status=('actual_q1_original_ARM_complete_packet_unwrapped_GQA_value_view_ABBA8_revisited' if view else
                        'actual_q1_original_ARM_complete_packet_unwrapped_CSE_ABBA8_revisited' if cse else
                        'actual_q1_original_ARM_complete_packet_unwrapped_runtime_ABBA8_revisited'),
        candidate_kind=kind,run=dict(path=str(path),sha256=sha(path)),plan=dict(path=str(plan_path),sha256=sha(plan_path)),
        build=dict(path=str(build_path),sha256=sha(build_path)),accepted_cold=dict(path=str(cold/'acceptance.json'),sha256=sha(cold/'acceptance.json')),
        measurements=measurements,DSP_calls=8,positive_early_model_gain=all(row['ratio']<.995 for row in measurements.values()),
        model_inference_acceptance=False)
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
