"""Early same-input, complete-output ABBA gate for scoped helper effects.

This replays an independently accepted real q1 input. It measures unwrapped
kernel/callback cost, not startup, generation semantics or long-state closure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import traceback


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','receipt','accepted-run','accepted-audit','baseline-library','workdir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','receipt','accepted-run','accepted-audit'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--candidate-kind',choices=('silu-effects','guarded-copy','guarded-copy-dynamic','multi-integer','multi-silu','fixed512-copy-q1','integer-tiles-q1','q1-runtime-combined','empty-tensor-cse-q1','gqa-value-view-q1','gqa-value-forward-q1','gqa-key-compact-q1'),default='silu-effects')
    p.add_argument('--accepted-model-gate',type=Path)
    p.add_argument('--accepted-model-gate-sha256')
    p.add_argument('--query',type=int,default=1)
    a=p.parse_args()
    if (platform.machine()!='aarch64' or sys.byteorder!='little' or a.output.exists()
            or not a.output.is_absolute() or not 1<=a.rounds<=8):raise ValueError('Actual ARM and fresh bounded model replay required')
    for name in ('catalog','receipt','accepted_run','accepted_audit'):
        if sha(getattr(a,name))!=getattr(a,name+'_sha256'):raise ValueError('Pinned complete replay input changed: '+name)
    receipt=json.loads(a.receipt.read_text());run=json.loads(a.accepted_run.read_text())
    audit=json.loads(a.accepted_audit.read_text());catalog=json.loads(a.catalog.read_text())
    dynamic=a.candidate_kind in ('guarded-copy-dynamic','multi-integer','multi-silu')
    shared=a.candidate_kind in ('gqa-value-forward-q1','gqa-key-compact-q1')
    contract=None
    if shared:
        from gqa_shared_copy_contract import CONTRACTS
        contract=CONTRACTS[a.candidate_kind]
    view=a.candidate_kind=='gqa-value-view-q1' or shared
    compiled_q1=view or a.candidate_kind=='empty-tensor-cse-q1'
    if (dynamic and (not 2<=a.query<=32 or receipt.get('query_bounds')!=[2,32]
                     or not receipt.get('input_bytes_are_maximum_capacity'))) or (not dynamic and a.query!=1):
        raise ValueError('Explicit query must match the original compiled graph bounds')
    if a.candidate_kind=='silu-effects':
        scoped=(receipt['status']=='scoped_query1_SiLU_effect_model_built_NOT_device_or_state_acceptance'
            and receipt['original_inspection_identical_except_helper_declaration']
            and receipt['helper_effect_contract']['actual_fmaf_entire_code_bytes']=='82c100efc03f2050')
    elif a.candidate_kind=='multi-integer':
        scoped=(receipt['status']=='exact_query2_to_32_integer_model_built_NOT_device_or_state_acceptance'
            and receipt['all_surviving_original_operations_properties_and_SSA_preserved']
            and receipt['original_caller_unchanged'] and receipt['original_integer_contractions_replaced']==197
            and receipt['test_hooks_absent_from_production_runtime'])
    elif a.candidate_kind=='multi-silu':
        scoped=(receipt['status']=='scoped_query2_to_32_SiLU_effect_model_built_NOT_device_or_state_acceptance'
            and receipt['original_inspection_identical_except_helper_declaration']
            and receipt['all_original_operations_properties_and_SSA_preserved']
            and receipt['original_integer_runtime_unchanged'] and receipt['original_integer_contractions_retained']==197
            and receipt['helper_effect_contract']['actual_fmaf_entire_code_bytes']=='82c100efc03f2050'
            and receipt['helper_effect_contract']['contract']=='original_Qwen_query2_to32_silu_readnone_nounwind_willreturn_v1'
            and receipt['test_hooks_absent_from_production_runtime'])
    elif compiled_q1:
        expected_status=(contract['model_status'] if shared else 'GQA_value_view_query1_model_built_NOT_device_or_state_acceptance' if view
                         else 'empty_tensor_CSE_query1_model_built_NOT_device_or_state_acceptance')
        change=(contract['change'] if shared else 'proven identity V layout copy to tensor shape view after original CSE' if view
                else 'CSE after original empty-tensor elimination')
        scoped=(receipt['status']==expected_status
            and receipt.get('query_bounds')==[1,1] and receipt['input_bytes_are_maximum_capacity'] is False
            and receipt['model_graph_unchanged'] and receipt['model_object_unchanged'] is False
            and receipt['original_caller_unchanged'] and receipt['original_runtime_objects_unchanged']
            and receipt['SDK_unmodified_replay_byte_identical']
            and receipt['only_compiler_change']==change
            and receipt['actual_model_compilations']==1 and receipt['test_hooks_absent_from_production_runtime'])
        if view:scoped=scoped and receipt['eliminated_identity_value_copies']==28
    elif a.candidate_kind in ('fixed512-copy-q1','integer-tiles-q1','q1-runtime-combined'):
        status={'fixed512-copy-q1':'fixed512_query1_copy_model_built_NOT_device_or_state_acceptance',
                'integer-tiles-q1':'exact_query1_integer_tiles_model_built_NOT_device_or_state_acceptance',
                'q1-runtime-combined':'exact_query1_integer_tiles_and_fixed512_model_built_NOT_device_or_state_acceptance'}[a.candidate_kind]
        scoped=(receipt['status']==status
            and receipt.get('query_bounds')==[1,1] and receipt['input_bytes_are_maximum_capacity'] is False
            and receipt['model_graph_unchanged'] and receipt['model_object_unchanged']
            and receipt['original_caller_unchanged'] and receipt['original_guarded_copy_runtime_unchanged']
            and receipt['actual_model_compilations']==0 and receipt['test_hooks_absent_from_production_runtime'])
    else:
        status=('guarded_query2_to_32_byte_copy_model_built_NOT_device_or_state_acceptance' if dynamic
                else 'guarded_query1_byte_copy_model_built_NOT_device_or_state_acceptance')
        scoped=(receipt['status']==status
            and receipt['model_graph_unchanged'] and receipt['model_object_unchanged']
            and receipt['original_caller_unchanged'] and receipt['test_hooks_absent_from_production_runtime'])
    if (not scoped or receipt['FP_evaluation_order_changed'] or receipt['power_votes_changed']
            or not receipt['original_math_header_unchanged'] or not receipt['constant_objects_identical']
            or audit['status']!='framework_native_numerical_acceptance_passed'
            or not audit['numerical_acceptance'] or audit['run_sha256']!=a.accepted_run_sha256
            or run['process_exit_code'] or run['CPU_model_fallback']):raise ValueError('Pinned candidate and independently accepted original q1 required')
    forward=next(row for row in run['forwards'] if row['query']==a.query)
    baseline_sha=forward['DSO_sha256']
    if a.candidate_kind in ('fixed512-copy-q1','integer-tiles-q1','q1-runtime-combined','empty-tensor-cse-q1') and receipt['original_model_library']['sha256']!=baseline_sha:
        raise ValueError('Query1 runtime relink must preserve the independently accepted model object')
    if a.candidate_kind=='multi-silu' or view:
        if (not a.accepted_model_gate or not a.accepted_model_gate_sha256
                or sha(a.accepted_model_gate)!=a.accepted_model_gate_sha256
                or receipt['baseline_model_gate']['sha256']!=a.accepted_model_gate_sha256):
            raise ValueError('Pinned actual original-ARM-packet integer baseline model gate required')
        from probe_model_memory_arm import accepted_baseline_sha
        baseline_sha=accepted_baseline_sha(json.loads(a.accepted_model_gate.read_text()),a.query,forward,
            a.accepted_run_sha256,a.accepted_audit_sha256,allow_query1_cse=view and not shared,allow_query1_gqa_value_view=shared)
        if receipt['original_model_library']['sha256']!=baseline_sha:
            raise ValueError('SiLU candidate base is not the same accepted integer model')
    elif a.accepted_model_gate or a.accepted_model_gate_sha256:
        raise ValueError('Explicit integer baseline gate only belongs to the multi-SiLU experiment')
    packet=a.accepted_run.parent/f"forward-{forward['step']:02d}-q{a.query}/input.bin"
    if sha(packet)!=forward['input_sha256'] or sha(a.baseline_library)!=baseline_sha:
        raise ValueError('Accepted complete input or original DSO changed')
    library=a.receipt.parent/'candidate.so'
    if sha(library)!=receipt['library']['sha256']:raise ValueError('Candidate DSO changed')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage
    from hexagon_live_compile_mailbox import atomic_json
    active=subprocess.run(['pgrep','-x','buffer_stream'],capture_output=True,timeout=10)
    if active.returncode not in (0,1) or active.stdout.strip():raise ValueError('DSP owner still active')
    stage(catalog,a.workdir,include_code=False)
    for row in receipt['constants']:
        if sha(a.workdir/Path(row['path']).name)!=row['sha256']:raise ValueError('Frozen model constant library changed')
    a.output.mkdir(parents=True)
    staged={}
    for name,path in [('baseline',a.baseline_library),('candidate',library)]:
        target=a.workdir/(sha(path)+'.so');shutil.copy2(path,target)
        if sha(target)!=sha(path):raise ValueError('Actual model DSO staging changed')
        staged[name]=target
    payload=packet.read_bytes();output_bytes=receipt['output_bytes']
    capacity_ok=len(payload)<=receipt['input_bytes'] if dynamic else len(payload)==receipt['input_bytes']
    if (not capacity_ok or len(payload)!=forward['invoke']['input_bytes']
            or output_bytes!=forward['invoke']['output_bytes']
            or payload[:16]!=b'HXDY\1\0\0\0'+struct.pack('<q',a.query)):
        raise ValueError('Frozen complete q1 packet required')
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)
        +';'+str(a.workdir/'rfsa')+';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),str(staged['baseline']),str(len(payload)),str(output_bytes)]
    record=dict(status='actual_same_input_q1_SiLU_effect_model_gate_running',source_sha256=sha(__file__),
        candidate_kind=a.candidate_kind,
        receipt_sha256=a.receipt_sha256,catalog_sha256=a.catalog_sha256,accepted_run_sha256=a.accepted_run_sha256,
        accepted_audit_sha256=a.accepted_audit_sha256,input_sha256=forward['input_sha256'],
        original_complete_output_sha256=forward['output_sha256'],baseline_library_sha256=baseline_sha,
        candidate_library_sha256=receipt['library']['sha256'],inline_CPU_observer=False,diagnostic_instrumentation=False,
        power_votes_changed=False,rounds=a.rounds,rows=[],model_inference_acceptance=False,
        scope='early_raw_complete_q1_packet_ABBA_NOT_native_generation_or_original_state_acceptance')
    if dynamic:
        record.update(query=a.query,input_bytes=len(payload),output_bytes=output_bytes,
            scope='early_raw_complete_original_dynamic_packet_ABBA_NOT_native_generation_or_state_acceptance')
    if compiled_q1:
        record['status']=('actual_same_input_q1_GQA_value_view_model_gate_running' if view
                          else 'actual_same_input_q1_empty_tensor_CSE_model_gate_running')
        if shared: record['status']='running_'+a.candidate_kind+'_complete_packet_gate'
        record['query']=1
        record['DSP_calls']=0
    if a.candidate_kind=='multi-silu':
        record['accepted_baseline_model_gate']=dict(path=str(a.accepted_model_gate),sha256=a.accepted_model_gate_sha256)
    def save():atomic_json(a.output/'result.json',record)
    save();previous=Path.cwd()
    try:
        os.chdir(a.workdir)
        with HexagonSharedPacketStream(argv,env,a.output,input_bytes=len(payload),output_bytes=output_bytes,
                initial_library=str(staged['baseline']),timeout=120) as stream:
            record['ready']=stream.start();save()
            for round_ in range(a.rounds):
                for index,name in enumerate(('baseline','candidate','candidate','baseline')):
                    record['pending']=dict(round=round_,index=index,library=name);save()
                    stream.select_library(str(staged[name]));output,event=stream.invoke_packet(payload,output_bytes)
                    if compiled_q1:
                        record['DSP_calls']+=1
                        record['last_completed_invoke']=event
                        save()
                    digest=hashlib.sha256(output).hexdigest()
                    if compiled_q1:
                        expected_fnv=forward['invoke']['AP_input_fnv1a32']
                        if any(event.get(key)!=expected_fnv for key in
                               ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')):
                            raise ValueError('Original complete packet three-FNV contract changed')
                    if digest!=forward['output_sha256']:
                        (a.output/'failed-output.bin').write_bytes(output);raise ValueError('Complete logits/56KV output differs from independent accepted ARM packet')
                    del output
                    record['rows'].append(dict(round=round_,index=index,library=name,full_output_sha256=digest,
                        all_logits_and_56_KV_words_bitexact=True,actual_invoke=event));save()
        if stream.process_exit_code:raise ValueError('DSP owner exit failed')
        record.pop('pending',None)
        record.update(status='actual_same_input_q1_SiLU_effect_model_full_packet_gate_passed',
            DSP_calls=len(record['rows']),process_exit_code=stream.process_exit_code,stream_events=stream.events)
        if a.candidate_kind=='guarded-copy':
            record['status']='actual_same_input_q1_guarded_copy_model_full_packet_gate_passed'
        elif a.candidate_kind=='fixed512-copy-q1':
            record['status']='actual_same_input_q1_fixed512_copy_model_full_packet_gate_passed'
        elif a.candidate_kind=='integer-tiles-q1':
            record['status']='actual_same_input_q1_integer_tiles_model_full_packet_gate_passed'
        elif a.candidate_kind=='q1-runtime-combined':
            record['status']='actual_same_input_q1_combined_runtime_model_full_packet_gate_passed'
        elif a.candidate_kind=='empty-tensor-cse-q1':
            record['status']='actual_same_input_q1_empty_tensor_CSE_model_full_packet_gate_passed'
        elif view:
            record['status']=contract['run_status'] if shared else 'actual_same_input_q1_GQA_value_view_model_full_packet_gate_passed'
        elif a.candidate_kind=='multi-integer':
            record['status']='actual_same_input_dynamic_integer_model_full_packet_gate_passed'
        elif a.candidate_kind=='multi-silu':
            record['status']='actual_same_input_dynamic_SiLU_effect_model_full_packet_gate_passed'
        elif dynamic:
            record['status']='actual_same_input_dynamic_guarded_copy_model_full_packet_gate_passed'
        save();os.chdir(previous);shutil.rmtree(a.workdir)
    except BaseException:record.update(status='failed',error=traceback.format_exc());save();raise
    finally:os.chdir(previous)
    print(record['status'])


if __name__=='__main__':main()
