"""Compare the complete CSE model packet, then read diagnostic region timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import sys
import traceback

from probe_model_memory_arm import sha, accepted_baseline_sha


def parse_profile(data, region_count):
    if not 1 <= region_count <= 4096 or len(data) != 128+4096*32 or data[:8] != b'QWLOOP1\0':
        raise ValueError('Invalid bounded loop profile packet')
    version, slots, failures, invocations, row_bytes = struct.unpack_from('<IIQQI',data,8)
    if (version,slots,row_bytes)!=(1,4096,32) or failures or not invocations or any(data[36:128]):
        raise ValueError('Incomplete loop timers or unsupported response schema')
    rows=[]
    for index in range(4096):
        calls, us, cycles, reserved = struct.unpack_from('<QQQQ',data,128+index*32)
        if reserved or (index < region_count and (calls != 1 or not cycles)):
            raise ValueError('Every top-level region must execute exactly once with valid timing')
        if index >= region_count and any((calls,us,cycles,reserved)):
            raise ValueError('Unexpected region outside the actual compiler descriptors')
        if index < region_count:
            rows.append(dict(id=index,calls=calls,microseconds=us,cycles=cycles))
    return dict(invocations=invocations, failures=failures, rows=rows,
                scope='diagnostic_top_level_regions_NOT_production_throughput')


def dynamic_accepted_baseline(gate, audit, gate_sha, audit_sha, receipt, forward):
    if (gate.get('status') != 'query2to32_empty_tensor_CSE_native_timing_and_original_states_accepted'
            or gate.get('candidate_kind') != 'empty-tensor-cse-q2to32'
            or gate.get('model_inference_acceptance') is not True
            or any(type(gate.get(k)) is not int or gate[k] != n for k, n in
                (('DSP_calls', 393), ('candidate_DSP_calls', 361), ('paired_baseline_DSP_calls', 32)))
            or audit.get('status') != 'query2to32_empty_tensor_CSE_native393_independently_revisited'
            or audit.get('closure') != receipt['accepted_closure']
            or receipt['accepted_closure']['sha256'] != gate_sha
            or receipt['accepted_audit']['sha256'] != audit_sha
            or audit.get('DSP_calls') != 393 or audit.get('new_DSP_calls') != 0
            or audit.get('model_inference_acceptance') is not True
            or not 2 <= forward['query'] <= 32
            or receipt.get('query_bounds') != [2, 32]
            or receipt.get('input_bytes_are_maximum_capacity') is not True
            or gate['candidate_libraries']['(2, 32)'] != forward['DSO_sha256']
            or receipt['original_model_library']['sha256'] != forward['DSO_sha256']):
        raise ValueError('Original dynamic diagnostic requires its own accepted CSE393 and independent audit')
    return forward['DSO_sha256']


def current_pv_dynamic_baseline(gate, audit, gate_sha, audit_sha, receipt, forward):
    from ordered_pv_release_acceptance import require_pv393
    require_pv393(gate, audit)
    if (receipt.get('current_PV393_typed_numerical_admission') is not True
            or any(audit['closure'].get(key) != receipt['accepted_closure'].get(key)
                for key in ('path','sha256'))
            or receipt['accepted_closure']['sha256'] != gate_sha
            or receipt['accepted_audit']['sha256'] != audit_sha
            or not 2 <= forward['query'] <= 32 or receipt.get('query_bounds') != [2,32]
            or receipt.get('input_bytes_are_maximum_capacity') is not True
            or gate['candidate_libraries']['(2, 32)'] != forward['DSO_sha256']
            or receipt['original_model_library']['sha256'] != forward['DSO_sha256']):
        raise ValueError('Current PV393 numerical proof and exact original dynamic packet required')
    return forward['DSO_sha256']


def stage_constants(stage, catalog, workdir, constants, split_proofs=None):
    layout = stage(catalog, workdir, include_code=False,
        **({'constant_load_split_proofs': split_proofs} if split_proofs else {})) or []
    if split_proofs and (len(layout) != 3
            or {row['original_sha256'] for row in layout} != {row['sha256'] for row in constants}
            or any(row['max_load_bytes'] != 256*1024**2 for row in layout)):
        raise ValueError('All three original payloads and bounded LOAD proofs required')
    for row in constants:
        staged_sha = sha(workdir/Path(row['path']).name)
        if layout:
            if not any(proof['original_sha256'] == row['sha256']
                    and proof['candidate_sha256'] == staged_sha for proof in layout):
                raise ValueError('Staged constant does not match its verified payload proof')
        elif staged_sha != row['sha256']:
            raise ValueError('Original constant library changed')
    return layout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','receipt','accepted-run','accepted-audit','baseline-library',
                 'accepted-model-gate','workdir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','receipt','accepted-run','accepted-audit','accepted-model-gate'):
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--dynamic-query', action='store_true')
    p.add_argument('--query', type=int, default=1)
    p.add_argument('--accepted-model-audit', type=Path)
    p.add_argument('--accepted-model-audit-sha256')
    p.add_argument('--constant-load-split-proofs', type=Path)
    p.add_argument('--constant-loader-mode', choices=('verified-payload-preserving',))
    p.add_argument('--current-pv-model',action='store_true')
    a=p.parse_args()
    if a.current_pv_model and not a.dynamic_query:
        p.error('Current PV loop diagnostic requires the original dynamic query mode')
    if (platform.machine()!='aarch64' or sys.byteorder!='little' or a.workdir.exists() or a.output.exists()
            or a.workdir.parent != Path('/dev/shm') or not a.workdir.name.startswith('qwen3-arm-dynamic-')
            or not a.output.is_absolute() or '..' in a.output.parts):
        raise ValueError('ARM board, fresh bounded workdir and output required')
    for name in ('catalog','receipt','accepted_run','accepted_audit','accepted_model_gate'):
        if sha(getattr(a,name))!=getattr(a,name+'_sha256'):
            raise ValueError('Pinned packet, model gate or deployment changed')
    receipt=json.loads(a.receipt.read_text()); native=json.loads(a.accepted_run.read_text())
    catalog=json.loads(a.catalog.read_text()); audit=json.loads(a.accepted_audit.read_text())
    if (receipt['status']!='diagnostic_inline_loop_model_linked_NOT_production_or_DSP'
            or not receipt['diagnostic_instrumentation'] or receipt['model_inference_acceptance']
            or not receipt['all_timer_state_and_calls_library_local']
            or audit['status']!='framework_native_numerical_acceptance_passed'
            or not audit['numerical_acceptance'] or audit['run_sha256']!=a.accepted_run_sha256
            or native['process_exit_code'] or native['CPU_model_fallback']
            or receipt['input_bytes']!=(235012880 if a.dynamic_query else 234885520)
            or receipt['output_bytes']!=235488768
            or bool(a.constant_load_split_proofs)!=bool(a.constant_loader_mode)
            or (a.dynamic_query and (not 2<=a.query<=32 or not a.accepted_model_audit
                or sha(a.accepted_model_audit)!=a.accepted_model_audit_sha256))
            or (not a.dynamic_query and (a.query!=1 or 'query_bounds' in receipt
                or a.accepted_model_audit or a.accepted_model_audit_sha256))):
        raise ValueError('Explicit separately linked diagnostic and complete accepted ARM reference required')
    forward=next(r for r in native['forwards'] if r['query']==a.query)
    gate=json.loads(a.accepted_model_gate.read_text())
    dynamic_gate=current_pv_dynamic_baseline if a.current_pv_model else dynamic_accepted_baseline
    baseline_sha=(dynamic_gate(gate,json.loads(a.accepted_model_audit.read_text()),
        a.accepted_model_gate_sha256,a.accepted_model_audit_sha256,receipt,forward) if a.dynamic_query
        else accepted_baseline_sha(gate,1,forward,a.accepted_run_sha256,a.accepted_audit_sha256,
                                  allow_query1_cse=True))
    packet=a.accepted_run.parent/f"forward-{forward['step']:02d}-q{a.query}/input.bin"
    if sha(packet)!=forward['input_sha256'] or sha(a.baseline_library)!=baseline_sha:
        raise ValueError('Exact original ARM packet or accepted CSE model changed')
    if receipt['original_model_library']['sha256']!=baseline_sha:
        raise ValueError('Diagnostic was not linked from the accepted CSE393 model')
    library=a.receipt.parent/'loop-profile.so'
    if sha(library)!=receipt['library']['sha256']:
        raise ValueError('Diagnostic model library changed')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage
    from hexagon_live_compile_mailbox import atomic_json
    constant_layout=stage_constants(stage,catalog,a.workdir,receipt['constants'],
        a.constant_load_split_proofs)
    libraries={}
    for role,path in (('baseline',a.baseline_library),('profile',library)):
        target=a.workdir/(sha(path)+'.so');shutil.copy2(path,target)
        if sha(target)!=sha(path):raise ValueError('Model staging changed')
        libraries[role]=target
    payload=packet.read_bytes();out_bytes=receipt['output_bytes']
    if ((not 16<len(payload)<=235012880 if a.dynamic_query else len(payload)!=234885520)
            or len(payload)!=forward['invoke']['input_bytes']
            or out_bytes!=forward['invoke']['output_bytes']
            or payload[:16]!=b'HXDY\1\0\0\0'+struct.pack('<q',a.query)):
        raise ValueError('Original complete selected-query input required')
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)
        +';'+str(a.workdir/'rfsa')+';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),
          str(libraries['baseline']),str(len(payload)),str(out_bytes)]
    a.output.mkdir(parents=True)
    record=dict(status='diagnostic_loop_model_ABBA_running',source_sha256=sha(__file__),
        receipt_sha256=a.receipt_sha256,source_dependencies=dict(probe_model_memory_arm=sha(Path(__file__).with_name('probe_model_memory_arm.py'))),
        catalog_sha256=a.catalog_sha256,accepted_run_sha256=a.accepted_run_sha256,
        accepted_audit_sha256=a.accepted_audit_sha256,accepted_model_gate_sha256=a.accepted_model_gate_sha256,
        input_sha256=forward['input_sha256'],expected_output_sha256=forward['output_sha256'],
        baseline_library_sha256=baseline_sha,profile_library_sha256=sha(library),
        rows=[],profiles=[],DSP_calls=0,model_calls=0,diagnostic_query_calls=0,
        diagnostic_instrumentation=True,model_inference_acceptance=False,power_votes_changed=False,
        query=a.query,constant_layout=constant_layout)
    if a.dynamic_query:
        record['accepted_model_audit_sha256']=a.accepted_model_audit_sha256
    def save():atomic_json(a.output/'result.json',record)
    save();previous=Path.cwd()
    try:
        os.chdir(a.workdir)
        with HexagonSharedPacketStream(argv,env,a.output,input_bytes=len(payload),output_bytes=out_bytes,
                initial_library=str(libraries['baseline']),timeout=120) as stream:
            record['ready']=stream.start();save()
            for index,role in enumerate(('baseline','profile','profile','baseline')):
                record['pending']=dict(index=index,role=role);save()
                stream.select_library(str(libraries[role]))
                output,event=stream.invoke_packet(payload,out_bytes)
                record['DSP_calls']+=1;record['model_calls']+=1
                record['pending']['actual_invoke']=event;save()
                digest=hashlib.sha256(output).hexdigest()
                if digest!=forward['output_sha256']:
                    (a.output/'failed-output.bin').write_bytes(output)
                    raise ValueError('Complete same-input logits/56KV differ from ARM W8A8')
                if any(event[k]!=forward['invoke']['AP_input_fnv1a32']
                       for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')):
                    raise ValueError('Original model packet three-FNV contract changed')
                if role=='profile' and index==1:
                    preserved=a.output/'profile-model-output-1.bin';preserved.write_bytes(output)
                    record['preserved_complete_profile_output']=dict(path=str(preserved),sha256=sha(preserved))
                del output
                record['rows'].append(dict(index=index,role=role,output_sha256=digest,
                    all_logits_and_56_KV_byteexact=True,actual_invoke=event));record.pop('pending');save()
                if role=='profile':
                    stats,event=stream.invoke_packet(b'QWLOOP1\0'+bytes(8),128+4096*32)
                    record['DSP_calls']+=1;record['diagnostic_query_calls']+=1
                    file=a.output/f'profile-{index}.bin';file.write_bytes(stats)
                    record['pending_query']=dict(actual_invoke=event,profile=dict(path=str(file),sha256=sha(file)));save()
                    profile=parse_profile(stats,receipt['region_count'])
                    if profile['invocations']!=len(record['profiles'])+1:
                        raise ValueError('Profile counters do not bind actual model calls')
                    record['profiles'].append(dict(index=index,profile=profile,actual_invoke=event,
                                                   raw=dict(path=str(file),sha256=sha(file))))
                    record.pop('pending_query');save()
        record.update(process_exit_code=stream.process_exit_code,stream_events=stream.events)
        if stream.process_exit_code!=0:raise ValueError('Actual board owner did not exit normally')
        record['status']='complete_ARM_bitexact_loop_region_ABBA4_and_queries2_DIAGNOSTIC_NOT_production_gain'
        save();os.chdir(previous);shutil.rmtree(a.workdir)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc())
        if 'stream' in locals():
            record.update(stream_events=stream.events,process_exit_code=stream.process_exit_code)
        save();raise
    finally:os.chdir(previous)
    print(record['status'])


if __name__=='__main__':main()
