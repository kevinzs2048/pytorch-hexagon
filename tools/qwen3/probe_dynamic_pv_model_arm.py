"""Replay the real cold packet and both ABI bounds with full model outputs.

q8 is the original ARM-W8A8 accepted packet. q2/q32 are derived cold inputs
compared to the unchanged accepted DSP model; no new CPU oracle is claimed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import traceback


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def resize_cold_packet(contract,payload,q):
    import numpy as np
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for,pack_request
    if payload[:16]!=b'HXDY\1\0\0\0'+struct.pack('<q',8) or contract['variables']!={'query':[2,32]}:
        raise ValueError('Original complete q8 dynamic packet and range2..32 required')
    layout=layout_for(contract,{'query':8})
    if len(payload)!=16+layout['input_bytes'] or len(layout['inputs'])!=60:
        raise ValueError('Original complete 60 tensor payloads required')
    pieces=[payload[16+r['offset']:16+r['offset']+r['bytes']] for r in layout['inputs']]
    tokens=np.frombuffer(pieces[0],dtype='<i8')
    if tokens.shape!=(8,) or any(np.frombuffer(pieces[i],dtype='<i8').tolist()!=list(range(8)) for i in (1,2)):
        raise ValueError('Original cold tokens and positions required')
    mask=np.frombuffer(pieces[5],dtype='<f4').reshape(8,1024);blocked=mask[0,1]
    if not np.isfinite(blocked) or blocked>=0 or not np.array_equal(mask,np.where(
            np.arange(1024)[None,:]<=np.arange(8)[:,None],np.float32(0),blocked)):
        raise ValueError('Original full causal mask must be proved before deriving new rows')
    for i in [3,4,*range(6,60)]:
        if np.frombuffer(pieces[i],dtype='<u4').any():raise ValueError('Only original empty cold KV may be resized')
    pieces[0]=np.resize(tokens,q).astype('<i8').tobytes()
    pieces[1]=pieces[2]=np.arange(q,dtype='<i8').tobytes()
    pieces[5]=np.where(np.arange(1024)[None,:]<=np.arange(q)[:,None],np.float32(0),blocked).astype('<f4').tobytes()
    return pack_request(contract,{'query':q},pieces)[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','receipt','accepted-run','accepted-audit','accepted-model-gate','accepted-model-audit',
                 'caller','baseline-library','constant-load-split-proofs','workdir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','receipt','accepted-run','accepted-audit','accepted-model-gate','accepted-model-audit','caller'):
        p.add_argument('--'+name+'-sha256',required=True)
    a=p.parse_args()
    if platform.machine()!='aarch64' or sys.byteorder!='little' or a.output.exists() or not a.output.is_absolute():
        raise ValueError('Actual ARM and fresh complete model comparison required')
    for name in ('catalog','receipt','accepted_run','accepted_audit','accepted_model_gate','accepted_model_audit','caller'):
        if sha(getattr(a,name))!=getattr(a,name+'_sha256'):raise ValueError('Pinned original model comparison inputs changed')
    read=lambda path:json.loads(Path(path).read_text());receipt=read(a.receipt);native=read(a.accepted_run)
    audit=read(a.accepted_audit);gate=read(a.accepted_model_gate);model_audit=read(a.accepted_model_audit)
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from ordered_pv_release_acceptance import require_pv393
    require_pv393(gate,model_audit)
    forward=next(row for row in native['forwards'] if row['query']==8)
    baseline_sha=gate['candidate_libraries']['(2, 32)']
    if (receipt['status']!='actual_dynamic_PV_register_model_linked_NOT_device_or_model_acceptance'
            or receipt['model_inference_acceptance'] or receipt['performance_acceptance'] or receipt['DSP_calls']
            or receipt['query_bounds']!=[2,32] or not receipt['input_bytes_are_maximum_capacity']
            or (receipt['input_arguments'],receipt['output_count'])!=(65,57)
            or receipt['original_model']['acceptance']['sha256']!=a.accepted_model_gate_sha256
            or receipt['original_model']['numerical_audit']['sha256']!=a.accepted_model_audit_sha256
            or receipt['original_model_library']['sha256']!=baseline_sha or forward['DSO_sha256']!=baseline_sha
            or audit['status']!='framework_native_numerical_acceptance_passed' or not audit['numerical_acceptance']
            or audit['run_sha256']!=a.accepted_run_sha256 or native['process_exit_code'] or native['CPU_model_fallback']
            or forward['cache_positions']!=list(range(8)) or sha(a.baseline_library)!=baseline_sha):
        raise ValueError('Original full PV393, native ARM W8A8 cold8 and unaccepted candidate required')
    caller=read(a.caller)
    if caller['wrapper']!=receipt['original_caller'] or caller['contract']['variables']!={'query':[2,32]}:
        raise ValueError('Exact original dynamic caller contract required')
    packet=a.accepted_run.parent/f"forward-{forward['step']:02d}-q8/input.bin"
    if sha(packet)!=forward['input_sha256']:raise ValueError('Original ARM cold packet changed')
    library=a.receipt.parent/'dynamic-pv-register-candidate.so'
    if sha(library)!=receipt['library']['sha256']:raise ValueError('Actual candidate changed')
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage,retain_packet
    from hexagon_live_compile_mailbox import atomic_json
    from probe_model_loop_regions_arm import stage_constants
    constants=stage_constants(stage,read(a.catalog),a.workdir,receipt['constants'],a.constant_load_split_proofs)
    libraries={}
    for role,path in [('baseline',a.baseline_library),('candidate',library)]:
        destination=a.workdir/(sha(path)+'.so');shutil.copy2(path,destination)
        if sha(destination)!=sha(path):raise ValueError('Actual model staging changed')
        libraries[role]=str(destination)
    original=packet.read_bytes();out_bytes=receipt['output_bytes'];a.output.mkdir()
    record=dict(status='dynamic_PV_model_three_queries_running_NOT_accepted',source_sha256=sha(__file__),
        receipt_sha256=a.receipt_sha256,catalog_sha256=a.catalog_sha256,accepted_run_sha256=a.accepted_run_sha256,
        accepted_audit_sha256=a.accepted_audit_sha256,accepted_model_gate_sha256=a.accepted_model_gate_sha256,
        accepted_model_audit_sha256=a.accepted_model_audit_sha256,caller_sha256=a.caller_sha256,
        baseline_library_sha256=baseline_sha,candidate_library_sha256=receipt['library']['sha256'],constant_layout=constants,
        original_query8_ARM_CPU_oracle_preserved=True,derived_query2_query32_new_CPU_oracle=False,
        diagnostic_instrumentation=False,power_votes_changed=False,model_inference_acceptance=False,performance_acceptance=False,
        rows=[],cases=[],DSP_calls=0)
    save=lambda:atomic_json(a.output/'result.json',record);save();previous=Path.cwd()
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)+';'+str(a.workdir/'rfsa')
        +';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),libraries['baseline'],str(receipt['input_bytes']),str(out_bytes)]
    try:
        os.chdir(a.workdir)
        with HexagonSharedPacketStream(argv,env,a.output,input_bytes=receipt['input_bytes'],output_bytes=out_bytes,
                initial_library=libraries['baseline'],timeout=240) as stream:
            record['ready']=stream.start();save()
            for q in (8,2,32):
                payload=original if q==8 else resize_cold_packet(caller['contract'],original,q)
                input_path=a.output/('q'+str(q)+'-input.bin');digest,storage=retain_packet(input_path,payload,'full')
                expected=forward['output_sha256'] if q==8 else None
                for index,role in enumerate(('baseline','candidate','candidate','baseline')):
                    record['pending']=dict(query=q,index=index,role=role);save()
                    stream.select_library(libraries[role]);output,event=stream.invoke_packet(payload,out_bytes)
                    record['DSP_calls']+=1;record['pending']['actual_invoke']=event;save()
                    output_path=a.output/('q'+str(q)+'-'+str(index)+'-'+role+'-output.bin')
                    output_sha,output_storage=retain_packet(output_path,output,'full')
                    if expected is None:expected=output_sha
                    if (output_sha!=expected or (q==8 and any(event[k]!=forward['invoke']['AP_input_fnv1a32']
                            for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')))):
                        raise ValueError('Complete actual candidate logits/56KV or original packet FNV differs')
                    record['rows'].append(dict(query=q,index=index,role=role,input_file=input_path.name,input_sha256=digest,
                        input_storage=storage,output_file=output_path.name,output_sha256=output_sha,output_storage=output_storage,
                        all_logits_and_56_KV_byteexact=True,actual_invoke=event));record.pop('pending');save()
                    del output
                record['cases'].append(q);save()
        if stream.process_exit_code or record['DSP_calls']!=12:raise ValueError('Actual complete ABBA12 and normal close required')
        record.update(status='actual_dynamic_PV_model_ABBA12_three_queries_full_packets_bitexact_NOT_state_or_generation_acceptance',
            process_exit_code=stream.process_exit_code,stream_events=stream.events)
        save();os.chdir(previous);shutil.rmtree(a.workdir)
    except BaseException:record.update(status='failed',error=traceback.format_exc());save();raise
    finally:os.chdir(previous)


if __name__=='__main__':main()
