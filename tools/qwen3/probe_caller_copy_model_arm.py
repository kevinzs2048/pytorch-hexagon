"""Original-packet ABBA for caller copying, using both unchanged model objects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','build','accepted-run','accepted-audit','primitive-review','workdir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','build','accepted-run','accepted-audit','primitive-review'):
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--constant-load-split-proofs',type=Path,required=True)
    p.add_argument('--global-copy',action='store_true',help='Explicit separately proved model-local memcpy dispatch')
    a=p.parse_args()
    if platform.machine()!='aarch64' or a.output.exists() or not a.output.is_absolute():
        raise ValueError('Actual ARM and fresh model gate required')
    for name in ('catalog','build','accepted_run','accepted_audit','primitive_review'):
        if sha(getattr(a,name))!=getattr(a,name+'_sha256'):raise ValueError('Pinned caller-copy gate input changed: '+name)
    read=lambda path:json.loads(Path(path).read_text());build=read(a.build);run=read(a.accepted_run)
    audit=read(a.accepted_audit);primitive=read(a.primitive_review)
    build_status=('model_local_global_HVX_copy_both_accepted_primaries_relinked_NOT_device_or_generation_acceptance'
        if a.global_copy else 'caller_only_HVX_byte_copy_two_original_models_built_NOT_device_or_generation_acceptance')
    primitive_status=('actual_DSP_model_local_global_copy_498_cases_ABBA4_complete_bytes_canaries_3FNV_and_normal_owner_independently_revisited_NOT_model_acceptance'
        if a.global_copy else 'actual_DSP_caller_copy_498_bounded_cases_complete_bytes_canaries_3FNV_and_normal_owner_independently_revisited_NOT_model_acceptance')
    if (build['status']!=build_status
            or build['new_model_PRIMARY_compilations'] or not build['model_math_and_objects_unchanged']
            or not build['owned_input_and_returned_buffer_lifetimes_unchanged']
            or audit['status']!='framework_native_numerical_acceptance_passed' or not audit['numerical_acceptance']
            or audit['run_sha256']!=a.accepted_run_sha256 or run['process_exit_code'] or run['CPU_model_fallback']
            or primitive['status']!=primitive_status
            or primitive['DSP_primitive_calls']!=(4 if a.global_copy else 1)
            or primitive['DSP_model_calls'] or primitive['model_inference_acceptance']
            or (a.global_copy and primitive.get('positive_primitive_gain') is not True)):
        raise ValueError('Own actual byte primitive and original ARM-exact complete model run required')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage
    from hexagon_live_compile_mailbox import atomic_json
    layout=stage(read(a.catalog),a.workdir,include_code=False,constant_load_split_proofs=a.constant_load_split_proofs)
    if not layout or len(layout)!=3:raise ValueError('All3 bounded original constant payloads required')
    models={}
    for model in build['models']:
        key=model['query_bounds'][0];original_name='baseline-q1.so' if key==1 else 'baseline-q2.so'
        candidate_name='candidate-q1.so' if key==1 else 'candidate-q2.so'
        original=a.build.parent/original_name;candidate=a.build.parent/candidate_name
        forward=next(row for row in run['forwards'] if row['query']==(1 if key==1 else 8))
        if sha(original)!=forward['DSO_sha256'] or sha(candidate)!=model['library']['sha256']:
            raise ValueError('Exact qualified baseline and actual caller-only candidate required')
        paths={}
        for role,path in (('baseline',original),('candidate',candidate)):
            target=a.workdir/(sha(path)+'.so');shutil.copy2(path,target)
            if sha(target)!=sha(path):raise ValueError('Staged model DSO changed')
            paths[role]=target
        models[1 if key==1 else 8]=paths
    a.output.mkdir(parents=True)
    record=dict(status='running',source_sha256=sha(__file__),build_sha256=a.build_sha256,
        accepted_run_sha256=a.accepted_run_sha256,accepted_audit_sha256=a.accepted_audit_sha256,
        primitive_review_sha256=a.primitive_review_sha256,rows=[],cases=[],DSP_calls=0,
        model_inference_acceptance=False,performance_acceptance=False,new_model_PRIMARY_compilations=0,
        candidate_kind='model-local-global-hvx-copy' if a.global_copy else 'caller-only-hvx-copy')
    save=lambda:atomic_json(a.output/'result.json',record);save();previous=Path.cwd()
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)+';'+str(a.workdir/'rfsa')+';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    try:
        os.chdir(a.workdir)
        for query in (1,8):
            forward=next(row for row in run['forwards'] if row['query']==query)
            packet=a.accepted_run.parent/f"forward-{forward['step']:02d}-q{query}/input.bin"
            if sha(packet)!=forward['input_sha256']:raise ValueError('Original complete accepted input packet changed')
            payload=packet.read_bytes();output_bytes=forward['invoke']['output_bytes'];paths=models[query]
            argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),str(paths['baseline']),str(len(payload)),str(output_bytes)]
            case_output=a.output/f'q{query}';case_output.mkdir()
            if a.global_copy:
                from hexagon_sparse_packet_storage import sparsify_verified_packet
                kept=case_output/'input.bin';kept.write_bytes(payload)
                sparsify_verified_packet(kept,forward['input_sha256'])
            with HexagonSharedPacketStream(argv,env,case_output,input_bytes=len(payload),output_bytes=output_bytes,
                    initial_library=str(paths['baseline']),timeout=120) as stream:
                stream.start()
                for index,role in enumerate(('baseline','candidate','candidate','baseline')):
                    stream.select_library(str(paths[role]));output,event=stream.invoke_packet(payload,output_bytes)
                    record['DSP_calls']+=1;digest=hashlib.sha256(output).hexdigest()
                    if (any(event[k]!=0 for k in ('rpc_result','DSP_result','output_result'))
                            or any(event[k]!=forward['invoke']['AP_input_fnv1a32'] for k in
                                ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'))):
                        raise ValueError('Actual full original packet or normal DSP invocation changed')
                    if digest!=forward['output_sha256']:
                        (case_output/'failed-output.bin').write_bytes(output)
                        if a.global_copy:
                            from tools.qwen3.probe_q2_dispatch_model_arm import first_difference
                            reference=packet.with_name('output.bin').read_bytes()
                            if hashlib.sha256(reference).hexdigest()!=forward['output_sha256']:
                                raise ValueError('Original ARM-exact failure reference changed')
                            contract=next(row['contract']for row in run['captures']
                                if row['DSO_sha256']==forward['DSO_sha256'])
                            record['first_differences']=first_difference(contract,query,reference,output)
                            save()
                        raise ValueError('Full model output differs from independent original ARM packet')
                    record['rows'].append(dict(query=query,index=index,role=role,input_sha256=forward['input_sha256'],
                        full_output_sha256=digest,all_logits_and_56_KV_words_bitexact=True,actual_invoke=event));save()
                    if a.global_copy:
                        kept=case_output/('output-'+str(index)+'.bin');kept.write_bytes(output)
                        sparsify_verified_packet(kept,digest)
                        record['rows'][-1].update(input_file=f'q{query}/input.bin',
                            output_file=f'q{query}/output-{index}.bin')
                        save()
            if stream.process_exit_code:raise ValueError('Actual model DSP owner failed')
            record['cases'].append(dict(query=query,process_exit_code=stream.process_exit_code,stream_events=stream.events));save()
        status=('model_local_global_copy_original_q1_q8_ABBA8_all_full_packets_exact_NOT_generation_acceptance'
            if a.global_copy else 'caller_only_byte_copy_original_q1_q8_ABBA8_all_full_packets_exact_NOT_generation_acceptance')
        record.update(status=status,process_exit_code=0);save()
        os.chdir(previous);shutil.rmtree(a.workdir)
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));save();raise
    finally:os.chdir(previous)


if __name__=='__main__':main()
