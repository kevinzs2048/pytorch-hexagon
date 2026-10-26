"""Replay all six original ARM-accepted q1 packets in one persistent ABBA session.

Preserve EVERY complete output as a sparse exact byte file for independent
revisit. No token-generation, corpus coverage or production acceptance claimed.
"""
import argparse,hashlib,json,os,platform,shutil,sys
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def retain_exact_sparse(path,data):
    with path.open('xb') as stream:
        # KV head strides are 512 KiB; 1 MiB chunks each contain live rows.
        # Use filesystem-sized holes so short-cache packets stay truly sparse.
        for offset in range(0,len(data),4096):
            block=data[offset:offset+4096]
            if block.count(0)==len(block):stream.seek(len(block),1)
            else:stream.write(block)
        stream.truncate(len(data))
    if sha(path)!=hashlib.sha256(data).hexdigest():raise ValueError('Sparse retention changed output bytes')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['sources','catalog','build','accepted-run','accepted-audit','workdir','output','constant-load-split-proofs']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['catalog','build','accepted-run','accepted-audit']:p.add_argument('--'+name+'-sha256',required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    if platform.machine()!='aarch64' or sys.byteorder!='little' or a.output.exists():raise ValueError('Actual ARM and fresh output required')
    for name in ['catalog','build','accepted_run','accepted_audit']:
        if sha(getattr(a,name))!=getattr(a,name+'_sha256'):raise ValueError('Pinned original packet input changed')
    build=read(a.build);native=read(a.accepted_run);audit=read(a.accepted_audit)
    forwards=[row for row in native['forwards'] if row['query']==1]
    if (build['status']!='actual_q1_runtime_extent_model_linked_DIAGNOSTIC_NOT_model_acceptance'
            or build['model_acceptance'] or build['performance_acceptance']
            or audit['status']!='framework_native_numerical_acceptance_passed' or not audit['numerical_acceptance']
            or audit['run_sha256']!=a.accepted_run_sha256 or native['process_exit_code'] or native['CPU_model_fallback']
            or len(forwards)!=6):raise ValueError('Exact original full ARM-W8A8 packet oracle and unaccepted q1 model required')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage
    from hexagon_live_compile_mailbox import atomic_json
    layout=stage(read(a.catalog),a.workdir,include_code=False,constant_load_split_proofs=a.constant_load_split_proofs)
    if not layout or len(layout)!=3:raise ValueError('Original bounded common constant staging required')
    paths={}
    for role,name,want in [('baseline','baseline-q1.so',build['original_library_replay']['sha256']),('candidate','candidate-q1.so',build['library']['sha256'])]:
        source=a.build.parent/name
        if sha(source)!=want:raise ValueError('Pinned actual model library changed')
        destination=a.workdir/(want+'.so');shutil.copy2(source,destination);paths[role]=str(destination)
        if sha(destination)!=want:raise ValueError('Staged model library changed')
    a.output.mkdir(parents=True);record=dict(status='actual_q1_runtime_extent_model_original24_ABBA_running_DIAGNOSTIC',
        source_sha256=sha(__file__),build_sha256=a.build_sha256,catalog_sha256=a.catalog_sha256,accepted_run_sha256=a.accepted_run_sha256,
        accepted_audit_sha256=a.accepted_audit_sha256,rows=[],DSP_calls=0,model_acceptance=False,performance_acceptance=False,
        generation_regression=False,constant_layout=layout)
    save=lambda:atomic_json(a.output/'result.json',record);save();previous=Path.cwd()
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)+';'+str(a.workdir/'rfsa')+';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    command=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),paths['baseline'],str(build['input_bytes']),str(build['output_bytes'])]
    try:
        os.chdir(a.workdir)
        (a.output/'stream').mkdir()
        with HexagonSharedPacketStream(command,env,a.output/'stream',input_bytes=build['input_bytes'],output_bytes=build['output_bytes'],initial_library=paths['baseline'],timeout=120) as stream:
            stream.start()
            for forward in forwards:
                packet=a.accepted_run.parent/f"forward-{forward['step']:02d}-q1/input.bin"
                if sha(packet)!=forward['input_sha256']:raise ValueError('Exact original accepted full q1 input changed')
                payload=packet.read_bytes()
                if len(payload)!=build['input_bytes']:raise ValueError('Original q1 full packet size changed')
                retain_exact_sparse(a.output/f"input-step{forward['step']:02d}.bin",payload)
                for index,role in enumerate(['baseline','candidate','candidate','baseline']):
                    stream.select_library(paths[role]);output,event=stream.invoke_packet(payload,build['output_bytes']);record['DSP_calls']+=1
                    if (any(event[k] for k in ['rpc_result','DSP_result','output_result'])
                            or any(event[k]!=forward['invoke']['AP_input_fnv1a32'] for k in ['AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'])
                            or hashlib.sha256(output).hexdigest()!=forward['output_sha256']):
                        retain_exact_sparse(a.output/f"failed-{forward['step']}-{index}.bin",output)
                        raise ValueError('Actual full model output or original three input FNV values differ')
                    retained=a.output/f"output-step{forward['step']:02d}-arm{index}-{role}.bin";retain_exact_sparse(retained,output)
                    record['rows'].append(dict(step=forward['step'],query=1,index=index,role=role,input_sha256=forward['input_sha256'],
                        packet_path=str(packet),full_output_sha256=sha(retained),full_output_path=retained.name,full_output_bytes=len(output),
                        actual_invoke=event,cache_positions=forward['cache_positions']));save()
        if stream.process_exit_code:raise ValueError('Actual model stream owner failed')
        record.update(status='actual_q1_runtime_extent_original24_ABBA_complete_full_packets_DIAGNOSTIC_NOT_generation_acceptance',
            process_exit_code=stream.process_exit_code,stream_events=stream.events)
        os.chdir(previous);shutil.rmtree(a.workdir);record['workdir_removed']=True;save()
    except BaseException as exc:record.update(status='failed',error=repr(exc));save();raise
    finally:os.chdir(previous)

if __name__=='__main__':main()
