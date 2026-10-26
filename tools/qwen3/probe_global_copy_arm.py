"""Actual original/candidate/candidate/original copy packets, all bytes and canaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','build','packet','expected','workdir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','build','packet','expected'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--constant-load-split-proofs',type=Path,required=True)
    a=p.parse_args()
    if platform.machine()!='aarch64' or sys.byteorder!='little' or a.output.exists() or not a.output.is_absolute():
        raise ValueError('Actual ARM and fresh absolute operator output required')
    for name in ('catalog','build','packet','expected'):
        if sha(getattr(a,name))!=getattr(a,name+'_sha256'):raise ValueError('Pinned copy probe input changed: '+name)
    build=json.loads(a.build.read_text())
    if (build['status']!='model_local_global_HVX_copy_both_accepted_primaries_relinked_NOT_device_or_generation_acceptance'
            or build['new_model_PRIMARY_compilations'] or build['DSP_calls']
            or build['all_original_model_primaries_and_constants_unchanged'] is not True
            or build['original_fixed512_text_unchanged'] is not True):
        raise ValueError('Actual unchanged-model, exact-delegation build required')
    payload=a.packet.read_bytes();expected=a.expected.read_bytes()
    if payload[:8]!=b'QWCPY1\0\0' or not expected:raise ValueError('Exact independent byte-copy packet required')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage
    from hexagon_live_compile_mailbox import atomic_json
    stage(json.loads(a.catalog.read_text()),a.workdir,include_code=False,
          constant_load_split_proofs=a.constant_load_split_proofs)
    paths={}
    for role in ('baseline','candidate'):
        row=build['primitive'][role]['library'];source=a.build.parent/(role+'-copy-probe.so')
        if sha(source)!=row['sha256']:raise ValueError('Actually built copy primitive changed')
        target=a.workdir/(row['sha256']+'.so');shutil.copy2(source,target)
        if sha(target)!=row['sha256']:raise ValueError('Staged copy primitive changed')
        paths[role]=target
    a.output.mkdir();record=dict(status='running',source_sha256=sha(__file__),build_sha256=a.build_sha256,
        packet_sha256=a.packet_sha256,expected_sha256=a.expected_sha256,rows=[],DSP_calls=0,
        libraries={role:sha(path)for role,path in paths.items()},DSP_model_calls=0,
        model_inference_acceptance=False,performance_acceptance=False)
    save=lambda:atomic_json(a.output/'result.json',record);save();previous=Path.cwd()
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)+';'+str(a.workdir/'rfsa')+
        ';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),str(paths['baseline']),
          str(len(payload)),str(len(expected))]
    try:
        os.chdir(a.workdir)
        with HexagonSharedPacketStream(argv,env,a.output,input_bytes=len(payload),output_bytes=len(expected),
                initial_library=str(paths['baseline']),timeout=120)as stream:
            record['ready']=stream.start();save()
            for index,role in enumerate(('baseline','candidate','candidate','baseline')):
                stream.select_library(str(paths[role]));output,event=stream.invoke_packet(payload,len(expected))
                record['DSP_calls']+=1;path=a.output/('output-'+str(index)+'.bin');path.write_bytes(output)
                record['rows'].append(dict(index=index,role=role,output_file=path.name,
                    output_sha256=sha(path),actual_invoke=event));save()
                if output!=expected or sha(path)!=a.expected_sha256:
                    raise ValueError('Every destination and canary byte must match independent ARM reference')
        if stream.process_exit_code:raise ValueError('Copy operator DSP owner failed')
        record.update(status='model_local_global_HVX_copy_ABBA4_all_bytes_canaries_exact_NOT_model_acceptance',
                      process_exit_code=stream.process_exit_code,stream_events=stream.events)
        save();os.chdir(previous);shutil.rmtree(a.workdir)
    except BaseException as error:record.update(status='failed',error=repr(error));save();raise
    finally:os.chdir(previous)


if __name__=='__main__':main()
