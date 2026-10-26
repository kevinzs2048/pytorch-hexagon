"""Compare actual current/tail QK objects in four rounding modes on ARM/DSP."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from q2_tail_packet import CASES,INPUT_CAPACITY,OUTPUT_CAPACITY,make_case,parse_output


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','receipt','workdir','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','receipt'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    a=p.parse_args()
    if (platform.machine()!='aarch64' or sys.byteorder!='little' or a.output.exists() or a.workdir.exists()
            or not a.output.is_absolute() or sha(a.catalog)!=a.catalog_sha256 or sha(a.receipt)!=a.receipt_sha256
            or a.deadline_unix-time.time()<600):raise ValueError('Actual ARM and fresh bounded pinned run required')
    receipt=json.loads(a.receipt.read_text())
    if (receipt['status']!='actual_Q2_row_major_guard_same_operators_linked_NOT_target_or_model'
            or receipt['DSP_calls'] or receipt['new_model_PRIMARY_compilations']
            or receipt['model_inference_acceptance'] or set(receipt['variants'])!={'baseline','tail'}):
        raise ValueError('Actual independently checked current/tail operator objects required')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import validate_workdir
    from hexagon_live_compile_mailbox import atomic_json
    import numpy as np
    validate_workdir(a.workdir)
    if subprocess.run(['pgrep','-x','buffer_stream'],capture_output=True,timeout=10).returncode!=1:
        raise ValueError('Another DSP owner exists')
    a.output.mkdir();a.workdir.mkdir();(a.workdir/'rfsa').mkdir()
    source=a.receipt.parent/'dynamic_qk_reference.c'
    if not any(Path(r['path']).name==source.name and r['sha256']==sha(source)for r in receipt['sources']):
        raise ValueError('Independent original scalar ARM reference changed')
    library=a.output/'ARM-reference.so';compiler=Path(shutil.which('gcc'))
    command=[str(compiler),'-O2','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',
        '-fno-builtin-fmaf',str(source),'-lm','-o',str(library)]
    subprocess.run(command,check=True,capture_output=True,timeout=60)
    reference=ctypes.CDLL(str(library));ptr=ctypes.POINTER(ctypes.c_float)
    fn=reference.qwen_dynamic_qk_reference;fn.argtypes=[ptr]*4+[ctypes.c_uint]*3;fn.restype=ctypes.c_int
    fnv=reference.qwen_dynamic_qk_fnv;fnv.argtypes=[ctypes.c_void_p,ctypes.c_uint];fnv.restype=ctypes.c_uint32
    libm=ctypes.CDLL('libm.so.6')
    if libm.fegetround()!=0:raise ValueError('Independent ARM reference must use its actual default rounding')
    catalog=json.loads(a.catalog.read_text());control=Path(catalog['control_deployment'])
    for row in catalog['runtime']['artifacts']:
        relative=Path(row['name'])if row['name']=='buffer_stream'else Path('rfsa')/row['name']
        if row['name']not in ('buffer_stream','libhexagon_buffer_runner_skel.so')or sha(control/relative)!=row['sha256']:
            raise ValueError('Original AP/DSP runtime changed')
        shutil.copy2(control/relative,a.workdir/relative)
    staged={}
    for kind,row in receipt['variants'].items():
        original=a.receipt.parent/(kind+'.so');target=a.workdir/(row['library']['sha256']+'.so')
        if sha(original)!=row['library']['sha256']:raise ValueError('Actual linked operator changed')
        shutil.copy2(original,target);staged[kind]=str(target)
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)+';'+str(a.workdir/'rfsa')
        +';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),staged['baseline'],str(INPUT_CAPACITY),str(OUTPUT_CAPACITY)]
    record=dict(status='actual_Q2_identical_tail_operator_running',source_sha256=sha(__file__),
        packet_source_sha256=sha(Path(__file__).with_name('q2_tail_packet.py')),
        receipt_sha256=a.receipt_sha256,catalog_sha256=a.catalog_sha256,
        ARM_reference=dict(command=command,compiler_sha256=sha(compiler),source_sha256=sha(source),
            library_sha256=sha(library),scope='nearest_finite_only; all_other_cases_compare_actual_current_DSP_bits_and_FP_flags'),
        rows=[],cases=[],actual_DSP_calls=0,model_inference_acceptance=False,performance_acceptance=False,
        power_votes_changed=False,deadline_unix=a.deadline_unix)
    save=lambda:atomic_json(a.output/'result.json',record);save();previous=Path.cwd()
    try:
        os.chdir(a.workdir)
        with HexagonSharedPacketStream(argv,env,a.output,input_bytes=INPUT_CAPACITY,output_bytes=OUTPUT_CAPACITY,
                initial_library=staged['baseline'],timeout=180)as stream:
            record['ready']=stream.start();save()
            for q,n,flag,pattern,extent,rounding in CASES:
                keys,query,initial,payload=make_case(np,q,n,flag,pattern,extent,rounding)
                name=f'q{q}-n{n}-b{flag}-{pattern}-r{rounding}'
                input_path=a.output/(name+'-input.bin');input_path.write_bytes(payload)
                scalar=int(fnv(ctypes.c_char_p(payload),len(payload)))
                arm=None;arm_path=None
                if rounding==0 and pattern not in ('infinity','qnan','snan'):
                    arm=np.empty_like(initial)
                    if fn(*(x.ctypes.data_as(ptr)for x in (keys,query,initial,arm)),q,n,flag):
                        raise ValueError('Independent ARM recurrence failed')
                    arm_path=a.output/(name+'-ARM-f32.bin');arm_path.write_bytes(arm.tobytes())
                baseline=None;baseline_flags=None;record['pending_case']=name;save()
                for order,kind in enumerate(('baseline','tail','tail','baseline')):
                    if a.deadline_unix-time.time()<60:raise TimeoutError('Deadline before next operator invocation')
                    stream.select_library(staged[kind]);response,event=stream.invoke_packet(payload,128+initial.nbytes)
                    record['actual_DSP_calls']+=1
                    output_path=a.output/(name+'-'+str(order)+'-'+kind+'-DSP.bin');output_path.write_bytes(response)
                    record['pending_invoke']=dict(case=name,order=order,kind=kind,actual_invoke=event,
                        input_file=input_path.name,input_sha256=sha(input_path),output_file=output_path.name,
                        output_sha256=sha(output_path));save()
                    parsed=parse_output(response,q,n,flag,rounding,extent,kind)
                    if order==0:baseline=response[128:];baseline_flags=parsed['FP_flags']
                    if (response[128:]!=baseline or parsed['FP_flags']!=baseline_flags
                            or (arm is not None and response[128:]!=arm.tobytes())
                            or any(event[k]!=scalar for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'))
                            or event['kernel_us']!=parsed['kernel_us']):
                        oracle=arm.reshape(-1).view(np.uint32)if arm is not None else np.frombuffer(baseline,dtype='<u4')
                        actual=np.frombuffer(response,dtype='<u4',offset=128);different=np.flatnonzero(actual!=oracle)
                        record['failed_word_comparison']=dict(total_different_words=int(different.size),
                            first_differences=[dict(word=int(i),head=int(i)//(q*n),row=(int(i)//n)%q,column=int(i)%n,
                                expected_bits=int(oracle[i]),actual_bits=int(actual[i]))for i in different[:64]])
                        save();raise ValueError('Actual target/ARM bits, FP flags or full-input FNV differ')
                    record['rows'].append(dict(case=name,query=q,columns=n,broadcast=flag,pattern=pattern,rounding=rounding,
                        extent=extent,order=order,kind=kind,input_file=input_path.name,input_sha256=sha(input_path),
                        output_file=output_path.name,output_sha256=sha(output_path),
                        ARM_reference_file=arm_path.name if arm_path else None,
                        ARM_reference_sha256=sha(arm_path)if arm_path else None,
                        ARM_reference_checked=arm is not None,all_target_FP32_bits_and_FP_flags_equal=True,
                        sequential_scalar_input_FNV=scalar,parsed=parsed,actual_invoke=event))
                    record.pop('pending_invoke',None);save()
                record['cases'].append(name);record.pop('pending_case',None);save()
        if stream.process_exit_code or len(record['rows'])!=4*len(CASES)or len(record['cases'])!=len(CASES):
            raise ValueError('All64 cases, ABBA256 and normal DSP owner exit required')
        record.update(status='actual_Q2_identical_tail_ABBA256_all_target_bits_four_rounding_FP_flags_complete_NOT_model',
            DSP_calls=4*len(CASES),process_exit_code=stream.process_exit_code,stream_events=stream.events)
        save();os.chdir(previous);shutil.rmtree(a.workdir)
    except BaseException:record.update(status='failed',error=traceback.format_exc());save();raise
    finally:os.chdir(previous)


if __name__=='__main__':main()
