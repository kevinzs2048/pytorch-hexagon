"""Compare original/scoped SiLU chains on real DSP, preserving all FP words.

Unwrapped ABBA libraries supply standalone timings. Separate counted libraries
establish actual multiplicity. Finite cases also require an independent native
ARM evaluation; special values must preserve the original DSP result exactly.
"""
import argparse
import ctypes
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
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def patterns(np,rows=1):
    if type(rows) is not int or not 1<=rows<=32:raise ValueError('Bounded dynamic row extent required')
    elements=3072*rows
    def edge_rows(values):
        base=np.resize(values,3072)
        return base if rows==1 else np.concatenate([np.roll(base,row) for row in range(rows)])
    edges=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,
        0x00800000,0x80800000,0x3f000000,0xbf000000,0x3f7fffff,0x3f800000,
        0x3f800001,0xbf800000,0x42cfffff,0x42d00000,0x42d00001,0xc2c80000,
        0xc2c80001,0x7f7fffff,0xff7fffff],dtype=np.uint32).view(np.float32)
    rng=np.random.default_rng(20261006)
    yield 'finite-boundaries',edge_rows(edges),edge_rows(np.array([1,-1,0.5,0,-0.0],dtype=np.float32)),True
    x=np.linspace(-16,16,elements,dtype=np.float32)
    yield 'finite-range',x,rng.uniform(-8,8,elements).astype(np.float32),True
    yield 'finite-adjacent',np.nextafter(x,np.float32(np.inf)),rng.uniform(-8,8,elements).astype(np.float32),True
    yield 'finite-seeded',rng.normal(0,4,elements).astype(np.float32),rng.normal(0,2,elements).astype(np.float32),True
    yield 'signed-zero-subnormal',edge_rows(edges[:8]),np.ones(elements,dtype=np.float32),True
    special=np.array([0x7f800000,0xff800000,0x7fc00001,0xffc00123,0x7f800001,0xff800001],dtype=np.uint32).view(np.float32)
    yield 'nonfinite-original-DSP-contract',edge_rows(special),np.ones(elements,dtype=np.float32),False


def parse_response(data,repeats,expected_calls,rows=1):
    if type(rows) is not int or not 1<=rows<=32 or len(data)!=128+4*3072*rows:raise ValueError('Wrong SiLU packet extent')
    elapsed,actual,cycles,calls=struct.unpack_from('<QI4xQQ',data)
    if not elapsed or actual!=repeats or not cycles or calls!=expected_calls:
        raise ValueError('Incomplete actual timing, repeat, or helper-call evidence')
    if data[32:40].hex()!='82c100efc03f2050' or any(data[12:16]) or any(data[40:128]):
        raise ValueError('Actual guarded fmaf body or reserved metadata changed')
    return dict(elapsed_us=elapsed,repeats=actual,processor_cycles=cycles,
                helper_calls=calls,effective_active_cycles_per_us=cycles/elapsed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','receipt','workdir','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','receipt'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--repeats',type=int,default=128)
    p.add_argument('--rows',type=int,default=1,help='Explicit dynamic primitive row extent; original query1 probe remains default')
    a=p.parse_args()
    if (platform.machine()!='aarch64' or sys.byteorder!='little' or not a.output.is_absolute()
            or a.output.exists() or not 1<=a.repeats<=1000 or not 1<=a.rows<=32):raise ValueError('Actual ARM and fresh bounded run required')
    if sha(a.catalog)!=a.catalog_sha256 or sha(a.receipt)!=a.receipt_sha256:raise ValueError('Pinned inputs changed')
    receipt=json.loads(a.receipt.read_text());catalog=json.loads(a.catalog.read_text())
    if (receipt['status']!='tiny_silu_effects_libraries_built_NOT_device_or_model_acceptance'
            or receipt['model_inference'] or receipt['FP_evaluation_order_changed'] or receipt['power_votes_changed']
            or receipt['options']['addFastMath']!='False' or len(receipt['rows'])!=4):raise ValueError('Pinned standalone probe required')
    dynamic=receipt.get('dynamic_row',False)
    if not dynamic and a.rows!=1:raise ValueError('Original static probe admits one row only')
    for row in receipt['sources']:
        if sha(a.receipt.parent/Path(row['path']).name)!=row['sha256']:raise ValueError('Probe source changed')
    libraries={}
    for row in receipt['rows']:
        if row['returncode'] or sha(a.receipt.parent/Path(row['library']['path']).name)!=row['library']['sha256']:
            raise ValueError('Probe library changed')
        libraries[row['name']]=a.receipt.parent/Path(row['library']['path']).name
    if set(libraries)!={'unknown','declaration-contract','unknown-counted','declaration-contract-counted'}:
        raise ValueError('Original/candidate and separate diagnostic libraries required')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import validate_workdir
    from hexagon_live_compile_mailbox import atomic_json
    import numpy as np
    validate_workdir(a.workdir)
    active=subprocess.run(['pgrep','-x','buffer_stream'],capture_output=True,timeout=10)
    if active.returncode not in (0,1) or active.stdout.strip():raise ValueError('DSP owner still active')
    a.output.mkdir(parents=True)
    gcc=Path(shutil.which('g++') or '')
    if not gcc.is_file():raise ValueError('Native ARM C++ compiler unavailable')
    reference=a.output/'ARM-original-sequence.so'
    command=[str(gcc),'-O3','-fPIC','-shared','-fno-fast-math','-ffp-contract=off','-fno-builtin-fmaf',
        str(a.receipt.parent/'silu_effects_reference.cpp'),'-lm','-o',str(reference)]
    compiled=subprocess.run(command,capture_output=True,timeout=60)
    (a.output/'reference-compile.stdout.txt').write_bytes(compiled.stdout);(a.output/'reference-compile.stderr.txt').write_bytes(compiled.stderr)
    compiled.check_returncode()
    native=ctypes.CDLL(str(reference)).qwen_silu_effects_reference
    pointer=ctypes.POINTER(ctypes.c_float);native.argtypes=[pointer,pointer,pointer,ctypes.c_int];native.restype=None
    a.workdir.mkdir();(a.workdir/'rfsa').mkdir();control=Path(catalog['control_deployment'])
    for row in catalog['runtime']['artifacts']:
        relative=Path(row['name']) if row['name']=='buffer_stream' else Path('rfsa')/row['name']
        if row['name'] not in ('buffer_stream','libhexagon_buffer_runner_skel.so') or sha(control/relative)!=row['sha256']:
            raise ValueError('Pinned AP/DSP transport changed')
        shutil.copy2(control/relative,a.workdir/relative)
    staged={}
    for name,path in libraries.items():
        target=a.workdir/(sha(path)+'.so');shutil.copy2(path,target)
        if sha(target)!=sha(path):raise ValueError('Staged primitive changed')
        staged[name]=target
    elements=3072*a.rows;input_bytes=128+8*elements;output_bytes=128+4*elements
    word_proof=(dict(all_FP32_words_identical_to_original_DSP=True,FP32_words_per_iteration=elements)
                if dynamic else dict(all_3072_FP32_words_identical_to_original_DSP=True))
    env=dict(os.environ,LD_LIBRARY_PATH=str(a.workdir),ADSP_LIBRARY_PATH=str(a.workdir)
        +';'+str(a.workdir/'rfsa')+';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),str(staged['unknown']),str(input_bytes),str(output_bytes)]
    record=dict(status='actual_DSP_silu_effects_running',model_inference=False,source_sha256=sha(__file__),
        receipt_sha256=a.receipt_sha256,catalog_sha256=a.catalog_sha256,power_votes_changed=False,
        ARM_reference=dict(command=command,compiler_sha256=sha(gcc),library_sha256=sha(reference)),
        scope='standalone_SiLU_original_unconstrained_FP_sequence_NOT_full_model_acceptance',rows=[],diagnostic_rows=[])
    if dynamic:record.update(dynamic_row=True,row_extent=a.rows,FP32_words_per_iteration=elements)
    def save():atomic_json(a.output/'result.json',record)
    save();previous=Path.cwd()
    try:
        os.chdir(a.workdir)
        with HexagonSharedPacketStream(argv,env,a.output,input_bytes=input_bytes,output_bytes=output_bytes,
                initial_library=str(staged['unknown']),timeout=120) as stream:
            record['ready']=stream.start();save()
            for pattern,x,other,finite in patterns(np,a.rows):
                expected=np.empty(elements,dtype=np.float32);native(x.ctypes.data_as(pointer),other.ctypes.data_as(pointer),expected.ctypes.data_as(pointer),elements)
                header=(b'QWSILU2\0'+struct.pack('<II',a.repeats,a.rows)+bytes(112) if dynamic else
                        b'QWSILU1\0'+struct.pack('<I',a.repeats)+bytes(116))
                payload=header+x.tobytes()+other.tobytes()
                (a.output/(pattern+'-input.bin')).write_bytes(payload);(a.output/(pattern+'-ARM.bin')).write_bytes(expected.tobytes())
                baseline=None
                for index,name in enumerate(('unknown','declaration-contract','declaration-contract','unknown')):
                    record['pending']=dict(pattern=pattern,index=index,library=name);save()
                    stream.select_library(str(staged[name]));output,event=stream.invoke_packet(payload,output_bytes)
                    (a.output/(pattern+'-'+str(index)+'-DSP.bin')).write_bytes(output)
                    timing=parse_response(output,a.repeats,0,a.rows);value=output[128:]
                    if baseline is None:baseline=value
                    if value!=baseline:raise ValueError('Scoped effects changed original DSP result: '+pattern)
                    if finite and value!=expected.tobytes():raise ValueError('Finite original sequence differs from independent ARM: '+pattern)
                    record['rows'].append(dict(pattern=pattern,index=index,library=name,diagnostic_call_counter=False,
                        **word_proof,independent_ARM_bits_identical=value==expected.tobytes(),
                        independent_ARM_match_required=finite,input_sha256=hashlib.sha256(payload).hexdigest(),
                        full_output_sha256=hashlib.sha256(output).hexdigest(),value_sha256=hashlib.sha256(value).hexdigest(),
                        actual_invoke=event,**timing));save()
                if pattern=='finite-seeded':
                    count_payload=payload[:8]+struct.pack('<I',1)+payload[12:]
                    for name,calls in [('unknown-counted',3*elements),('declaration-contract-counted',elements)]:
                        stream.select_library(str(staged[name]));output,event=stream.invoke_packet(count_payload,output_bytes)
                        (a.output/(name+'-DSP.bin')).write_bytes(output);timing=parse_response(output,1,calls,a.rows)
                        if output[128:]!=baseline:raise ValueError('Diagnostic call counter changed outputs')
                        record['diagnostic_rows'].append(dict(library=name,diagnostic_call_counter=True,
                            **word_proof,actual_invoke=event,**timing));save()
        if stream.process_exit_code:raise ValueError('DSP owner exit failed')
        record.pop('pending',None)
        record.update(status='actual_DSP_scoped_SiLU_effects_original_bits_and_finite_ARM_sequence_passed',
            DSP_calls=len(record['rows'])+len(record['diagnostic_rows']),process_exit_code=stream.process_exit_code,
            stream_events=stream.events,model_inference_acceptance=False)
        save();os.chdir(previous);shutil.rmtree(a.workdir)
    except BaseException:record.update(status='failed',error=traceback.format_exc());save();raise
    finally:os.chdir(previous)
    print(record['status'])


if __name__=='__main__':main()
