"""Replay an accepted q1 packet with a frozen and diagnostic model library.

All complete outputs must have the accepted byte hash. ABBA replay measures
instrumentation cost; diagnostic queries are separate RPCs, not model calls.
Only external wrapped memory calls are counted, never inlined copy instructions.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import struct
import sys
import traceback


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fmaf_inputs():
    # Finite binary32 boundaries, exact cancellation and independent random
    # triples. This observes the implementation; it is not an all-input proof.
    edge=(0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x80800000,
          0x3f000000,0xbf000000,0x3f7fffff,0x3f800000,0xbf800000,
          0x3f800001,0x40000000,0xc0000000,0x7f7fffff,0xff7fffff,0x4b800000)
    triples=[(a,b,c) for a in edge for b in edge for c in edge]
    rng=random.Random(0x51464d41)
    def finite():
        while True:
            word=rng.getrandbits(32)
            if (word&0x7f800000)!=0x7f800000:return word
    triples.extend(tuple(finite() for _ in range(3)) for _ in range(512))
    return b'QWFMA01\0'+struct.pack('<II',len(triples),0)+b''.join(struct.pack('<III',*t) for t in triples)


def parse_fmaf_effects(data,payload):
    if len(payload)<16 or payload[:8]!=b'QWFMA01\0':raise ValueError('Invalid fmaf query')
    count,reserved=struct.unpack_from('<II',payload,8)
    if not 1<=count<=8192 or reserved or len(payload)!=16+12*count:
        raise ValueError('Invalid bounded fmaf input')
    if len(data)!=256+4*count or data[:8]!=b'QWFMA01\0':raise ValueError('Invalid fmaf response')
    version,actual,address,default_address,found,symbol_address=struct.unpack_from('<IIIIiI',data,8)
    if version!=1 or actual!=count or not address or address!=default_address:
        raise ValueError('Imported and global fmaf provider are not the same actual symbol')
    before,after=struct.unpack_from('<ii',data,40)
    if before!=0x1234 or after!=before:raise ValueError('Actual fmaf changed errno')
    if any(data[240:256]):raise ValueError('Unexpected fmaf metadata')
    libc=ctypes.CDLL('libm.so.6');callee=libc.fmaf
    callee.argtypes=[ctypes.c_float]*3;callee.restype=ctypes.c_float
    for i in range(count):
        values=struct.unpack_from('<fff',payload,16+12*i)
        expected=struct.pack('<f',callee(*values));actual=data[256+4*i:260+4*i]
        if actual!=expected:raise ValueError('DSP/independent ARM libm fmaf bits differ at triple '+str(i))
    def name(start,size):
        raw=data[start:start+size]
        if b'\0' not in raw:raise ValueError('Unterminated runtime symbol name')
        return raw.split(b'\0',1)[0].decode('utf-8')
    return dict(imported_address=address,default_address=default_address,
        dladdr_result=found,symbol_address=symbol_address,provider=name(48,160),symbol=name(208,32),
        first_eight_code_bytes=data[32:40].hex(),finite_test_triples=count,
        independent_ARM_libm_result_bits_identical=True,errno_unchanged=True,
        observations_only_not_purity_or_model_optimization_acceptance=True)


def parse_profile(data,expected_kinds=(1,2,3),required_kinds=(1,2,3)):
    if (not expected_kinds or len(set(expected_kinds))!=len(expected_kinds)
            or any(type(k) is not int or not 1<=k<=16 for k in expected_kinds)
            or not set(required_kinds)<=set(expected_kinds)):
        raise ValueError('Unknown external profiling categories')
    if len(data)!=128+4096*48 or data[:8]!=b'QWMEM01\0':
        raise ValueError('Invalid memory profile packet')
    version,slots,overflow,invocations,row_bytes,entry_address=struct.unpack_from('<IIQQII',data,8)
    if (version,slots,row_bytes)!=(1,4096,48) or overflow or not invocations or not entry_address:
        raise ValueError('Incomplete or overflowed memory profile')
    rows=[];totals={str(k):dict(calls=0,bytes=0,microseconds=0,cycles=0) for k in expected_kinds}
    for offset in range(128,len(data),48):
        kind,site,size,calls,bytes_,us,cycles=struct.unpack_from('<IIQQQQQ',data,offset)
        if not kind:
            if any((site,size,calls,bytes_,us,cycles)):raise ValueError('Nonempty unused profile row')
            continue
        if kind not in expected_kinds or not calls or bytes_!=size*calls:
            raise ValueError('Invalid external-call count or byte coverage')
        row=dict(kind=kind,return_address=site,relative_to_model_entry=site-entry_address,
            size=size,calls=calls,bytes=bytes_,microseconds=us,cycles=cycles)
        rows.append(row)
        for key in totals[str(kind)]:totals[str(kind)][key]+=row[key]
    if not all(totals[str(kind)]['calls'] for kind in required_kinds):
        raise ValueError('Actual model memory wrapper calls were not observed')
    return dict(invocations=invocations,entry_address=entry_address,overflow=overflow,
        rows=rows,totals=totals,scope='external_wrapped_calls_during_model_entry_only')


def accepted_baseline_sha(gate,query,forward,run_sha,audit_sha,allow_query1_cse=False,
                          allow_query1_gqa_value_view=False):
    scoped=(gate['status']=='actual_same_input_dynamic_integer_model_full_packet_gate_passed'
            and gate['candidate_kind']=='multi-integer')
    q1_cse=(allow_query1_cse and query==1
            and gate['status']=='actual_same_input_q1_empty_tensor_CSE_model_full_packet_gate_passed'
            and gate['candidate_kind']=='empty-tensor-cse-q1')
    q1_view=(allow_query1_gqa_value_view and query==1
            and gate['status']=='actual_same_input_q1_GQA_value_view_model_full_packet_gate_passed'
            and gate['candidate_kind']=='gqa-value-view-q1')
    if (not (scoped or q1_cse or q1_view) or gate['query']!=query
            or gate['DSP_calls']!=8 or len(gate['rows'])!=8 or gate['process_exit_code']
            or gate['inline_CPU_observer'] is not False or gate['diagnostic_instrumentation'] is not False
            or gate['accepted_run_sha256']!=run_sha or gate['accepted_audit_sha256']!=audit_sha
            or gate['input_sha256']!=forward['input_sha256']
            or gate['original_complete_output_sha256']!=forward['output_sha256']):
        raise ValueError('Complete original ARM packet and actual baseline integer model gate required')
    for i,row in enumerate(gate['rows']):
        if (row['library']!=('baseline','candidate','candidate','baseline')[i%4]
                or row['index']!=i%4 or row['round']!=i//4
                or row['all_logits_and_56_KV_words_bitexact'] is not True
                or row['full_output_sha256']!=forward['output_sha256']):
            raise ValueError('Complete exact original-packet ABBA rows required')
        if q1_cse or q1_view:
            if any(row['actual_invoke'].get(key)!=forward['invoke']['AP_input_fnv1a32']
                   for key in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')):
                raise ValueError('Complete original packet FNV contract changed')
        if q1_view:
            invoke=row['actual_invoke']
            if (any(invoke.get(key)!=forward['invoke'][key] for key in ('input_bytes','output_bytes'))
                    or any(invoke.get(key)!=0 for key in ('rpc_result','DSP_result','output_result'))):
                raise ValueError('Actual full original packet size and normal result required')
    return gate['candidate_library_sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','catalog','receipt','accepted-run','accepted-audit','baseline-library','workdir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('catalog','receipt','accepted-run','accepted-audit'):
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--query',type=int,default=1)
    parser.add_argument('--constant-load-split-proofs',type=Path)
    parser.add_argument('--constant-loader-mode',choices=('verified-payload-preserving',))
    parser.add_argument('--accepted-model-gate',type=Path,
                        help='Actual completed same-original-packet gate for the explicit diagnostic baseline library')
    parser.add_argument('--accepted-model-gate-sha256')
    args=parser.parse_args()
    if bool(args.constant_load_split_proofs)!=bool(args.constant_loader_mode):
        parser.error('Paired explicit verified constant loader policy required')
    if platform.machine()!='aarch64' or sys.byteorder!='little' or args.output.exists() or not args.output.is_absolute():
        raise ValueError('Actual ARM and fresh absolute output required')
    for name in ('catalog','receipt','accepted_run','accepted_audit'):
        if sha(getattr(args,name))!=getattr(args,name+'_sha256'):
            raise ValueError('Pinned diagnostic or acceptance input changed: '+name)
    receipt=json.loads(args.receipt.read_text());run=json.loads(args.accepted_run.read_text())
    audit=json.loads(args.accepted_audit.read_text());catalog=json.loads(args.catalog.read_text())
    dynamic=receipt.get('query_bounds')==[2,32]
    if ((dynamic and (not 2<=args.query<=32 or not receipt.get('input_bytes_are_maximum_capacity')))
            or (not dynamic and (args.query!=1 or 'query_bounds' in receipt))):
        raise ValueError('Explicit query must be within the original compiled graph contract')
    if (receipt['status']!='diagnostic_model_memory_library_built_NOT_device_acceptance'
            or not receipt['model_object_unchanged'] or receipt['model_graph_changed']
            or receipt['power_votes_changed'] or audit['status']!='framework_native_numerical_acceptance_passed'
            or not audit['numerical_acceptance'] or audit['run_sha256']!=args.accepted_run_sha256
            or run['process_exit_code'] or run['CPU_model_fallback']):
        raise ValueError('Frozen diagnostic and independent original acceptance required')
    forward=next(row for row in run['forwards'] if row['query']==args.query)
    if bool(args.accepted_model_gate)!=bool(args.accepted_model_gate_sha256):
        raise ValueError('Paired explicit original-packet model gate identity required')
    baseline_sha=forward['DSO_sha256']
    if args.accepted_model_gate:
        if sha(args.accepted_model_gate)!=args.accepted_model_gate_sha256:
            raise ValueError('Actual baseline model gate changed')
        gate=json.loads(args.accepted_model_gate.read_text())
        q1_cse=(not dynamic and args.query==1 and gate.get('candidate_kind')=='empty-tensor-cse-q1')
        if not (dynamic or q1_cse):raise ValueError('Explicit original dynamic integer or query1 CSE gate required')
        baseline_sha=accepted_baseline_sha(gate,args.query,forward,args.accepted_run_sha256,args.accepted_audit_sha256,
                                           allow_query1_cse=q1_cse)
    packet=args.accepted_run.parent/f"forward-{forward['step']:02d}-q{args.query}/input.bin"
    if sha(packet)!=forward['input_sha256'] or sha(args.baseline_library)!=baseline_sha:
        raise ValueError('Accepted complete input or original model DSO changed')
    library=args.receipt.parent/'memory-profile.so'
    if sha(library)!=receipt['library']['sha256']:raise ValueError('Diagnostic model DSO changed')
    sys.path[:0]=[str(args.sources/'models/qwen3/runtime/arm'),str(args.sources)]
    from hexagon_runtime.hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage
    from hexagon_live_compile_mailbox import atomic_json
    constant_layout=stage(catalog,args.workdir,include_code=False,
        **({'constant_load_split_proofs':args.constant_load_split_proofs} if args.constant_load_split_proofs else {})) or []
    loader_record=None
    if args.constant_load_split_proofs:
        if (len(constant_layout)!=3 or
                {row['original_sha256'] for row in constant_layout}!={row['sha256'] for row in receipt['constants']} or
                any(row['max_load_bytes']!=256*1024**2 for row in constant_layout)):
            raise ValueError('All three original constants and verified bounded LOAD proofs required')
        loader_record=dict(kind='bounded_readonly_LOAD_payload_preserving_v1',
            original_weights_code_and_symbol_values_preserved=True,
            loader_failure_fixed=False,libraries=constant_layout)
    for row in receipt['constants']:
        staged_sha=sha(args.workdir/Path(row['path']).name)
        if constant_layout:
            if not any(proof['original_sha256']==row['sha256'] and proof['candidate_sha256']==staged_sha
                    for proof in constant_layout):
                raise ValueError('Verified staged constant payload differs from frozen model receipt')
        elif staged_sha!=row['sha256']:
            raise ValueError('Frozen model constant library changed')
    args.output.mkdir(parents=True)
    libraries={}
    for name,path in [('baseline',args.baseline_library),('profile',library)]:
        target=args.workdir/(sha(path)+'.so');shutil.copy2(path,target)
        if sha(target)!=sha(path):raise ValueError('Model DSO staging changed')
        libraries[name]=target
    payload=packet.read_bytes();output_bytes=receipt['output_bytes']
    capacity_ok=(len(payload)<=receipt['input_bytes'] if dynamic else len(payload)==receipt['input_bytes'])
    if (not capacity_ok or len(payload)!=forward['invoke']['input_bytes']
            or output_bytes!=forward['invoke']['output_bytes']
            or payload[:16]!=b'HXDY\1\0\0\0'+struct.pack('<q',args.query)):
        raise ValueError('Frozen complete original query input format required')
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.workdir),ADSP_LIBRARY_PATH=str(args.workdir)
        +';'+str(args.workdir/'rfsa')+';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv=['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(args.workdir/'buffer_stream'),
        str(libraries['baseline']),str(len(payload)),str(output_bytes)]
    record=dict(status='diagnostic_model_memory_replay_running',source_sha256=sha(__file__),
        receipt_sha256=args.receipt_sha256,catalog_sha256=args.catalog_sha256,
        accepted_run_sha256=args.accepted_run_sha256,accepted_audit_sha256=args.accepted_audit_sha256,
        input_sha256=forward['input_sha256'],expected_output_sha256=forward['output_sha256'],
        baseline_library_sha256=baseline_sha,profile_library_sha256=receipt['library']['sha256'],
        model_object_unchanged=True,model_graph_changed=False,power_votes_changed=False,
        diagnostic_instrumentation=True,model_inference_acceptance=False,
        external_profile_symbols=receipt.get('external_profile_symbols'),
        rows=[],profiles=[],model_calls=0,diagnostic_query_calls=0)
    record['query']=args.query
    if receipt.get('integer_call_timeline') is True:
        import integer_call_timeline
        record['timeline_parser_sha256']=sha(integer_call_timeline.__file__)
    if loader_record is not None:
        record['constant_loader_layout']=loader_record
    if args.accepted_model_gate:
        record['accepted_baseline_model_gate']=dict(path=str(args.accepted_model_gate),sha256=args.accepted_model_gate_sha256)
    if receipt.get('inspect_fmaf'):
        record['fmaf_effects']=[]
        fmaf_payload=fmaf_inputs()
    def save():atomic_json(args.output/'result.json',record)
    save();previous=Path.cwd()
    try:
        os.chdir(args.workdir)
        with HexagonSharedPacketStream(argv,env,args.output,input_bytes=len(payload),
                output_bytes=output_bytes,library_path_prefix='/dev/shm/qwen3-', initial_library=str(libraries['baseline']),timeout=120) as stream:
            record['ready']=stream.start();save()
            for index,name in enumerate(('baseline','profile','profile','baseline')):
                record['pending']=dict(index=index,library=name);save()
                stream.select_library(str(libraries[name]))
                output,event=stream.invoke_packet(payload,output_bytes)
                digest=hashlib.sha256(output).hexdigest();record['model_calls']+=1
                if digest!=forward['output_sha256']:
                    (args.output/'failed-output.bin').write_bytes(output)
                    raise ValueError('Complete model output differs from accepted ARM-exact packet')
                if index==1:
                    kept=args.output/'complete-profile-output.bin'
                    kept.write_bytes(output)
                    if sha(kept)!=digest:
                        raise ValueError('Retained complete diagnostic output changed')
                    record['preserved_complete_profile_output']=dict(path=str(kept),sha256=digest,bytes=len(output))
                del output
                record['rows'].append(dict(index=index,library=name,output_sha256=digest,
                    all_logits_and_56_KV_byteexact=True,actual_invoke=event));save()
                if name=='profile':
                    timeline=receipt.get('integer_call_timeline') is True
                    if timeline:
                        from integer_call_timeline import MAGIC,PACKET_BYTES,parse_timeline
                        if receipt.get('attention_phase_timeline') is True:
                            from integer_call_timeline import ATTENTION_MAGIC as MAGIC,ATTENTION_PACKET_BYTES as PACKET_BYTES
                            from integer_call_timeline import parse_attention_timeline as parse_timeline
                    stats,query_event=stream.invoke_packet((MAGIC if timeline else b'QWMEM01\0')+bytes(8),
                        PACKET_BYTES if timeline else 128+4096*48)
                    record['diagnostic_query_calls']+=1
                    (args.output/f'profile-{index}.bin').write_bytes(stats)
                    profile=(parse_timeline(stats,args.query) if timeline else
                        parse_profile(stats,receipt.get('external_profile_kinds',(1,2,3)),
                            receipt.get('required_profile_kinds',(1,2,3))))
                    if not dynamic and '10' in profile['totals'] and profile['totals']['10']['calls']!=197:
                        raise ValueError('Diagnostic integer calls do not bind all 197 accepted operators')
                    if profile['invocations']!=len(record['profiles'])+1:
                        raise ValueError('Memory counters do not bind actual model invocations')
                    record['profiles'].append(dict(index=index,profile=profile,actual_invoke=query_event));save()
                    if receipt.get('inspect_fmaf'):
                        count=struct.unpack_from('<I',fmaf_payload,8)[0]
                        response,effects_event=stream.invoke_packet(fmaf_payload,256+count*4)
                        record['diagnostic_query_calls']+=1;save()
                        (args.output/f'fmaf-effects-{index}.bin').write_bytes(response)
                        effects=parse_fmaf_effects(response,fmaf_payload)
                        record['fmaf_effects'].append(dict(index=index,effects=effects,actual_invoke=effects_event));save()
        if stream.process_exit_code:raise ValueError('DSP owner failed')
        record.pop('pending',None)
        record.update(status='diagnostic_model_memory_complete_packet_bitexact',
            process_exit_code=stream.process_exit_code,stream_events=stream.events,
            DSP_calls=record['model_calls']+record['diagnostic_query_calls'])
        save();os.chdir(previous);shutil.rmtree(args.workdir)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc());save();raise
    finally:os.chdir(previous)
    print(record['status'])


if __name__=='__main__':main()
