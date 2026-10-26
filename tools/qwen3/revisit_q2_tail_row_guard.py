"""Recompute retained QK words, guard proofs, FP state, full FNV and owners."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
import json
from pathlib import Path
import shlex
import statistics
import struct
import subprocess
import tempfile
import time
from build import ROOT,HERE,entry,sha,write
from profile_model_memory import checked
from q2_tail_packet import CASES,parse_output
from revisit_model_integer_timeline import historical_owner
from revisit_release_index import indexed_files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','run','plan','controller','build-controller','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip();revision=git('rev-parse','HEAD')
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain'):
        raise ValueError('Fresh frozen independent reviewer required')
    plan=read(a.plan);process=read(a.run/'process.json');board=a.run/'board';run=read(board/'result.json')
    files={r['name']:r for r in plan['files']}
    for row in plan['files']:
        if sha(row['source'])!=row['sha256']:raise ValueError('Staged source or artifact changed')
    receipt_path=Path(files['result.json']['source']);receipt=read(receipt_path)
    fixture=read(checked(receipt['fixture']));all28=read(checked(receipt['all28']));sdk=read(checked(receipt['SDK']))
    refs=[]
    for proof in (receipt,fixture,all28,sdk):indexed_files(proof,refs)
    if (receipt['status']!='actual_Q2_row_major_guard_same_operators_linked_NOT_target_or_model'
            or receipt['isolated_operator_compilations']!=0 or receipt.get('original_operator_objects_unchanged')is not True or receipt['new_model_PRIMARY_compilations']
            or receipt['options']['addFastMath']!='False'
            or receipt['original_compiler']['sha256']!='4043395bddd404bcb8370f98fe068a3ddf2a481f10e1e11847a355d965f034ce'
            or fixture['positive']!=2 or fixture['unsafe_rejected']!=10
            or all28['all28_QK_regions']!=28 or all28['fixture']!=receipt['fixture']
            or sdk['base_replay_byte_identical']is not True
            or any(proof['DSP_calls']for proof in (receipt,fixture,all28,sdk))):
        raise ValueError('Strict current compiler, original FMA proof, all28 and operator-only build required')
    controls=[]
    for controller,proof,script,output,digest in (
        (a.controller,process,'run_staged_probe.py',a.run,process['source_sha256']),
        (a.build_controller,receipt,'build_q2_tail_row_guard.py',receipt_path.parent,receipt['source']['sha256']),
        (Path(receipt['fixture_controller']['path']),fixture,'inspect_q2_tail_fixture.py',Path(receipt['fixture']['path']).parent,fixture['source']['sha256']),
        (Path(receipt['all28_controller']['path']),all28,'inspect_q2_tail_model.py',Path(receipt['all28']['path']).parent,all28['source']['sha256']),
        (Path(fixture['SDK_controller']['path']),sdk,'build_q2_tail_preflight_sdk.py',Path(fixture['SDK']['path']).parent,sdk['source']['sha256'])):
        historical_owner(controller,script,output,digest,proof['source_revision']);controls.append(entry(controller))
    for path,digest in sdk['original_link_objects'].items():
        if sha(path)!=digest:raise ValueError('An original compiler object changed')
    for key,name in [('dispatch','buffer_reuse/DynamicQKRegisterDispatch.cpp'),
            ('identical_tail','buffer_reuse/DynamicQKIdenticalTail.inc'),('integer_guard','runtime/profile/q2_qk_positive_zero_extent.c')]:
        if checked(sdk[key]).read_bytes()!=(HERE/name).read_bytes():raise ValueError('Tested transform/guard differs from independent source')
    for name in ('q2_tail_probe.c','q2_qk_positive_zero_extent.c','zero_tail_hvx_guard.c'):
        ref=next(row for row in receipt['sources']if Path(row['path']).name==name)
        if checked(ref).read_bytes()!=(compiler_tools()/'runtime/profile'/name).read_bytes():
            raise ValueError('Actual floating-state adapter or integer guard differs from independent source')
    original=read(checked(receipt['original_build']))
    original_review=read(checked(receipt['original_target_review']))
    historical_owner(checked(receipt['original_controller']),'build_q2_tail_physical_probe.py',Path(receipt['original_build']['path']).parent,original['source']['sha256'],original['source_revision'])
    historical_owner(checked(receipt['original_target_review_controller']),'revisit_q2_tail_probe.py',Path(receipt['original_target_review']['path']),original_review['source']['sha256'],original_review['source_revision'])
    if (original_review['receipt']!=receipt['original_build']or original_review['DSP_calls']!=256
            or checked(receipt['integer_guard_variant']).read_bytes()!=(compiler_tools()/'runtime/profile/q2_qk_positive_zero_extent_row_major.c').read_bytes()):
        raise ValueError('Explicit new integer guard and exact original target operator proof required')
    checked(receipt['integer_guard_object'])
    commands=receipt['commands']
    if len(commands)!=5 or any(r['returncode']for r in commands):raise ValueError('Exact original2 replays, guard compile and2 relinks required')
    old_commands=original['commands'][-2:]
    for i,kind in enumerate(('baseline','tail')):
        if receipt['variants'][kind]['object']!=original['variants'][kind]['object']:
            raise ValueError('Both exact actually target-tested operators must remain unchanged')
        replay=old_commands[i]['argv'][:]
        replay[replay.index('-o')+1]=str(receipt_path.parent/(kind+'-original-replay.so'))
        if replay!=commands[i]['argv']or sha(replay[replay.index('-o')+1])!=original['variants'][kind]['library']['sha256']:
            raise ValueError('Both old complete library byte replays required')
        expected=old_commands[i]['argv'][:]
        old_guard=str(Path(receipt['original_build']['path']).parent/'q2_qk_positive_zero_extent.c.o')
        expected[expected.index(old_guard)]=receipt['integer_guard_object']['path']
        expected[expected.index('-o')+1]=receipt['variants'][kind]['library']['path']
        if expected!=commands[i+3]['argv']:raise ValueError('Only new integer guard object may replace original link input')
    base=read(checked(sdk['base_sdk']));old_review=read(checked(base['physical_review']));old=read(checked(old_review['receipt']))
    if (receipt['variants']['baseline']['object']!=old['variants']['dispatch']['object']
            or receipt['current_baseline_library']!=old['variants']['dispatch']['library']
            or receipt['variants']['tail']['actual_fixture']!=fixture['cases'][0]['output']):
        raise ValueError('The exact actual current baseline object and checked tail input are required')
    if (process['status']!='complete' or process['returncode'] or process['output_copy_returncode']
            or process['actual_DSP_calls']!=256 or process['plan']!=entry(a.plan)
            or process['result']!=entry(board/'result.json') or Path('/proc',str(process['ssh_pid'])).exists()
            or run['status']!='actual_Q2_identical_tail_ABBA256_all_target_bits_four_rounding_FP_flags_complete_NOT_model'
            or run['DSP_calls']!=256 or run['actual_DSP_calls']!=256 or run['process_exit_code']
            or run['source_sha256']!=files['probe_q2_tail_arm.py']['sha256']
            or run['packet_source_sha256']!=files['q2_tail_packet.py']['sha256']
            or run['packet_source_sha256']!=sha(HERE/'q2_tail_packet.py')
            or run['receipt_sha256']!=sha(receipt_path) or run['deadline_unix']!=plan['deadline_unix']
            or run['model_inference_acceptance'] or run['performance_acceptance'] or run['power_votes_changed']):
        raise ValueError('Complete normally closed source-bound256 RPCs required')
    for kind,row in receipt['variants'].items():
        if files[kind+'.so']['sha256']!=row['library']['sha256']:raise ValueError('Actual operator library changed')
    raw=board/'stream.stdout.txt';events=[json.loads(line)for line in raw.read_text().splitlines()]
    invokes=[e for e in events if e['stage']=='invoke'];ready=[e for e in events if e['stage']=='ready']
    if (events!=run['stream_events'] or events[-1]['stage']!='close'
            or invokes!=[r['actual_invoke']for r in run['rows']]
            or [e['ordinal']for e in invokes]!=list(range(1,257))
            or any(e.get(k,0)for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))
            or len(ready)!=1 or ready[0]!=run['ready']):
        raise ValueError('Every unique raw RPC, actual library and normal close must match')
    source=compiler_tools()/'runtime/profile/dynamic_qk_reference.c'
    if sha(source)!=run['ARM_reference']['source_sha256'] or sha(source)!=files[source.name]['sha256']:
        raise ValueError('Independent ARM recurrence source changed')
    import numpy as np
    names=[f'q{q}-n{n}-b{flag}-{pattern}-r{rounding}'for q,n,flag,pattern,extent,rounding in CASES]
    if run['cases']!=names or len(run['rows'])!=256:raise ValueError('All64 cases and exact ABBA256 required')
    cases=[];gains=[];arm_cases=0
    with tempfile.TemporaryDirectory(prefix='qwen-q2-tail-independent-')as temporary:
        library=Path(temporary)/'reference.so'
        subprocess.run(['/usr/bin/gcc','-O2','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',
            '-fno-builtin-fmaf',str(source),'-lm','-o',str(library)],check=True,capture_output=True)
        reference=ctypes.CDLL(str(library));ptr=ctypes.POINTER(ctypes.c_float)
        fn=reference.qwen_dynamic_qk_reference;fn.argtypes=[ptr]*4+[ctypes.c_uint]*3;fn.restype=ctypes.c_int
        fnv=reference.qwen_dynamic_qk_fnv;fnv.argtypes=[ctypes.c_void_p,ctypes.c_uint];fnv.restype=ctypes.c_uint32
        for index,(q,n,flag,pattern,extent,rounding)in enumerate(CASES):
            name=names[index];rows=1 if flag else q;group=run['rows'][4*index:4*index+4]
            if [(r['case'],r['query'],r['columns'],r['broadcast'],r['pattern'],r['extent'],r['rounding'],r['order'],r['kind'])for r in group]!=[
                    (name,q,n,flag,pattern,extent,rounding,i,kind)for i,kind in enumerate(('baseline','tail','tail','baseline'))]:
                raise ValueError('Exact original cases and ABBA order required')
            path=board/group[0]['input_file'];payload=path.read_bytes()
            if (payload[:8]!=b'Q2TAIL01' or payload[8:24]!=struct.pack('<IIII',q,n,flag,rounding)
                    or any(payload[24:128]) or len(payload)!=128+4*(16*128*n+16*rows*128+16*q*n)):
                raise ValueError('Complete original typed operator input required')
            keys=np.frombuffer(payload,dtype='<f4',count=16*128*n,offset=128).reshape(16,128,n)
            query=np.frombuffer(payload,dtype='<f4',count=16*rows*128,offset=128+4*16*128*n)
            initial=np.frombuffer(payload,dtype='<f4',count=16*q*n,offset=128+4*(16*128*n+16*rows*128)).reshape(16,q,n)
            proven=1024
            if n==1024 and not flag:
                nonzero=np.any(keys.view(np.uint32)!=0,axis=(0,1))|np.any(initial.view(np.uint32)!=0,axis=(0,1))
                used=np.flatnonzero(nonzero);prefix=0 if not used.size else (int(used[-1])//128+1)*128
                proven=1024 if prefix>=768 else prefix
            if proven!=extent:raise ValueError('Independent complete bit-pattern guard disagrees')
            arm_required=rounding==0 and all(np.isfinite(x).all()for x in (keys,query,initial))
            expected=None;arm_ref=None
            if arm_required:
                expected=np.empty_like(initial)
                if fn(*(x.ctypes.data_as(ptr)for x in (keys,query,initial,expected)),q,n,flag):
                    raise ValueError('Independent host ordered recurrence failed')
                arm_ref=board/group[0]['ARM_reference_file']
                if arm_ref.read_bytes()!=expected.tobytes():raise ValueError('Retained ARM and rebuilt host reference differ')
                arm_cases+=1
            scalar=int(fnv(ctypes.c_char_p(payload),len(payload)));baseline=None;baseline_flags=None
            for row in group:
                output=board/row['output_file'];data=output.read_bytes();e=row['actual_invoke']
                parsed=parse_output(data,q,n,flag,rounding,extent,row['kind'])
                if row['order']==0:baseline=data[128:];baseline_flags=parsed['FP_flags']
                if (data[128:]!=baseline or parsed['FP_flags']!=baseline_flags or parsed!=row['parsed']
                        or row['input_file']!=path.name or row['input_sha256']!=sha(path) or row['output_sha256']!=sha(output)
                        or (expected is not None and data[128:]!=expected.tobytes())
                        or row['ARM_reference_checked']!=arm_required
                        or (arm_required and (row['ARM_reference_file']!=arm_ref.name or row['ARM_reference_sha256']!=sha(arm_ref)))
                        or (not arm_required and (row['ARM_reference_file']is not None or row['ARM_reference_sha256']is not None))
                        or row['all_target_FP32_bits_and_FP_flags_equal']is not True
                        or row['sequential_scalar_input_FNV']!=scalar or e['kernel_us']!=parsed['kernel_us']
                        or e['input_bytes']!=len(payload) or e['output_bytes']!=len(data)
                        or e['library_slot']!=(0 if row['kind']=='baseline'else 1)
                        or any(e[k]!=scalar for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'))):
                    raise ValueError('Every retained target word, FP flag, guard count, full hash and FNV must match independently')
            measure=None
            if rounding==0 and group[1]['parsed']['eligibility']:
                average=lambda kind:statistics.mean(r['parsed']['kernel_us']for r in group if r['kind']==kind)
                old,new=average('baseline'),average('tail');measure=dict(case=name,baseline_us=old,tail_us=new,ratio=new/old);gains.append(measure)
            cases.append(dict(case=name,input=entry(path),scalar_FNV=scalar,guard_extent=proven,rounding=rounding,
                ARM_reference=entry(arm_ref)if arm_ref else None,outputs=[entry(board/r['output_file'])for r in group],measurement=measure))
    pid=ready[0]['ARM_pid'];board_config=read(a.config)['board']
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board_config['ssh_target'],
        shlex.join([board_config['python'],'-c','from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()',str(pid)])],check=True)
    if git('rev-parse','HEAD')!=revision or git('status','--porcelain'):raise ValueError('Independent reviewer source changed')
    positive=bool(gains)and statistics.geometric_mean(r['ratio']for r in gains)<0.97
    write(a.output,dict(status='actual_Q2_row_major_guard_ABBA256_full_bits_four_rounding_FP_flags_and_extents_independently_revisited_NOT_model',
        source_root=str(ROOT),source_revision=revision,source=entry(__file__),DSP_calls=256,new_DSP_calls=0,
        new_model_PRIMARY_compilations=0,model_inference_acceptance=False,performance_acceptance=False,
        primitive_positive_gain=positive,primitive_measurements=gains,nearest_finite_ARM_and_host_reference_cases=arm_cases,
        four_rounding_modes_original_target_bits_and_FP_flags_equal=True,
        all_runtime_guard_counts_and_independent_positive_zero_extents_equal=True,
        receipt=entry(receipt_path),run=entry(board/'result.json'),plan=entry(a.plan),controls=controls,
        raw=entry(raw),cases=cases,actual_AP_owner_gone=pid,independent_reference_source=entry(source),
        host_reference_compiler=entry('/usr/bin/gcc'),completed_unix=time.time()))


if __name__=='__main__':main()
