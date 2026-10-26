"""Independently re-read all eight complete original/candidate model packets."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import tempfile
import time
from build import ROOT, HERE, entry, sha, write
from native_model_acceptance import require_model_acceptance, accepted_cold
from profile_model_memory import checked
from revisit_q2_dispatch_preflight import actual_owner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','build','build-controller','primitive','primitive-controller','plan','run','controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    revision=git('rev-parse','HEAD')
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain'):
        raise ValueError('Fresh committed independent full-packet review required')
    build=read(a.build/'result.json');primitive=read(a.primitive);process=read(a.run/'process.json')
    actual_owner(a.build_controller,build,'build_global_hvx_copy.py',a.build)
    actual_owner(a.primitive_controller,primitive,'revisit_global_copy_physical.py',a.primitive)
    actual_owner(a.controller,dict(source_root=str(ROOT),source_revision=process['source_revision'],
        source=dict(path=str(HERE/'run_staged_probe.py'),sha256=process['source_sha256'])),'run_staged_probe.py',a.run)
    if (process['status']!='complete' or process['returncode'] or process['output_copy_returncode']
            or process['actual_DSP_calls']!=8 or process['plan']!=entry(a.plan)
            or primitive['status']!='actual_DSP_model_local_global_copy_498_cases_ABBA4_complete_bytes_canaries_3FNV_and_normal_owner_independently_revisited_NOT_model_acceptance'
            or primitive['positive_primitive_gain'] is not True or primitive['DSP_primitive_calls']!=4
            or primitive['DSP_model_calls'] or primitive['model_inference_acceptance']
            or primitive['build']!=entry(a.build/'result.json')):
        raise ValueError('Own complete actual operator4 before normally owned model8 required')
    index_path=checked(build['accepted_index']);model,closure=require_model_acceptance(read(index_path))
    cold=accepted_cold(model,closure);acceptance=read(cold/'acceptance.json')
    audit_path=checked(acceptance['audit']);audit=read(audit_path)
    native_path=audit_path.parent/audit_path.name.removesuffix('-independent-audit.json')/'result.json'
    native=read(native_path)
    if (audit['status']!='framework_native_numerical_acceptance_passed' or audit['numerical_acceptance'] is not True
            or audit['run_sha256']!=sha(native_path) or native['process_exit_code'] or native['CPU_model_fallback']):
        raise ValueError('Original complete same-input ARM W8A8 evidence required')
    plan=read(a.plan)
    for row in plan['files']:
        if sha(row['source'])!=row['sha256']:raise ValueError('Actually staged bytes changed')
    board=a.run/'board';result=read(board/'result.json')
    if (result['status']!='model_local_global_copy_original_q1_q8_ABBA8_all_full_packets_exact_NOT_generation_acceptance'
            or result['candidate_kind']!='model-local-global-hvx-copy' or result['DSP_calls']!=8
            or result['process_exit_code'] or result['new_model_PRIMARY_compilations']
            or result['model_inference_acceptance'] or result['performance_acceptance']
            or result['build_sha256']!=sha(a.build/'result.json')
            or result['primitive_review_sha256']!=sha(a.primitive)
            or result['accepted_run_sha256']!=sha(native_path) or result['accepted_audit_sha256']!=sha(audit_path)
            or result['source_sha256']!=sha(HERE/'probe_caller_copy_model_arm.py')
            or len(result['rows'])!=8 or len(result['cases'])!=2):
        raise ValueError('Exact own q1/q8 full model packets and honest diagnostic scope required')
    timings={};proofs=[];pids=[]
    with tempfile.TemporaryDirectory(prefix='qwen-global-model-fnv-')as temporary:
        source=compiler_tools()/'runtime/profile/physical_gqa_pv_reference.c';library=Path(temporary)/'reference.so'
        subprocess.run(['cc','-O2','-shared','-fPIC',str(source),'-lm','-o',str(library)],check=True,capture_output=True)
        fnv=ctypes.CDLL(str(library)).qwen_physical_scalar_fnv32
        fnv.argtypes=[ctypes.c_void_p,ctypes.c_uint];fnv.restype=ctypes.c_uint32
        for case,query in enumerate((1,8)):
            forward=next(row for row in native['forwards']if row['query']==query)
            original=native_path.parent/f"forward-{forward['step']:02d}-q{query}"
            packet=board/f'q{query}/input.bin';payload=packet.read_bytes()
            expected_path=original/'output.bin';expected=expected_path.read_bytes()
            if (sha(packet)!=forward['input_sha256'] or payload!=(original/'input.bin').read_bytes()
                    or sha(expected_path)!=forward['output_sha256'] or len(expected)!=235488768):
                raise ValueError('All actual input/output bytes must preserve the original ARM reference')
            scalar=int(fnv(ctypes.c_char_p(payload),len(payload)))
            case_record=result['cases'][case];raw=board/f'q{query}/stream.stdout.txt'
            events=[json.loads(line)for line in raw.read_text().splitlines()];invokes=[e for e in events if e['stage']=='invoke']
            rows=result['rows'][case*4:case*4+4];built=next(m for m in build['models']if m['query_bounds'][0]==(1 if query==1 else 2))
            if (case_record['query']!=query or case_record['process_exit_code'] or case_record['stream_events']!=events
                    or len(invokes)!=4 or events[-1]['stage']!='close'
                    or any(e.get(k,0)for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))
                    or built['baseline']['sha256']!=forward['DSO_sha256']
                    or built['all_copy_symbols_locally_bound'] is not True):
                raise ValueError('Each full model stream and actual accepted baseline must be exact')
            for name in ('primary','baseline','candidate','accepted_HVX_runtime','accepted_HVX_source'):
                checked(built[name])
            for ref in built['original_link_inputs']+built['constants']:checked(ref)
            outputs=[]
            for i,(row,event)in enumerate(zip(rows,invokes)):
                output=board/row['output_file'];role=('baseline','candidate','candidate','baseline')[i]
                if (row['query']!=query or row['index']!=i or row['role']!=role or row['actual_invoke']!=event
                        or event['ordinal']!=i+1 or event['library_slot']!=(0,1,1,0)[i]
                        or event['input_bytes']!=len(payload) or event['output_bytes']!=len(expected)
                        or row['all_logits_and_56_KV_words_bitexact'] is not True
                        or row['full_output_sha256']!=forward['output_sha256']
                        or any(event[k]!=scalar for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'))
                        or sha(output)!=forward['output_sha256'] or output.read_bytes()!=expected):
                    raise ValueError('Every logits/KV bit, full byte packet, role and independent three-FNV identity must match')
                outputs.append(entry(output))
            average=lambda role:statistics.mean(row['actual_invoke']['kernel_us']for row in rows if row['role']==role)
            baseline,candidate=average('baseline'),average('candidate')
            timings[str(query)]=dict(baseline=baseline,candidate=candidate,ratio=candidate/baseline)
            ready=next(event for event in events if event['stage']=='ready');pids.append(ready['ARM_pid'])
            proofs.append(dict(query=query,input=entry(packet),original_input=entry(original/'input.bin'),
                original_output=entry(expected_path),outputs=outputs,raw=entry(raw),independent_full_input_FNV=scalar,
                original_model=built['baseline'],candidate_model=built['candidate']))
    config=read(a.config)
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],
        shlex.join([config['board']['python'],'-c','from pathlib import Path;import sys;assert all(not Path("/proc",p).exists()for p in sys.argv[1:])',*map(str,pids)])],check=True)
    if git('rev-parse','HEAD')!=revision or git('status','--porcelain'):raise ValueError('Reviewer source changed')
    write(a.output,dict(status='model_local_global_HVX_copy_original_ARM_q1_q8_ABBA8_all_logits_56KV_complete_packets_3FNV_and_owners_independently_revisited_NOT_generation_acceptance',
        source_root=str(ROOT),source_revision=revision,source=entry(__file__),build=entry(a.build/'result.json'),
        build_controller=entry(a.build_controller),primitive=entry(a.primitive),primitive_controller=entry(a.primitive_controller),
        run=entry(a.run/'process.json'),controller=entry(a.controller),proofs=proofs,kernel_microseconds=timings,
        positive_model_gain=timings['1']['ratio']<.98 and timings['8']['ratio']<=1.02,
        independent_FNV_source=entry(source),DSP_calls=8,new_DSP_calls=0,new_model_PRIMARY_compilations=0,
        model_inference_acceptance=False,performance_acceptance=False,shared_default_changed=False,completed_unix=time.time()))


if __name__=='__main__':main()
