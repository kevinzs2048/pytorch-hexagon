"""Independently revisit four complete startup arms and a recorded interruption."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from build import ROOT, entry, sha, write
from resume_native_compile_session_abba import require_completed_arm
from run_native_compile_session_abba import summarize_abba
from revisit_native_startup import startup_summary


def require_normal_owner(path, deadline):
    owner=json.loads(Path(path).read_text())
    if (owner['status']!='completed' or type(owner['returncode']) is not int or owner['returncode']!=0
            or owner['authorized_deadline_unix']!=deadline or owner['completed_unix']>=deadline
            or any(Path('/proc',str(owner[key])).exists() for key in ('controller_pid','child_pid'))):
        raise ValueError('Actual normal completed owner within original deadline required')
    return owner


def require_exact_parameter_guards(run, oracle):
    value=run['parameter_validation'];old=oracle['parameter_validation']
    constants=value['guarded_python_float_constants']
    if (value['mode']!='exact-bytes' or value['all_logical_parameter_bytes_checked_each_forward'] is not True
            or value['version_only_check'] is not False or value['initial_compiler_SHA_pins_preserved'] is not True
            or type(value['guarded_python_float_forwards_checked']) is not int
            or value['guarded_python_float_forwards_checked']!=16
            or type(constants['count']) is not int or constants['count']!=142
            or constants['FP64_bytes_per_constant']!=8
            or constants['owner_identity_and_all_bytes_checked_each_forward'] is not True
            or constants!=old['guarded_python_float_constants']):
        raise ValueError('All original parameter bytes and all 142 FP64 guards on every forward required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','oracle','closure','controller','numerical-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();read=lambda p:json.loads(Path(p).read_text())
    git=lambda root,*words:subprocess.check_output(['git','-C',str(root),*words],text=True).strip()
    if not args.output.is_absolute() or args.output.exists() or git(ROOT,'status','--porcelain'):
        raise ValueError('Fresh independent result and frozen clean reviewer required')
    closure=read(args.closure);deadline=closure['original_deadline_preserved']
    if (closure['status']!='original_host_worker_vs_preloaded_session_ABBA64_completed_after_recorded_server_restart_DIAGNOSTIC'
            or type(closure['DSP_calls']) is not int or closure['DSP_calls']!=64
            or closure['model_inference_acceptance'] is not False or closure['performance_acceptance'] is not False):
        raise ValueError('Complete diagnostic64 with honest interrupted-supervisor history required')
    require_normal_owner(args.controller,deadline)
    interruption=closure['interruption']
    if sha(interruption['path'])!=interruption['sha256']:
        raise ValueError('Recorded interrupted arm changed')
    interrupted=read(interruption['path'])
    if (interrupted['excluded_interrupted_DSP_calls']!=closure['excluded_interrupted_DSP_calls']
            or read(interrupted['original_root_controller']['path'])['status']!='failed'
            or read(interrupted['interrupted_arm_controller']['path'])['status']!='failed'):
        raise ValueError('Interrupted supervisor must remain failed and its partial DSP calls excluded')
    sys.path[:0]=[str(args.numerical_source/'tools/qwen3'),str(args.numerical_source/'qwen3/runtime/arm'),str(args.numerical_source)]
    from compare_performance import accepted,compare
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    from hexagon_host_timing import audit_arm_forward_timings
    from bounded_loader_policy import verify_native_layout
    acceptance,oracle,_=accepted(args.oracle)
    if (git(args.numerical_source,'rev-parse','HEAD')!=acceptance['numerical_revision']
            or git(args.numerical_source,'status','--porcelain')):
        raise ValueError('Original complete numerical oracle and frozen compiler source required')
    config=read(args.config);raw=[];ap=[];arms=[];runs=[]
    for row in closure['rows']:
        proof=require_completed_arm(row,deadline);folder=Path(row['result']['path']).parent
        owner=require_normal_owner(row['controller']['path'],deadline)
        relative='tools/qwen3/probe_native_compile_session.py'
        producer=subprocess.check_output(['git','-C',owner['source_root'],'show',owner['source_revision']+':'+relative])
        if hashlib.sha256(producer).hexdigest()!=proof['source_sha256']:
            raise ValueError('Actual historical producer executable source changed')
        if sha(proof['worker_source']['path'])!=proof['worker_source']['sha256']:
            raise ValueError('Original or session worker source changed')
        run=read(folder/'native-result.json')
        if (sha(folder/'native-result.json')!=proof['native_sha256']
                or run['process_exit_code']!=0 or run['CPU_model_fallback'] is not False
                or run['native_generate'] is not True or run['source_sha256']!=proof['original_board_source_sha256']
                or type(run['DSP_calls']) is not int or run['DSP_calls']!=16
                or run['frontend_invocations']!=0 or run['machine_code_compilations']!=0
                or read(folder/'run-exit.json')!={'board_returncode':0,'worker_returncode':0}
                or any(Path('/proc',str(pid)).exists() for pid in proof['active_processes'].values())):
            raise ValueError('Own unchanged native16, all process exits and zero compiler work required')
        compare(oracle,run);audit_arm_forward_timings(run);require_exact_parameter_guards(run,oracle)
        if verify_native_layout(oracle)!=verify_native_layout(run):
            raise ValueError('Actual original constant loader payload changed')
        if [r['terminal_observation'] for r in oracle['requests']]!=[r['terminal_observation'] for r in run['requests']]:
            raise ValueError('Original EOS/stopping behavior changed')
        chain=audit_compile_handoff(run,Path(proof['host_mailbox']),rebuilt=True)
        if chain!=proof['complete_compile_chain_rehashed'] or startup_summary(run)!=proof['startup']:
            raise ValueError('Actual full compiler chain or disjoint startup timings changed')
        trace=Path(config['observations'])/Path(proof['native_output']).name/'stream.stdout.txt'
        events=[json.loads(line) for line in trace.read_text().splitlines()]
        if (not events or events[-1]['stage']!='close'
                or [e for e in events if e.get('stage')=='invoke']!=[f['invoke'] for f in run['forwards']]
                or any(len({e[k] for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')})!=1
                    for e in events if e.get('stage')=='invoke')
                or any(e.get(k,0) for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
            raise ValueError('All actual raw AP/RPC/DSP invocations and normal close required')
        ready=[e for e in events if e.get('stage')=='ready']
        if len(ready)!=1 or type(ready[0].get('ARM_pid')) is not int:
            raise ValueError('Actual AP owner PID required')
        ap.append(ready[0]['ARM_pid']);raw.append(entry(trace));runs.append(proof)
        arms.append(dict(result=row['result'],controller=row['controller'],native=entry(folder/'native-result.json'),
                         complete_compiler_chain=chain,worker_mode=row['worker_mode']))
    measurements=summarize_abba(runs)
    if measurements!=closure['measurements']:
        raise ValueError('Actual four-arm startup statistics changed')
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],shlex.join([
        config['board']['python'],'-c','from pathlib import Path;import sys;assert all(not Path("/proc",p).exists() for p in sys.argv[1:])',*map(str,ap)])],check=True)
    write(args.output,dict(status='host_session_ABBA64_all_original_packets_raw_FNV_compiler_chains_and_normal_arm_owners_independently_revisited_DIAGNOSTIC',
        closure=entry(args.closure),controller=entry(args.controller),oracle=entry(args.oracle/'acceptance.json'),
        arms=arms,raw_invocation_traces=raw,actual_AP_owners_gone=ap,measurements=measurements,
        DSP_calls=64,new_DSP_calls=0,excluded_interrupted_DSP_calls=closure['excluded_interrupted_DSP_calls'],
        original_model_DSOs_parameter_checks_142_FP64_guards_and_loader_unchanged=True,
        model_inference_acceptance=False,performance_acceptance=False,production_configuration_changed=False,
        TUI_startup_claimed=False,reviewer_source=entry(__file__),reviewer_revision=git(ROOT,'rev-parse','HEAD'),completed_unix=time.time()))


if __name__=='__main__':main()
