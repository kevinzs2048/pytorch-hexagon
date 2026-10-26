"""Recheck startup observations against an accepted oracle without new DSP.

Sequential capture callbacks are disjoint from compiled forward callbacks.
Request-relative backend offsets and nested framework spans are not additive.
The residual includes Dynamo, generation control and AOT output writeback;
it is not labelled pure Dynamo time or a new performance acceptance.
"""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from build import ROOT, entry, sha, write

BOOTSTRAP=('import_and_configure','load_and_verify_native_model',
    'prepare_cache_and_decompositions','stage_runtime_and_constants')
CAPTURE=('AOT_prepare_graph','bind_immutable_inputs','identify_live_graph',
    'publish_live_capture','wait_for_compiler_host','verify_and_stage_reply','record_and_bind_callback')


def duration(record,schema,stages):
    if (record.get('schema')!=schema or record.get('clock')!='perf_counter_ns'
            or set(record.get('stages_ns',{}))!=set(stages)
            or any(type(v) is not int or v<0 for v in record['stages_ns'].values())
            or type(record.get('total_ns')) is not int
            or record['total_ns']!=sum(record['stages_ns'].values())):
        raise ValueError('Complete nonnegative sequential timing required')
    return record['total_ns']


def startup_summary(run):
    startup=run['startup_timing'];captures=startup['captures'];forwards=run['forwards']
    if (startup.get('clock')!='perf_counter_ns' or startup.get('nested_framework_spans_not_added') is not True
            or startup.get('scope')!='ARM_bootstrap_and_actual_capture_callback'
            or len(captures)!=len(run['captures'])
            or [c['capture_index'] for c in captures]!=list(range(len(captures)))):
        raise ValueError('Complete original startup capture ledger required')
    bootstrap=duration(startup['bootstrap'],'ARM_native_bootstrap_stage_timing_v1',BOOTSTRAP)
    summaries=[];next_forward=0
    for request in run['requests']:
        timing=request['generation_timing'];start=request['first_forward'];end=request['end_forward']
        if type(start) is not int or type(end) is not int or start!=next_forward or not start<end<=len(forwards):
            raise ValueError('Every actual forward must bind exactly one complete request')
        next_forward=end;own=forwards[start:end];count=len(request['generated_ids']);ready=timing['generated_token_ready_ns']
        if (not own or any(f['request_index']!=request['index'] for f in own)
                or any(type(f['arm_timing']['total_ns']) is not int or f['arm_timing']['total_ns']<0 for f in own)
                or timing.get('schema')!='native_generate_streamer_timing_v1'
                or timing.get('clock')!='perf_counter_ns' or timing.get('token_ids_verified') is not True
                or timing.get('includes_capture_compiler_handoff_and_AOT_writeback') is not True
                or len(ready)!=count or timing['generated_tokens']!=count or not ready
                or any(type(v) is not int or v<0 for v in ready) or ready!=sorted(ready)
                or timing['first_generated_token_ready_ns']!=ready[0]
                or timing['inter_token_ns']!=[b-a for a,b in zip(ready,ready[1:])]
                or any(type(timing.get(k)) is not int for k in ('full_generate_ns','native_stream_end_ns'))
                or not ready[-1]<=timing['native_stream_end_ns']<=timing['full_generate_ns']):
            raise ValueError('Complete actual selected-token timing required')
        callbacks=sum(f['arm_timing']['total_ns'] for f in own)
        stages=dict.fromkeys(CAPTURE,0);spans=[]
        for capture in captures:
            if capture['request_index']!=request['index']:continue
            elapsed=duration(capture['sequential_callback'],'ARM_native_capture_stage_timing_v1',CAPTURE)
            offset=capture['request_start_to_backend_enter_ns'];prior=capture['prior_forwards_in_this_request']
            if (type(offset) is not int or offset<0 or type(prior) is not int or not 0<=prior<len(own)
                    or own[prior]['AOT_capture_index']!=capture['capture_index']
                    or offset+elapsed>timing['full_generate_ns']
                    or (spans and offset<spans[-1]['end_ns'])):
                raise ValueError('Actual disjoint request-relative capture intervals required')
            spans.append(dict(capture_index=capture['capture_index'],start_ns=offset,end_ns=offset+elapsed,
                prior_forwards_in_this_request=prior))
            for stage,value in capture['sequential_callback']['stages_ns'].items():stages[stage]+=value
        capture_total=sum(stages.values());residual=timing['full_generate_ns']-callbacks-capture_total
        if residual<0:raise ValueError('Sequential callback spans exceed actual generation time')
        summaries.append(dict(request_index=request['index'],generated_tokens=count,
            full_generate_seconds=timing['full_generate_ns']/1e9,first_token_seconds=ready[0]/1e9,
            median_inter_token_seconds=statistics.median(timing['inter_token_ns'])/1e9 if count>1 else None,
            ARM_forward_callback_seconds=callbacks/1e9,capture_callback_seconds=capture_total/1e9,
            capture_stages_seconds={k:v/1e9 for k,v in stages.items()},capture_intervals=spans,
            generation_outside_measured_callbacks_seconds=residual/1e9,
            residual_includes_Dynamo_control_and_AOT_writeback=True,
            request_start_offsets_added=False,nested_framework_spans_added=False))
    if next_forward!=len(forwards) or sum(len(r['capture_intervals']) for r in summaries)!=len(captures):
        raise ValueError('Every actual capture must bind exactly one completed request')
    return dict(bootstrap_seconds=bootstrap/1e9,
        bootstrap_stages_seconds={k:v/1e9 for k,v in startup['bootstrap']['stages_ns'].items()},
        requests=summaries,performance_acceptance=False,TUI_startup_claimed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','oracle','native','oracle-controller','native-controller','numerical-source','benchmark-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    git=lambda root,*x:subprocess.check_output(['git','-C',str(root),*x],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git(ROOT,'status','--porcelain'):
        raise ValueError('Fresh absolute independent result and committed reviewer required')
    sys.path[:0]=[str(a.numerical_source/'tools/qwen3'),str(a.numerical_source/'qwen3/runtime/arm'),str(a.numerical_source)]
    from compare_performance import accepted,compare
    from hexagon_host_timing import audit_arm_forward_timings
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    from bounded_loader_policy import verify_native_layout
    acceptance,oracle,_=accepted(a.oracle);proof=read(a.native/'result.json');run=read(a.native/'native-result.json')
    for source,revision in ((a.numerical_source,acceptance['numerical_revision']),
            (a.benchmark_source,read(a.native_controller)['source_revision'])):
        if git(source,'rev-parse','HEAD')!=revision or git(source,'status','--porcelain'):
            raise ValueError('Both actual numerical and diagnostic sources must remain frozen')
    if (proof['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
            or proof['coordinator_source_sha256']!=sha(a.benchmark_source/'tools/qwen3/benchmark_accepted_generation.py')
            or proof['oracle_acceptance_sha256']!=sha(a.oracle/'acceptance.json')
            or proof['oracle_native_sha256']!=sha(a.oracle/'native-result.json')
            or proof['native_sha256']!=sha(a.native/'native-result.json')
            or proof['numerical_source_revision']!=acceptance['numerical_revision']
            or proof['inline_CPU_observer'] is not False or proof['new_CPU_reference_execution'] is not False
            or proof['full_input_output_model_token_FNV_equality'] is not True
            or proof['production_TUI_startup_claimed'] is not False
            or run['process_exit_code']!=0 or run['CPU_model_fallback'] is not False
            or run['native_generate'] is not True or type(run['DSP_calls']) is not int
            or run['DSP_calls']!=proof['DSP_calls'] or run['DSP_calls']!=oracle['DSP_calls']
            or len(run['forwards'])!=run['DSP_calls']
            or run['frontend_invocations'] or run['machine_code_compilations']
            or read(a.native/'run-exit.json')!={'board_returncode':0,'worker_returncode':0}):
        raise ValueError('Own normal native timing, complete oracle equality and no inline observer required')
    compare(oracle,run);audit_arm_forward_timings(run)
    if verify_native_layout(oracle)!=verify_native_layout(run):
        raise ValueError('Actual original constant payloads and loader layout changed')
    if [r['terminal_observation'] for r in oracle['requests']]!=[r['terminal_observation'] for r in run['requests']]:
        raise ValueError('Actual native stopping behavior changed')
    worker=proof['worker'];mailbox=Path(worker[worker.index('--mailbox')+1])
    chain=audit_compile_handoff(run,mailbox,rebuilt=True)
    if chain!=proof['complete_compile_chain_rehashed']:raise ValueError('Actual full compiler reply chain changed')
    controllers=[]
    for path in (a.oracle_controller,a.native_controller):
        owner=read(path)
        if (owner['status']!='completed' or owner['returncode']!=0
                or owner['completed_unix']>=owner['authorized_deadline_unix']
                or any(Path('/proc',str(owner[k])).exists() for k in ('controller_pid','child_pid'))):
            raise ValueError('Normal actual authorized controllers required')
        controllers.append(entry(path))
    config=read(a.config);raws=[];ap=[]
    for folder,native,commands in ((a.oracle,oracle,read(a.oracle/'commands.json')),(a.native,run,proof)):
        if any(Path('/proc',str(pid)).exists() for pid in read(folder/'active-processes.json').values()):
            raise ValueError('Actual host session owners must be gone')
        board=commands['board'];output=Path(board[board.index('--output')+1])
        raw=Path(config['observations'])/output.name/'stream.stdout.txt';events=[json.loads(line) for line in raw.read_text().splitlines()]
        if (not events or events[-1]['stage']!='close'
                or [e for e in events if e.get('stage')=='invoke']!=[f['invoke'] for f in native['forwards']]
                or any(len({e[k] for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')})!=1
                    for e in events if e.get('stage')=='invoke')
                or any(e.get(k,0) for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
            raise ValueError('Complete raw invocation stream and normal close required')
        ready=[e for e in events if e.get('stage')=='ready']
        if len(ready)!=1 or type(ready[0].get('ARM_pid')) is not int:raise ValueError('Actual AP owner identity required')
        ap.append(ready[0]['ARM_pid']);raws.append(entry(raw))
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],shlex.join([
        config['board']['python'],'-c','from pathlib import Path;import sys;assert all(not Path("/proc",p).exists() for p in sys.argv[1:])',*map(str,ap)])],check=True)
    write(a.output,dict(status='native_startup_sequential_spans_full_oracle_raw_FNV_and_normal_owners_independently_revisited',
        DSP_calls=oracle['DSP_calls']+run['DSP_calls'],new_DSP_calls=0,new_model_PRIMARY_compilations=0,
        model_inference_acceptance=False,performance_acceptance=False,diagnostic_only=True,
        oracle=entry(a.oracle/'acceptance.json'),native=entry(a.native/'result.json'),controllers=controllers,
        full_original_input_output_token_and_three_FNV_equality=True,compile_chain=chain,raw_invocation_traces=raws,
        actual_AP_owners_gone=ap,summary=startup_summary(run),inline_CPU_observer=False,
        reviewer_source=entry(__file__),reviewer_revision=git(ROOT,'rev-parse','HEAD'),completed_unix=time.time()))


if __name__=='__main__':main()
