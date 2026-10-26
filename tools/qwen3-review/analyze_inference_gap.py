"""Attribute warm execution without adding nested timers or capture overhead."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def partition(row, token_seconds):
    arm = row['arm_timing']; stages = arm['stages_ns']; event = row['invoke']; p = event['profile']
    assert sum(stages.values()) == arm['total_ns']
    assert not any(event[k] for k in ('rpc_result','DSP_result','output_result'))
    assert len({event[k] for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')}) == 1
    s = {k:v/1e9 for k,v in stages.items()}
    kernel = event['kernel_us']/1e6
    entry = p['DSP_entry_total_us']/1e6
    checks = (p['DSP_rpc_checksum_us']+p['DSP_worker_checksum_us'])/1e6
    exclusive = dict(
        DSP_model_kernel=kernel,
        DSP_entry_wrapper=entry-kernel,
        DSP_input_checks=checks,
        RPC_unattributed=event['seconds']-entry-checks,
        AP_buffer_operations=p['AP_total_seconds']-event['seconds'],
        Python_AP_exchange=s['exchange_with_AP']-p['AP_total_seconds'],
        ARM_parameter_validation=s['validate_frozen_parameters'],
        ARM_input_packing=s['pack_and_validate_inputs'],
        ARM_packet_SHA=s['record_request']+s['record_response'],
        ARM_decode_select_cleanup=s['reconstruct_owned_outputs']+s['select_compiled_library']+s['release_file_scratch'],
        framework_AOT_writeback_and_token_selection=token_seconds-arm['total_ns']/1e9)
    assert all(math.isfinite(x) and x >= -0.001 for x in exclusive.values())
    assert abs(sum(exclusive.values())-token_seconds) < 1e-8
    return exclusive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--profiles', type=Path, nargs='+', required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--downstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if not a.output.is_absolute() or a.output.exists():
        p.error('Fresh absolute analysis output required')
    sys.path[:0]=[str(a.downstream/'tools/qwen3'),str(a.downstream/'test/python/torch-mlir'),str(a.downstream)]
    from compare_performance import accepted,compare
    from probe_model_memory_arm import parse_profile
    from integer_call_timeline import parse_attention_timeline
    _,oracle,_=accepted(a.oracle)
    read=lambda path:json.loads(Path(path).read_text())
    samples=[];runs=[];prefill=[];libraries={}
    for path in a.runs:
        d=read(path/'native-result.json');terminal=read(path/'result.json')
        assert terminal['status']=='accepted_dynamic_model_native_AOT_generation_completed'
        assert terminal['worker'] is None and terminal['CPU_reference_executed'] is False
        assert d['frontend_invocations']==d['machine_code_compilations']==0 and d['CPU_model_fallback'] is False
        compare(oracle,d)
        events=[json.loads(s) for s in (path/'stream.stdout.txt').read_text().splitlines()]
        assert [e for e in events if e.get('stage')=='invoke']==[f['invoke']for f in d['forwards']]
        assert events[-1]['stage']=='close' and events[-1]['rpc_result']==0
        first_decode=True;kept=[]
        for request in d['requests']:
            forwards=[f for f in d['forwards']if f['request_index']==request['index']]
            decode=[f for f in forwards if f['query']==1]
            intervals=request['generation_timing']['inter_token_ns']
            assert len(decode)==len(intervals) and all(f['query']>1 for f in forwards if f not in decode)
            for row in forwards:
                key='q1' if row['query']==1 else 'q2_to_32'
                if key in libraries:assert libraries[key]==row['DSO_sha256']
                libraries[key]=row['DSO_sha256']
            for row,ns in zip(decode,intervals):
                if first_decode:
                    first_decode=False;continue
                seconds=ns/1e9;parts=partition(row,seconds)
                sample=dict(run=str(path),step=row['step'],request_index=request['index'],
                    cache_positions=row['cache_positions'],token_seconds=seconds,
                    callback_seconds=row['arm_timing']['total_ns']/1e9,exclusive_seconds=parts)
                samples.append(sample);kept.append(sample)
            if request['index']>0:
                for row in forwards:
                    if row['query']>1:
                        prefill.append(dict(run=str(path),query=row['query'],kernel_seconds=row['invoke']['kernel_us']/1e6,
                            callback_seconds=row['arm_timing']['total_ns']/1e9,
                            token_ready_seconds=request['generation_timing']['first_generated_token_ready_ns']/1e9))
        runs.append(dict(native=ref(path/'native-result.json'),result=ref(path/'result.json'),DSP_calls=d['DSP_calls'],
            steady_decode_samples=len(kept),median_token_seconds=statistics.median(x['token_seconds']for x in kept)))
    means={k:statistics.mean(x['exclusive_seconds'][k]for x in samples)for k in samples[0]['exclusive_seconds']}
    total=statistics.mean(x['token_seconds']for x in samples)
    assert abs(sum(means.values())-total)<1e-8
    diagnostics=[]
    for path in a.profiles:
        d=read(path/'board/result.json');owner=read(path/'process.json')
        assert owner['status']=='complete' and owner['returncode']==owner['output_copy_returncode']==0
        assert d['status']=='diagnostic_model_memory_complete_packet_bitexact' and d['process_exit_code']==0
        key='q1'if d['query']==1 else 'q2_to_32'
        assert d['baseline_library_sha256']==libraries[key]
        assert d['model_calls']==4 and d['diagnostic_query_calls']==2 and len(d['profiles'])==2
        raw=[json.loads(s)for s in (path/'board/stream.stdout.txt').read_text().splitlines()]
        assert raw==d['stream_events'] and raw[-1]['stage']=='close' and raw[-1]['rpc_result']==0
        for row in d['rows']:
            assert row['all_logits_and_56_KV_byteexact'] is True and row['output_sha256']==d['expected_output_sha256']
            event=row['actual_invoke']
            assert len({event[k]for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')})==1
        for row in d['profiles']:
            blob=(path/'board'/('profile-'+str(row['index'])+'.bin')).read_bytes()
            if 'attention_phases'in row['profile']:
                parsed=parse_attention_timeline(blob,d['query'])
            else:
                kinds=[int(k)for k in d['external_profile_symbols']]
                parsed=parse_profile(blob,kinds,kinds if kinds==[13]else [1,2,3,10])
            assert parsed==row['profile']
        baseline=statistics.mean(r['actual_invoke']['kernel_us']/1e6 for r in d['rows']if r['library']=='baseline')
        instrumented=statistics.mean(r['actual_invoke']['kernel_us']/1e6 for r in d['rows']if r['library']=='profile')
        summary=dict(result=ref(path/'board/result.json'),query=d['query'],baseline_kernel_seconds=baseline,
            instrumented_kernel_seconds=instrumented,instrumentation_ratio=instrumented/baseline,
            model_calls=4,query_calls=2,full_output_bytes=235488768,full_output_SHA=d['expected_output_sha256'])
        if 'attention_phases'in d['profiles'][0]['profile']:
            profiles=[r['profile']for r in d['profiles']]
            summary['timeline_seconds']={k.removesuffix('_us'):statistics.mean(x[k]for x in profiles)/1e6
                for k in ('total_us','callee_us','gaps_us','tail_us')}
            summary['attention_intervals_seconds']={k.removesuffix('_us'):statistics.mean(x['attention_phase_totals'][k]for x in profiles)/1e6
                for k in profiles[0]['attention_phase_totals']}
            summary['attention_interval_fraction']=summary['attention_intervals_seconds']['complete_gap']/summary['timeline_seconds']['total']
        else:
            summary['external_calls']={name:dict(calls=d['profiles'][0]['profile']['totals'][k]['calls'],
                bytes=d['profiles'][0]['profile']['totals'][k]['bytes'],
                seconds=statistics.mean(r['profile']['totals'][k]['microseconds']for r in d['profiles'])/1e6)
                for k,name in d['external_profile_symbols'].items()}
        diagnostics.append(summary)
    result=dict(status='warm_execution_and_exact_current_model_diagnostics_revisited',runs=runs,libraries=libraries,
        scope='two_short_prompts_batch1_KV1024_greedy_pinned_downstream_NOT_upstream_default_benchmark',
        cold_capture_load_and_compilation_excluded_from_steady_decode=True,
        per_process_first_q1_sample_excluded=True,inline_CPU_reference=False,new_model_compilations=0,
        steady_decode_samples=len(samples),median_token_seconds=statistics.median(x['token_seconds']for x in samples),
        mean_token_seconds=total,mean_exclusive_seconds=means,
        fractions={k:v/total for k,v in means.items()},warm_prefill=prefill,diagnostics=diagnostics,samples=samples,
        notes=['RPC residual is not pure transfer.','Attention intervals include arithmetic, layouts and guards; they are not isolated QK/PV costs.',
            'The diagnostic model includes downstream numerical policies and helpers; DSP time alone does not prove an upstream compiler bug.',
            'Per-call profiler overhead is measured by ABBA and must not be interpreted as original production time.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['status','steady_decode_samples','median_token_seconds','mean_exclusive_seconds']},indent=2))


if __name__=='__main__':main()
