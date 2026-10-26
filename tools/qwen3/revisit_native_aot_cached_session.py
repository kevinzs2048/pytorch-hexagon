"""Independently rehash native16, raw invokes, all guards and AOT cache protocol.

This is diagnostic closure only. Cache counters or timings alone never become
model acceptance; every original input/output packet and compile chain remains
required through the preaccepted full-packet numerical oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from build import ROOT,entry,sha,write
from probe_native_aot_cached_session import wrapped_native_command
from revisit_native_compile_session_abba import require_normal_owner,require_exact_parameter_guards
from revisit_native_startup import startup_summary


def require_cache_protocol(cache,native,manifest,case):
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    payload=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if (cache['status']!='original_native_entry_completed_with_recorded_AOT_cache_protocol_DIAGNOSTIC'
            or cache['factory_restored'] is not True or cache['original_configuration_restored'] is not True
            or cache['namespace']!=digest or cache['manifest_sha256']!=payload
            or cache['native_source_sha256']!=native['source_sha256']
            or cache['adapter_sha256']!=manifest['adapter_sha256']
            or cache['production_configuration_changed'] is not False
            or cache['model_inference_acceptance'] is not False or cache['performance_acceptance'] is not False
            or manifest['module_constants_sha256']!=native['parameter_validation']['guarded_python_float_constants']['ledger_sha256']
            or cache['original_configuration']['bypass_autograd_cache_key'] is not False
            or cache['original_configuration']['unsafe_skip_cache_dynamic_shape_guards'] is not False):
        raise ValueError('Actual original native source, all model guards and deterministic AOT protocol required')
    events=cache['events'];counters=cache['framework_cache_counters_delta']
    if case=='original':
        # Ordinary AOT compilation counts each compiled capture even when no
        # native output cache is installed. These are not cache hit counters.
        expected={'total':len(native['captures']),'ok':len(native['captures'])}
        if (cache['aot_mode']!='original' or events or len(native['captures'])!=2
                or any(type(counters.get(k,0)) is not int or counters.get(k,0)!=v
                       for k,v in expected.items())
                or any(type(v) is not int or v!=0 for k,v in counters.items() if k not in expected)):
            raise ValueError('Original arm must retain the uncached AOT compiler')
        return
    stage='original_native_compiler_on_AOT_'+('miss' if case=='seed' else 'hit_rebind')
    if (cache['aot_mode']!='cache' or len(events)!=len(native['captures']) or len(events)!=2
            or [event['stage'] for event in events]!=[stage,stage]
            or [event['graph'] for event in events]!=[capture['graph'] for capture in native['captures']]):
        raise ValueError('Exactly both original captured graphs must use the required native cache path')
    expected={'autograd_cache_hit':0 if case=='seed' else 2,
        'autograd_cache_miss':2 if case=='seed' else 0,'autograd_cache_saved':2 if case=='seed' else 0,
        'autograd_cache_bypass':0,'autograd_cache_guard_miss':0}
    if any(type(counters.get(key,0)) is not int or counters.get(key,0)!=value for key,value in expected.items()):
        raise ValueError('Actual framework two-graph save/hit and no bypass required')
    if case=='hit' and any(event['current_model_rebound'] is not True for event in events):
        raise ValueError('Every cached graph must bind the current original native model')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','oracle','native','controller','numerical-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cache-case',choices=('original','seed','hit'),required=True)
    args=parser.parse_args();read=lambda p:json.loads(Path(p).read_text())
    git=lambda root,*words:subprocess.check_output(['git','-C',str(root),*words],text=True).strip()
    if not args.output.is_absolute() or args.output.exists() or git(ROOT,'status','--porcelain'):
        raise ValueError('Fresh result and frozen clean independent reviewer required')
    outer=read(args.controller);owner=require_normal_owner(args.controller,outer['authorized_deadline_unix'])
    sys.path[:0]=[str(args.numerical_source/'tools/qwen3'),str(args.numerical_source/'qwen3/runtime/arm'),str(args.numerical_source)]
    from compare_performance import accepted,compare
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    from hexagon_host_timing import audit_arm_forward_timings
    from bounded_loader_policy import verify_native_layout
    acceptance,oracle,_=accepted(args.oracle)
    if git(args.numerical_source,'rev-parse','HEAD')!=acceptance['numerical_revision'] or git(args.numerical_source,'status','--porcelain'):
        raise ValueError('Own unchanged numerical oracle source required')
    proof=read(args.native/'result.json');native=read(args.native/'native-result.json')
    protocol=read(args.native/'AOT-protocol-controller.json');cache=read(args.native/'AOT-cache-report.json')
    if (proof['status']!='host_worker_native16_full_packet_oracle_and_compile_chain_matched_DIAGNOSTIC'
            or protocol['status']!='original_native16_with_actual_recorded_AOT_protocol_DIAGNOSTIC'
            or sha(args.native/'native-result.json')!=proof['native_sha256']
            or protocol['native_coordinator']!=entry(args.native/'result.json')
            or protocol['AOT_cache_report']!=entry(args.native/'AOT-cache-report.json')
            or native['process_exit_code']!=0 or native['CPU_model_fallback'] is not False
            or type(native['DSP_calls']) is not int or native['DSP_calls']!=16
            or native['frontend_invocations']!=0 or native['machine_code_compilations']!=0
            or read(args.native/'run-exit.json')!={'board_returncode':0,'worker_returncode':0}
            or any(Path('/proc',str(pid)).exists() for pid in proof['active_processes'].values())):
        raise ValueError('Actual native16, original packet/compile closure and normal owners required')
    for row in (protocol['driver_source'],protocol['delegate_source'],proof['worker_source']):
        relative=str(Path(row['path']).relative_to(ROOT))
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',owner['source_revision']+':'+relative])
        if hashlib.sha256(blob).hexdigest()!=row['sha256'] or sha(row['path'])!=row['sha256']:
            raise ValueError('Actual producer executable or current pinned bytes changed')
    for deployed in protocol['deployed_sources']:
        row=deployed['source']
        if sha(row['path'])!=row['sha256']:raise ValueError('Recorded adapter/wrapper bytes changed')
        key={'native_aot_output.py':'adapter_sha256','run_native_aot_cached_arm.py':'wrapper_sha256'}[Path(row['path']).name]
        if cache[key]!=row['sha256']:raise ValueError('Actual executed adapter/wrapper source differs')
    worker=proof['worker'];context=read(worker[worker.index('--context')+1])
    if protocol['manifest']['model_parameter_sha256']!=context['historical_parameter_sha256']:
        raise ValueError('Original complete FP32 model identity differs from namespace')
    require_cache_protocol(cache,native,protocol['manifest'],args.cache_case)
    launch=protocol['actual_native_launches']
    if len(launch)!=1:raise ValueError('Exactly one actual recorded wrapped native launch required')
    words=shlex.split(launch[0]['actual'][-1]);wrapper=words[4]
    flags=words[5:words.index('--catalog')]
    actual,changed=wrapped_native_command(launch[0]['original'],proof['board'][:5],wrapper,flags[2:])
    if not changed or actual!=launch[0]['actual']:
        raise ValueError('Recorded wrapper changed original native argv')
    compare(oracle,native);audit_arm_forward_timings(native);require_exact_parameter_guards(native,oracle)
    if (verify_native_layout(oracle)!=verify_native_layout(native)
            or [r['terminal_observation'] for r in oracle['requests']]!=[r['terminal_observation'] for r in native['requests']]):
        raise ValueError('Original loader or stopping behavior changed')
    chain=audit_compile_handoff(native,Path(proof['host_mailbox']),rebuilt=True)
    if chain!=proof['complete_compile_chain_rehashed'] or startup_summary(native)!=proof['startup']:
        raise ValueError('Original complete compiler chain or disjoint timing changed')
    config=read(args.config);raw=Path(config['observations'])/Path(proof['native_output']).name/'stream.stdout.txt'
    events=[json.loads(line) for line in raw.read_text().splitlines()]
    if (not events or events[-1]['stage']!='close'
            or [e for e in events if e.get('stage')=='invoke']!=[f['invoke'] for f in native['forwards']]
            or any(len({e[k] for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')})!=1
                for e in events if e.get('stage')=='invoke')
            or any(e.get(k,0) for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
        raise ValueError('All actual raw RPC/DSP invokes and normal close required')
    ready=[e for e in events if e.get('stage')=='ready']
    if len(ready)!=1 or type(ready[0]['ARM_pid']) is not int:raise ValueError('Actual AP owner required')
    ap=ready[0]['ARM_pid']
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],shlex.join([
        config['board']['python'],'-c','from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()',str(ap)])],check=True)
    write(args.output,dict(status='native_AOT_protocol16_all_original_packets_full_guards_raw_FNV_and_normal_owners_independently_revisited_DIAGNOSTIC',
        oracle=entry(args.oracle/'acceptance.json'),native=entry(args.native/'result.json'),controller=entry(args.controller),
        AOT_protocol=entry(args.native/'AOT-protocol-controller.json'),AOT_cache_report=entry(args.native/'AOT-cache-report.json'),
        cache_case=args.cache_case,DSP_calls=16,new_DSP_calls=0,compile_chain=chain,
        raw_invocation_trace=entry(raw),actual_AP_owner_gone=ap,summary=startup_summary(native),
        original_full_parameters_and_142_FP64_guards_retained=True,model_inference_acceptance=False,
        performance_acceptance=False,TUI_startup_claimed=False,reviewer_source=entry(__file__),
        reviewer_revision=git(ROOT,'rev-parse','HEAD'),completed_unix=time.time()))


if __name__=='__main__':main()
