"""Revisit cached native state transitions through the original full-packet oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from build import ROOT,entry,sha,write
from probe_native_aot_online_regression import observed_command,session_command
from revisit_native_compile_session_abba import require_normal_owner


def require_observed_cache_protocol(cache,native,manifest):
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    payload=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if (cache['status']!='original_native_observer_completed_with_recorded_AOT_cache_protocol_DIAGNOSTIC'
            or cache['namespace']!=digest or cache['manifest_sha256']!=payload or cache['aot_mode']!='cache'
            or cache['native_source_sha256']!=native['source_sha256'] or cache['adapter_sha256']!=manifest['adapter_sha256']
            or cache['factory_restored'] is not True or cache['original_configuration_restored'] is not True
            or cache['original_online_CPU_reference_and_actual_DSP_observer_retained'] is not True
            or cache['production_configuration_changed'] is not False
            or cache['model_inference_acceptance'] is not False or cache['performance_acceptance'] is not False
            or manifest['module_constants_sha256']!=native['parameter_validation']['guarded_python_float_constants']['ledger_sha256']
            or cache['original_configuration']['bypass_autograd_cache_key'] is not False
            or cache['original_configuration']['unsafe_skip_cache_dynamic_shape_guards'] is not False):
        raise ValueError('Original native observer, all constants and unchanged full guard protocol required')
    events=cache['events'];counters=cache['framework_cache_counters_delta']
    if (len(events)!=len(native['captures']) or not 1<=len(events)<=2
            or [e['graph'] for e in events]!=[c['graph'] for c in native['captures']]
            or any(e['stage'] not in ('original_native_compiler_on_AOT_miss','original_native_compiler_on_AOT_hit_rebind') for e in events)
            or any(e.get('current_model_rebound') is not True for e in events if e['stage'].endswith('hit_rebind'))):
        raise ValueError('Every actual capture must use original miss or current-model cached rebind')
    hits=sum(e['stage'].endswith('hit_rebind') for e in events);misses=len(events)-hits
    expected=dict(total=len(events),autograd_cache_hit=hits,autograd_cache_miss=misses,
                  autograd_cache_saved=misses,autograd_cache_bypass=0)
    if (any(type(counters.get(k,0)) is not int or counters.get(k,0)!=v for k,v in expected.items())
            or type(counters.get('autograd_cache_guard_miss',0)) is not int
            or counters.get('autograd_cache_guard_miss',0)<0):
        raise ValueError('Recorded framework hit/miss/save counts and no cache bypass required')
    return dict(captures=len(events),hits=hits,misses=misses,guard_misses=counters.get('autograd_cache_guard_miss',0),
                ordinary_guard_miss_fallback_allowed=True,unsafe_guard_skip=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline','native','controller','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda root,*args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git(ROOT,'status','--porcelain'):
        raise ValueError('Fresh absolute independent proof and clean reviewer source required')
    outer=read(a.controller);owner=require_normal_owner(a.controller,outer['authorized_deadline_unix'])
    protocol=read(a.native/'AOT-online-protocol-controller.json');cache=read(a.native/'AOT-cache-report.json')
    if (owner['source_root']!=str(ROOT) or owner['command'][1]!=protocol['driver_source']['path']
            or owner['command'][owner['command'].index('--output')+1]!=str(a.native)
            or protocol['deadline_unix']!=owner['authorized_deadline_unix']):
        raise ValueError('Actual normal owner must execute this original observed diagnostic within its deadline')
    online=Path(protocol['online_source_root']);numerical=Path(protocol['numerical_source_root'])
    for root,rev in ((online,protocol['online_source_revision']),(numerical,protocol['numerical_source_revision'])):
        if git(root,'rev-parse','HEAD')!=rev or git(root,'status','--porcelain'):
            raise ValueError('Both original observer and numerical source must remain frozen')
    sys.path[:0]=[str(numerical/'tools/qwen3'),str(numerical/'qwen3/runtime/arm'),str(numerical)]
    from compare_performance import accepted,compare
    from run_integer_regressions import validate_float_policy,validate_parameter_mode
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    from bounded_loader_policy import verify_native_layout
    from hexagon_host_timing import audit_arm_forward_timings
    _,baseline,_=accepted(a.baseline);acceptance,native,_=accepted(a.native);commands=read(a.native/'commands.json')
    if (protocol['status']!='original_online_numerical_acceptance_with_recorded_native_AOT_protocol_DIAGNOSTIC'
            or protocol['original_online_acceptance']!=entry(a.native/'acceptance.json')
            or protocol['AOT_cache_report']!=entry(a.native/'AOT-cache-report.json')
            or protocol['model_inference_acceptance'] is not False or protocol['performance_acceptance'] is not False
            or protocol['production_configuration_changed'] is not False or native['process_exit_code']!=0
            or native['CPU_model_fallback'] is not False or native['frontend_invocations']!=0 or native['machine_code_compilations']!=0
            or acceptance['numerical_revision']!=protocol['numerical_source_revision']
            or commands['revision']!=protocol['online_source_revision']
            or any(Path('/proc',str(pid)).exists() for pid in read(a.native/'active-processes.json').values())):
        raise ValueError('Own original independent full-packet state acceptance and normal owners required')
    def source(row,root,revision):
        relative=str(Path(row['path']).relative_to(root))
        blob=subprocess.check_output(['git','-C',str(root),'show',revision+':'+relative])
        if hashlib.sha256(blob).hexdigest()!=row['sha256'] or entry(row['path'])!=row:
            raise ValueError('Actual committed producer source differs')
    source(protocol['driver_source'],ROOT,owner['source_revision']);source(protocol['worker_source'],ROOT,owner['source_revision'])
    for key in ('delegate_source','original_native_source','original_observer_source'):
        source(protocol[key],online,protocol['online_source_revision'])
    for row in protocol['deployed_sources']:
        source(row['source'],ROOT,owner['source_revision'])
        key={'native_aot_output.py':'adapter_sha256','run_native_aot_observed_arm.py':'wrapper_sha256'}[Path(row['source']['path']).name]
        if cache[key]!=row['source']['sha256']:raise ValueError('Actual executed adapter/wrapper differs')
    if cache['observer_source_sha256']!=protocol['original_observer_source']['sha256']:
        raise ValueError('Original packet observer executable changed')
    manifest=protocol['manifest'];context=read(commands['worker'][commands['worker'].index('--context')+1])
    if (manifest['model_parameter_sha256']!=context['historical_parameter_sha256']
            or manifest['numerical_revision']!=protocol['numerical_source_revision']
            or manifest['recipe_sha256']!=commands['recipe']['sha256']
            or manifest['runtime_catalog_sha256']!=commands['runtime_override']['sha256']):
        raise ValueError('Full original model, numerical recipe and runtime namespace required')
    summary=require_observed_cache_protocol(cache,native,manifest)
    observers=protocol['actual_observer_launches'];workers=protocol['actual_worker_launches']
    if len(observers)!=1 or len(workers)!=1:raise ValueError('Exactly one recorded observer and cache-only worker required')
    words=shlex.split(observers[0]['actual'][-1]);original_words=shlex.split(observers[0]['original'][-1])
    if original_words!=commands['board'] or workers[0]['original']!=commands['worker']:
        raise ValueError('Recorded original launches differ from original coordinator')
    args=words[5:words.index('--sources')]
    expected,changed=observed_command(observers[0]['original'],commands['board'][:4],commands['board'][4],words[4],args)
    if not changed or expected!=observers[0]['actual']:raise ValueError('Actual wrapper changed original observer arguments')
    actual,changed=session_command(workers[0]['original'],commands['worker'][:2],protocol['worker_source']['path'],
        numerical,protocol['numerical_source_revision'],protocol['deadline_unix'])
    if not changed or actual!=workers[0]['actual']:raise ValueError('Actual worker is not the full-check cache-only session')
    compare(baseline,native);validate_parameter_mode(native,'exact-bytes');validate_float_policy(native,True)
    audit_arm_forward_timings(native)
    if (verify_native_layout(baseline)!=verify_native_layout(native)
            or native['parameter_validation']['guarded_python_float_constants']!=baseline['parameter_validation']['guarded_python_float_constants']
            or [r['terminal_observation'] for r in baseline['requests']]!=[r['terminal_observation'] for r in native['requests']]):
        raise ValueError('Original loader, module constants or native stopping behavior changed')
    config=read(a.config);mailbox=Path(commands['worker'][commands['worker'].index('--mailbox')+1])
    chain=audit_compile_handoff(native,mailbox,rebuilt=True)
    if chain!=acceptance['compile_handoff']:raise ValueError('All actual compiler replies and artifacts must rehash')
    output=Path(config['observations'])/Path(commands['board'][commands['board'].index('--output')+1]).name
    raw=output/'stream.stdout.txt';events=[json.loads(line) for line in raw.read_text().splitlines()]
    if (not events or events[-1]['stage']!='close'
            or [e for e in events if e.get('stage')=='invoke']!=[f['invoke'] for f in native['forwards']]
            or any(len({e[k] for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')})!=1 for e in events if e.get('stage')=='invoke')
            or any(e.get(k,0) for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
        raise ValueError('Complete raw AP/RPC/DSP invokes and normal close required')
    ready=[e for e in events if e.get('stage')=='ready']
    if len(ready)!=1 or type(ready[0]['ARM_pid']) is not int:raise ValueError('Actual AP owner PID required')
    ap=ready[0]['ARM_pid']
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],shlex.join([
        config['board']['python'],'-c','from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()',str(ap)])],check=True)
    write(a.output,dict(status='native_AOT_original_online_state_packets_full_guards_raw_FNV_and_owners_independently_revisited_DIAGNOSTIC',
        baseline=entry(a.baseline/'acceptance.json'),native=entry(a.native/'acceptance.json'),controller=entry(a.controller),
        AOT_protocol=entry(a.native/'AOT-online-protocol-controller.json'),AOT_cache_report=entry(a.native/'AOT-cache-report.json'),
        cache=summary,DSP_calls=native['DSP_calls'],new_DSP_calls=0,compile_chain=chain,
        raw_invocation_trace=entry(raw),actual_AP_owner_gone=ap,
        original_reference_observer_full_parameters_142_FP64_and_stopping_retained=True,
        model_inference_acceptance=False,performance_acceptance=False,production_configuration_changed=False,
        reviewer_source=entry(__file__),reviewer_revision=git(ROOT,'rev-parse','HEAD'),completed_unix=time.time()))


if __name__=='__main__':main()
