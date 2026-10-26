"""Start one full q1 closure after actual transport exit and packet revalidation.

The supervisor owns no DSP session while waiting. All original cold9, warm16,
native64 and state304 work remains in run_helper_effects_iteration.py.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from build import ROOT,entry,sha,write
from compare_performance import accepted
from run_cold_boundary_probes import identity
from run_integer_regressions import candidate_libraries,compare_integer,validate_float_policy,validate_parameter_mode
from run_regression import require_configured_python


def replay_host(host,recipe,cold):
    """Rehash the public cache with today's sources, never invoke a compiler."""
    sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
    from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
    from reuse_accepted_compiler_objects import ReadOnlyCompilerCache
    captures=json.loads((cold/'native-result.json').read_text())['captures']
    if len(host['rows'])!=2 or {tuple(r['query_bounds']) for r in host['rows']}!={(1,1),(2,32)}:
        raise ValueError('Both actual public query domains required')
    proofs=[]
    for row in host['rows']:
        if entry(row['frontend']['path'])!=row['frontend'] or sha(row['DSO']['path'])!=row['DSO']['sha256']:
            raise ValueError('Public frontend or tested runtime bytes changed')
        folder=Path(row['DSO']['path']).parent.parent
        capture=[r for r in captures if r['contract']['variables']['query']==row['query_bounds']]
        if len(capture)!=1:raise ValueError('Unambiguous actual query contract required')
        frontend_path=Path(row['frontend']['path'])
        cache=CapturedQwenMachineCache(folder.parent,recipe,sha(recipe),frontend_path.parent)
        cache.cache=ReadOnlyCompilerCache(cache.cache,folder.name[:64])
        result=cache.get_or_compile(json.loads(frontend_path.read_text()),runtime_contract=capture[0]['contract'])
        if (not result['cache_hit'] or result['machine_code_compilations']
                or result['model_object_codegen_invocations'] or Path(result['directory'])!=folder):
            raise ValueError('Current public cache must authenticate without any compiler')
        proofs.append(dict(query_bounds=row['query_bounds'],machine_key=result['key'],DSO=row['DSO']))
    return dict(rows=proofs,DSP_calls=0,compiler_invocations=0)


def finished_process(path,revision=None):
    process=json.loads(path.read_text())
    if (process['status'] not in ('complete','completed') or process['returncode']!=0
            or type(process.get('controller_pid')) is not int
            or Path('/proc',str(process['controller_pid'])).exists()
            or revision is not None and process['source_revision']!=revision):
        raise ValueError('Actual normal controller exit and original process absence required')
    return process


def native_transport(closure,baseline):
    groups={};proofs=[]
    _,libraries=candidate_libraries(baseline/'cold')
    for index,kind in enumerate(('baseline','candidate','candidate','baseline')):
        folder=closure/('native-abba-'+str(index)+'-'+kind)
        proof=json.loads((folder/'result.json').read_text());native=json.loads((folder/'native-result.json').read_text())
        oracle=closure/'warm16' if kind=='candidate' else baseline/'warm16'
        _,original,_=accepted(oracle)
        if (proof['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
                or proof['DSP_calls']!=16 or proof['inline_CPU_observer'] is not False
                or proof['full_input_output_model_token_FNV_equality'] is not True
                or proof['native_sha256']!=sha(folder/'native-result.json')
                or proof['oracle_native_sha256']!=sha(oracle/'native-result.json')
                or json.loads((folder/'run-exit.json').read_text())!={'board_returncode':0,'worker_returncode':0}
                or native['process_exit_code'] or native['CPU_model_fallback'] or native['DSP_calls']!=16):
            raise ValueError('Original complete native16 and normal exit required')
        compare_integer(original,native,libraries);validate_parameter_mode(native,'exact-bytes');validate_float_policy(native,True)
        proofs.append(entry(folder/'result.json'))
        for query,row in proof['measurement']['query_groups'].items():
            values=row['median_seconds'];groups.setdefault(query,[]).append(dict(kind=kind,
                kernel_seconds=values['DSP_kernel_seconds'],callback_seconds=values['ARM_callback_seconds']))
    timing={}
    for query,rows in groups.items():
        if len(rows)!=4:raise ValueError('Complete original native ABBA query coverage required')
        timing[query]={}
        for metric in ('kernel_seconds','callback_seconds'):
            old=statistics.mean(r[metric] for r in rows if r['kind']=='baseline')
            new=statistics.mean(r[metric] for r in rows if r['kind']=='candidate')
            timing[query][metric]=dict(baseline=old,candidate=new,ratio=new/old)
    if timing!=json.loads((closure/'result.json').read_text())['native_model_timing']:
        raise ValueError('Recorded transport timing differs from actual original reports')
    return dict(proofs=proofs,measurements=timing,DSP_calls=64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','recipe-build','early-model-revisit','host-preflight','host-process',
                 'baseline-closure','original-state-suite','transport-stage','transport-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('early-model-revisit','host-preflight','host-process'):
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--name',required=True);p.add_argument('--deadline-unix',type=int,required=True)
    p.add_argument('--preflight-only',action='store_true',help='Revalidate the current public cache without waiting or DSP work')
    a=p.parse_args()
    if (not a.output.is_absolute() or a.output.exists() or a.output.is_symlink()
            or a.deadline_unix-time.time()<4*3600):p.error('Fresh full393 output and four-hour start budget required')
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed supervisor and numerical source required')
    for name in ('early_model_revisit','host_preflight','host_process'):
        if sha(getattr(a,name))!=getattr(a,name+'_sha256'):raise ValueError('Actual preflight or early proof changed')
    finished_process(a.host_process)
    host=json.loads(a.host_preflight.read_text());recipe=a.recipe_build/'machine-recipe.json'
    if (host['status']!='actual_two_query_graphs_exact_objects_reused_and_tested_runtime_DSOs_reproduced_NOT_DSP_acceptance'
            or host['recipe']!=entry(recipe) or host['model_revisit']!=entry(a.early_model_revisit)
            or host['new_model_object_codegen_invocations'] or host['DSP_calls']):
        raise ValueError('Actual public two-graph recipe preflight required')
    config=json.loads(a.config.read_text());require_configured_python(config)
    host_replay=replay_host(host,recipe,a.baseline_closure/'cold')
    a.output.mkdir()
    record=dict(status='waiting_for_actual_transport_normal_exit',supervisor_pid=os.getpid(),
        source_revision=revision,source_sha256=sha(__file__),start_unix=time.time(),
        host_preflight=entry(a.host_preflight),early_model_revisit=entry(a.early_model_revisit),
        recipe=entry(recipe),config=entry(a.config),host_cache_replay=host_replay,
        DSP_calls=0,model_inference_acceptance=False)
    save=lambda:write(a.output/'supervisor.json',record)
    def pinned():
        if (subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=revision
                or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
                or entry(recipe)!=record['recipe'] or entry(a.config)!=record['config']):
            raise ValueError('Frozen supervisor source, configuration or recipe changed')
    save()
    try:
        if a.preflight_only:
            record.update(status='current_source_public_cache_revalidated_without_compilation_or_DSP',completed_unix=time.time())
            save();print(record['status']);return
        process_path=a.transport_stage/'controller-process.json'
        while True:
            pinned()
            try:process=json.loads(process_path.read_text())
            except json.JSONDecodeError:
                if a.deadline_unix-time.time()<4*3600:raise TimeoutError('Insufficient remaining full393 start budget')
                time.sleep(1);continue
            if process['status'] in ('complete','completed'):
                finished_process(process_path);break
            live=identity(process['controller_pid'])
            if (process['status']!='running' or live['start_tick']!=process['process_start_tick']
                    or live['state']=='Z'):raise ValueError('The original transport handle stopped without normal exit evidence')
            if a.deadline_unix-time.time()<4*3600:raise TimeoutError('Insufficient remaining full393 start budget')
            time.sleep(20)
        prior=a.transport_stage/'run';result=json.loads((prior/'result.json').read_text())
        accepted_transport=result['status']=='unchanged_model_exact_block_transport_native_timing_and_original_states_accepted'
        deferred_transport=result['status']=='transport_candidate_deferred_after_nonpositive_native_callback'
        if (not (accepted_transport or deferred_transport) or result['DSP_calls']!=(384 if accepted_transport else 80)
                or result['model_inference_acceptance'] is not accepted_transport
                or result['baseline_closure']!=str(a.baseline_closure)):
            raise ValueError('Actual accepted384 or explicit deferred80 transport required')
        transport_revision=subprocess.check_output(['git','-C',str(a.transport_source),'rev-parse','HEAD'],text=True).strip()
        finished_process(process_path,transport_revision)
        if subprocess.check_output(['git','-C',str(a.transport_source),'status','--porcelain'],text=True).strip():
            raise ValueError('Original transport source changed')
        sys.path[:0]=[str(a.transport_source/'tools/qwen3'),str(ROOT/'qwen3/runtime/arm')]
        from run_transport_iteration import baseline,CASES,positive_callback_gain
        baseline(a.baseline_closure,a.baseline_closure.parent/'controller-process.json',
                 sha(a.baseline_closure.parent/'controller-process.json'),a.original_state_suite)
        transport_proof=native_transport(prior,a.baseline_closure)
        if positive_callback_gain(transport_proof['measurements']) is not accepted_transport:
            raise ValueError('Transport acceptance selection differs from actual native gain')
        _,libraries=candidate_libraries(a.baseline_closure/'cold')
        _,old,_=accepted(a.baseline_closure/'warm16');_,new,_=accepted(prior/'warm16')
        if old['DSP_calls']!=16 or new['DSP_calls']!=16:raise ValueError('Actual warm16 required')
        compare_integer(old,new,libraries);validate_parameter_mode(new,'exact-bytes');validate_float_policy(new,True)
        if accepted_transport:
            states=json.loads((prior/'state304/result.json').read_text())
            if (states['status']!='all_integer_candidate_state_regressions_passed' or states['DSP_calls']!=304
                    or result['state_result']!=entry(prior/'state304/result.json')):
                raise ValueError('All original304 state calls and exact final proof required')
            for case,count in CASES.items():
                _,old,_=accepted(a.original_state_suite/case);_,new,_=accepted(prior/'state304'/case)
                if old['DSP_calls']!=count or new['DSP_calls']!=count:raise ValueError('Original state count changed')
                compare_integer(old,new,libraries);validate_parameter_mode(new,'exact-bytes');validate_float_policy(new,True)
        record['transport_independent_preflight']=dict(result=entry(prior/'result.json'),
            process=entry(process_path),native=transport_proof,original_states304_rechecked=accepted_transport)
        warm=prior/'warm16' if accepted_transport else a.baseline_closure/'warm16'
        warm_commands=json.loads((warm/'commands.json').read_text());runtime=warm_commands['runtime_override']
        catalog=Path(runtime['path']);board_catalog=Path(runtime['board_path'])
        if sha(catalog)!=runtime['sha256']:raise ValueError('Selected actual runtime catalog changed')
        owner=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],
            'pgrep -x buffer_stream'],capture_output=True,timeout=15)
        if owner.returncode!=1:raise ValueError('Prior DSP owner remains or owner inspection failed')
        pinned()
        record['host_cache_replay_before_DSP']=replay_host(host,recipe,a.baseline_closure/'cold')
        command=[config['python'],str(ROOT/'tools/qwen3/run_helper_effects_iteration.py'),
            '--config',str(a.config),'--recipe-build',str(a.recipe_build),'--early-model-revisit',str(a.early_model_revisit),
            '--early-model-revisit-sha256',a.early_model_revisit_sha256,'--baseline-cold',str(a.baseline_closure/'cold'),
            '--baseline-warm',str(warm),'--original-state-suite',str(a.original_state_suite),
            '--runtime-catalog',str(catalog),'--board-runtime-catalog',str(board_catalog),
            '--output',str(a.output/'run'),'--name',a.name,'--deadline-unix',str(a.deadline_unix),
            '--candidate-kind','q1-runtime-combined','--prior-transport-closure',str(prior),
            '--prior-transport-process',str(process_path),'--prior-transport-process-sha256',sha(process_path)]
        write(a.output/'controller-command.json',command)
        with (a.output/'controller.log').open('xb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            terminal=dict(status='running',controller_pid=child.pid,process_start_tick=identity(child.pid)['start_tick'],
                source_revision=revision,start_unix=time.time());write(a.output/'controller-process.json',terminal)
            record.update(status='running_original_full393',controller_pid=child.pid);save()
            returncode=child.wait();terminal.update(status='complete',returncode=returncode,completed_unix=time.time())
            write(a.output/'controller-process.json',terminal)
            if returncode:raise ValueError('Actual original full393 controller failed')
        record.update(status='original_full393_controller_exited_normally_REQUIRES_independent_final_revisit',completed_unix=time.time());save()
    except BaseException as e:record.update(status='failed',error=str(e),completed_unix=time.time());save();raise
    print(record['status'])


if __name__=='__main__':main()
