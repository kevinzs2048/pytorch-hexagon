"""Run the original full393 suite for the distinct dynamic PV candidate.

Fresh admission requires four hours. A separately verified same-source cold9
and warm16 may resume the remaining368 with two and a half hours. Complete
acceptance still requires an independent full closure audit.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from build import ROOT,entry,sha,write
from ordered_pv_release_acceptance import require_pv393
from revisit_release_index import indexed_files
from revisit_native_compile_session_abba import require_normal_owner


def require_admission(deadline,now):
    if deadline-now<14400:raise ValueError('Original dynamic PV393 requires four hours before starting')


def require_resume_admission(deadline,now):
    if deadline-now<9000:raise ValueError('Remaining native64 and state304 require two and a half hours before starting')


def same_reference(row,path):
    return (set(row) in ({'path','sha256'},{'path','sha256','bytes'})
        and {k:row[k] for k in ('path','sha256')}==entry(path)
        and ('bytes' not in row or type(row['bytes']) is int and row['bytes']==Path(path).stat().st_size))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','numerical-source','recipe-build','preflight','preflight-controller','baseline-closure',
                 'baseline-independent-revisit','original-state-suite','runtime-catalog','board-runtime-catalog','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--deadline-unix',type=int,required=True)
    for name in ('verified-cold','verified-warm','generation-review'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args();resuming=bool(a.generation_review)
    if len([x for x in (a.verified_cold,a.verified_warm,a.generation_review) if x]) not in (0,3):
        raise ValueError('Resume requires both complete original generation stages and their independent25 review')
    (require_resume_admission if resuming else require_admission)(a.deadline_unix,time.time());read=lambda path:json.loads(Path(path).read_text())
    git=lambda root,*args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if (a.output.exists() or not a.output.is_absolute() or not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.name)
            or git(ROOT,'status','--porcelain') or git(a.numerical_source,'status','--porcelain')):
        raise ValueError('Fresh named full closure and clean committed sources required')
    revision=git(a.numerical_source,'rev-parse','HEAD');maintenance=git(ROOT,'rev-parse','HEAD')
    prior=read(a.baseline_closure/'result.json');require_pv393(prior,read(a.baseline_independent_revisit))
    build=read(a.recipe_build/'result.json');preflight=read(a.preflight);refs=[];indexed_files(build,refs);indexed_files(preflight,refs)
    preowner=read(a.preflight_controller);require_normal_owner(a.preflight_controller,preowner['authorized_deadline_unix'])
    if (preflight['status']!='dynamic_PV_actual_public_two_graph_reuse_exact_tested_DSOs_NOT_DSP_or_native393_acceptance'
            or preflight['source_revision']!=revision or preowner['source_root']!=str(a.numerical_source)
            or preowner['source_revision']!=revision or preflight['recipe']!=build['recipe']
            or preflight['new_model_object_codegen_invocations'] or preflight['DSP_calls']):
        raise ValueError('Own normal public two-query exact-reuse candidate preflight required')
    sys.path[:0]=[str(a.numerical_source/'tools/qwen3'),str(a.numerical_source/'qwen3/runtime/arm'),str(a.numerical_source)]
    from build_dynamic_pv_native_recipe import validate_transition
    from dynamic_pv_native_contract import require_compiler
    from run_integer_regressions import candidate_libraries,compare_integer,validate_parameter_mode,validate_float_policy
    from compare_performance import accepted
    from bounded_loader_policy import configured_policy,verify_native_layout
    from run_helper_effects_iteration import require_common_runtime,positive_native_gain
    cold0=a.baseline_closure/'cold';warm0=a.baseline_closure/'warm16';base,original_libraries=candidate_libraries(cold0)
    recipe=a.recipe_build/'machine-recipe.json';recipe_sha=sha(recipe);validate_transition(read(base['recipe']['path']),read(recipe))
    model=read(build['tested_model']['path']);sdk=read(model['sdk']['path']);code=read(model['code']['path'])
    require_compiler(sdk,Path(code['compiler_diagnostics']['path']).read_text())
    if (not same_reference(model['original_model']['acceptance'],a.baseline_closure/'result.json')
            or not same_reference(model['original_model']['numerical_audit'],a.baseline_independent_revisit)):
        raise ValueError('Actually tested candidate must derive from this exact accepted PV393')
    expected={(1,1):original_libraries[(1,1)],(2,32):model['library']['sha256']}
    if {tuple(r['query_bounds']):r['DSO']['sha256'] for r in preflight['rows']}!=expected:
        raise ValueError('Public preflight must reproduce both exact tested model libraries')
    if resuming:
        proof=read(a.generation_review);indexed_files(proof,refs)
        if (proof['status']!='dynamic_PV_original_native_generation_full_packets_and_normal_owners_independently_revisited_NOT_full393_acceptance'
                or proof['DSP_calls']!=25 or proof['new_DSP_calls'] or proof['model_inference_acceptance'] or proof['performance_acceptance']
                or proof['source_root']!=str(a.numerical_source) or proof['source_revision']!=revision
                or proof['recipe_build']!=entry(a.recipe_build/'result.json') or len(proof['sessions'])!=2
                or proof['candidate_libraries']!={str(k):v for k,v in expected.items()}
                or [Path(r['native']['path']).parent for r in proof['sessions']]!=[a.verified_cold,a.verified_warm]
                or [Path(r['baseline']['path']).parent for r in proof['sessions']]!=[cold0,warm0]):
            raise ValueError('Resume may only reuse actual same-source cold9/warm16 with all packets independently reviewed')
        for row in proof['sessions']:
            owner=read(row['controller']['path']);require_normal_owner(row['controller']['path'],owner['authorized_deadline_unix'])
    config=read(a.config);policy=configured_policy(config)
    if not policy:raise ValueError('Original common bounded constant loader required')
    require_common_runtime(read(warm0/'commands.json'),a.runtime_catalog,a.board_runtime_catalog)
    _,baseline,_=accepted(warm0);validate_parameter_mode(baseline,'exact-bytes');validate_float_policy(baseline,True)
    a.output.mkdir();record=dict(status='dynamic_PV_original393_admitted',candidate_kind='ordered-pv-register-q2to32',
        source_revision=maintenance,numerical_source_root=str(a.numerical_source),numerical_source_revision=revision,
        source=entry(__file__),recipe=entry(recipe),recipe_build=entry(a.recipe_build/'result.json'),preflight=entry(a.preflight),
        preflight_controller=entry(a.preflight_controller),baseline_closure=entry(a.baseline_closure/'result.json'),
        baseline_independent_revisit=entry(a.baseline_independent_revisit),original_state_suite=str(a.original_state_suite),
        fresh_original_full393_minimum_remaining_seconds=14400,actual_start_minimum_remaining_seconds=9000 if resuming else 14400,
        continuation_reuses_independently_verified25=resuming,original_numerical_dependencies_unchanged=True,
        new_primary_model_codegen_invocations=0,model_inference_acceptance=False,performance_acceptance=False,
        deadline_unix=a.deadline_unix,steps=[],common_payload_preserving_loader_policy=policy,
        common_native_runtime_override=dict(path=str(a.runtime_catalog),board_path=str(a.board_runtime_catalog),sha256=sha(a.runtime_catalog)))
    def save():write(a.output/'result.json',record)
    def run(stage,command):
        if (git(ROOT,'rev-parse','HEAD')!=maintenance or git(a.numerical_source,'rev-parse','HEAD')!=revision
                or git(ROOT,'status','--porcelain') or git(a.numerical_source,'status','--porcelain')
                or sha(recipe)!=recipe_sha or time.time()>=a.deadline_unix):raise ValueError('Pinned sources, recipe or budget changed')
        owner=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],'pgrep -x buffer_stream'],capture_output=True,timeout=15)
        if owner.returncode!=1:raise ValueError('DSP must be free before each original stage')
        record['status']='running_'+stage;save();write(a.output/(stage+'-command.json'),command)
        with (a.output/(stage+'.log')).open('xb') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        record['steps'].append(dict(stage=stage,completed_unix=time.time()));save()
    save();python=config['python'];tools=a.numerical_source/'tools/qwen3';cold=a.verified_cold if resuming else a.output/'cold'
    try:
        if not resuming:
            run('cold',[python,str(tools/'run_native.py'),'--config',str(a.config),'--build',str(a.recipe_build),'--output',str(cold),'--name',a.name+'-cold','--specialize-floats'])
        _,libraries=candidate_libraries(cold);actual=read(cold/'native-result.json')
        if libraries!=expected or actual['DSP_calls']!=9:raise ValueError('Own cold9 must bind unchanged q1 and actual dynamic PV candidate')
        compare_integer(read(cold0/'native-result.json'),actual,libraries);validate_float_policy(actual,True);fingerprint=verify_native_layout(actual)
        record['candidate_libraries']={str(k):v for k,v in libraries.items()};warm=a.verified_warm if resuming else a.output/'warm16'
        record.update(cold_run=str(cold),warm_run=str(warm))
        if resuming:record['reused_generation_review']=entry(a.generation_review)
        save()
        if not resuming:run('warm16',[python,str(tools/'run_online_regression.py'),'--config',str(a.config),'--accepted-run',str(cold),'--output',str(warm),'--name',a.name+'-warm16',
            '--observations',config['observations'],'--board-observations',config['board']['observations'],'--prompts-json',json.dumps([
                'The capital of France is Paris. The capital of Germany is Berlin.','Complete this Python function:\ndef add(a, b):\n    return']),
            '--max-new-tokens','8','--chunk-size','32','--runtime-catalog',str(a.runtime_catalog),'--board-runtime-catalog',str(a.board_runtime_catalog),
            '--generation-timing','--parameter-check','exact-bytes','--specialize-floats'])
        _,actual,_=accepted(warm)
        if actual['DSP_calls']!=16 or verify_native_layout(actual)!=fingerprint:raise ValueError('Original warm16 and exact common loader required')
        compare_integer(baseline,actual,libraries);validate_parameter_mode(actual,'exact-bytes');validate_float_policy(actual,True)
        rows=[];groups={}
        for order,kind in enumerate(('baseline','candidate','candidate','baseline')):
            folder=a.output/('native-abba-'+str(order)+'-'+kind);oracle=warm0 if kind=='baseline' else warm
            run('native-abba-'+str(order),[python,str(tools/'benchmark_accepted_generation.py'),'--config',str(a.config),'--oracle',str(oracle),'--output',str(folder),'--name',a.name+'-native-abba-'+str(order)+'-'+kind])
            proof=read(folder/'result.json');native=read(folder/'native-result.json')
            if (proof['DSP_calls']!=16 or proof['inline_CPU_observer'] is not False or proof.get('timings_eligible_for_performance_acceptance') is not True
                    or proof['full_input_output_model_token_FNV_equality'] is not True or verify_native_layout(native)!=fingerprint):
                raise ValueError('Actual unwrapped native16 and full original packets required')
            compare_integer(baseline,native,libraries if kind=='candidate' else original_libraries);validate_parameter_mode(native,'exact-bytes');validate_float_policy(native,True)
            for query,measure in proof['measurement']['query_groups'].items():
                med=measure['median_seconds'];groups.setdefault(query,[]).append(dict(kind=kind,kernel_seconds=med['DSP_kernel_seconds'],callback_seconds=med['ARM_callback_seconds']))
            rows.append(dict(order=order,kind=kind,proof=entry(folder/'result.json')))
        timing={}
        for query,measured in groups.items():
            if len(measured)!=4:raise ValueError('Identical native query coverage in all four timing arms required')
            timing[query]={}
            for metric in ('kernel_seconds','callback_seconds'):
                old=statistics.mean(r[metric] for r in measured if r['kind']=='baseline');new=statistics.mean(r[metric] for r in measured if r['kind']=='candidate')
                timing[query][metric]=dict(baseline=old,candidate=new,ratio=new/old)
        record['native_model_timing']=timing;write(a.output/'native-ABBA-result.json',dict(status='complete_selected_helper_native_model_ABBA',rows=rows,measurements=timing,DSP_calls=64,model_inference_acceptance=False));save()
        if not positive_native_gain(timing,True):
            record.update(status='dynamic_PV_deferred_after_nonpositive_original_native89',DSP_calls=89,candidate_DSP_calls=57,paired_baseline_DSP_calls=32,original_state304_executed=False,completed_unix=time.time());return
        states=a.output/'state304';run('state304',[python,str(tools/'run_integer_regressions.py'),'--baseline-suite',str(a.original_state_suite),'--accepted-run',str(cold),
            '--runtime-catalog',str(a.runtime_catalog),'--board-runtime-catalog',str(a.board_runtime_catalog),'--output',str(states),'--name',a.name+'-state',
            '--deadline-unix',str(a.deadline_unix),'--parameter-check','exact-bytes','--specialize-floats'])
        state=read(states/'result.json')
        if state['status']!='all_integer_candidate_state_regressions_passed' or state['DSP_calls']!=304:raise ValueError('All six original states and304 actual calls required')
        record.update(status='dynamic_PV_original393_complete_NOT_independently_accepted',DSP_calls=393,candidate_DSP_calls=361,paired_baseline_DSP_calls=32,
            original_state304_executed=True,state_result={k:entry(states/'result.json')[k] for k in ('path','sha256')},
            newly_executed_DSP_calls=368 if resuming else 393,reused_independently_verified_DSP_calls=25 if resuming else 0,completed_unix=time.time())
    except BaseException as error:record.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
