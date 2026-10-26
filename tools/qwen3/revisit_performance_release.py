"""Independently recheck terminal dynamic performance closures without DSP work.

Use the frozen numerical checkout's validators and cache classes. Recalculate
timings from raw native reports, not a controller's claimed speedup. Historical
compiler counts are kept separate from new preparation and object reuse.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

AUDIT_ROOT=Path(__file__).resolve().parents[2]
CASES={'tails':6,'eos':9,'old-long32':33,'prefill1024':33,'decode993':95,'code64':128}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def entry(path):return dict(path=str(Path(path).absolute()),sha256=sha(path))


def require_complete(result,status,calls):
    if (result['status']!=status or result['DSP_calls']!=calls
            or result['model_inference_acceptance'] is not True):
        raise ValueError('Complete original closure with numerical acceptance required')


def require_measurement(proof,raw,summary):
    if (proof['DSP_calls']!=16 or raw['DSP_calls']!=16
            or proof['inline_CPU_observer'] is not False
            or proof['full_input_output_model_token_FNV_equality'] is not True
            or raw['process_exit_code'] or raw['CPU_model_fallback']
            or raw['frontend_invocations'] or raw['machine_code_compilations']
            or proof['measurement']!=summary):
        raise ValueError('Actual native16 and statistics recalculated from the raw trace required')


def native_abba(closure):
    from analyze_performance import summarize
    from compare_performance import accepted
    from run_integer_regressions import candidate_libraries,compare_integer,validate_float_policy,validate_parameter_mode
    rows={};receipts=[]
    for i,kind in enumerate(('baseline','candidate','candidate','baseline')):
        folder=closure/('native-abba-'+str(i)+'-'+kind)
        proof=read(folder/'result.json');raw=read(folder/'native-result.json')
        command=read(closure/('native-abba-'+str(i)+'-command.json'))
        if command.count('--oracle')!=1:raise ValueError('Original unambiguous native oracle required')
        oracle=Path(command[command.index('--oracle')+1])
        _,original,_=accepted(oracle)
        _,libraries=candidate_libraries(Path(read(oracle/'commands.json')['accepted_run']))
        if (proof['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
                or proof['native_sha256']!=sha(folder/'native-result.json')
                or proof['oracle_native_sha256']!=sha(oracle/'native-result.json')
                or proof['oracle_acceptance_sha256']!=sha(oracle/'acceptance.json')
                or read(folder/'run-exit.json')!={'board_returncode':0,'worker_returncode':0}):
            raise ValueError('Actual normal native exit and original oracle identity required')
        actual=summarize(raw);require_measurement(proof,raw,actual)
        compare_integer(original,raw,libraries);validate_parameter_mode(raw,'exact-bytes');validate_float_policy(raw,True)
        for query,row in actual['query_groups'].items():
            med=row['median_seconds'];rows.setdefault(query,[]).append(dict(kind=kind,
                kernel_seconds=med['DSP_kernel_seconds'],callback_seconds=med['ARM_callback_seconds']))
        receipts.append(entry(folder/'result.json'))
    timing={}
    if set(rows)!={'1','13','14'}:raise ValueError('Original q1/q13/q14 coverage required')
    for query,values in rows.items():
        if len(values)!=4:raise ValueError('Complete original ABBA required for every query')
        timing[query]={}
        for metric in ('kernel_seconds','callback_seconds'):
            old=statistics.mean(r[metric] for r in values if r['kind']=='baseline')
            new=statistics.mean(r[metric] for r in values if r['kind']=='candidate')
            timing[query][metric]=dict(baseline=old,candidate=new,ratio=new/old)
    if timing!=read(closure/'result.json')['native_model_timing']:
        raise ValueError('Controller measurements differ from original raw native reports')
    return dict(receipts=receipts,raw_native_measurements=timing,DSP_calls=64)


def states(closure,original,libraries):
    from compare_performance import accepted
    from run_integer_regressions import compare_integer,validate_float_policy,validate_parameter_mode
    result=read(closure/'state304/result.json');plan=read(closure/'state304/plan.json')
    if (result['status']!='all_integer_candidate_state_regressions_passed' or result['DSP_calls']!=304
            or [(r['case'],r['expected_DSP_calls']) for r in plan['cases']]!=list(CASES.items())):
        raise ValueError('All six original states and counts required')
    proofs=[]
    for case,count in CASES.items():
        _,old,_=accepted(original/case);_,new,_=accepted(closure/'state304'/case)
        if old['DSP_calls']!=count or new['DSP_calls']!=count:raise ValueError('Original state count changed')
        compare_integer(old,new,libraries);validate_parameter_mode(new,'exact-bytes');validate_float_policy(new,True)
        proofs.append(entry(closure/'state304'/case/'acceptance.json'))
    if read(closure/'result.json')['state_result']!=entry(closure/'state304/result.json'):
        raise ValueError('Final state receipt changed')
    return dict(proofs=proofs,DSP_calls=304)


def cold_cache(cold,base):
    from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
    from reuse_accepted_compiler_objects import ReadOnlyCompilerCache
    from reuse_accepted_prepared_linalg import frontend_entry
    native=read(cold/'native-result.json');recipe=Path(base['recipe']['path']);rows=[]
    for capture in native['captures']:
        bounds=capture['contract']['variables']['query']
        frontend_path=cold/'frontend-cache'/(capture['frontend_key']+'.json')
        frontend=frontend_entry(frontend_path)
        cache=CapturedQwenMachineCache(cold/'machine-cache',recipe,sha(recipe),cold/'frontend-cache')
        cache.cache=ReadOnlyCompilerCache(cache.cache,capture['machine_key'])
        replay=cache.get_or_compile(frontend,runtime_contract=capture['contract']);folder=Path(replay['directory'])
        object_reuse=read(folder/'objects/reuse-result.json')
        linked=read(folder/'linked/result.json')
        if (not replay['cache_hit'] or replay['machine_code_compilations'] or replay['model_object_codegen_invocations']
                or object_reuse['status']!='authenticated_exact_compiler_stage_objects_reused_NOT_model_acceptance'
                or object_reuse['new_machine_code_compilations'] or object_reuse['DSP_calls']
                or object_reuse['query_bounds']!=bounds
                or object_reuse['original_complete_policy_key_and_all_retained_artifacts_rehashed'] is not True
                or sha(linked['library']['path'])!=capture['DSO_sha256']):
            raise ValueError('Actual cold cache and exact primary object reuse required')
        prep=folder/'prepared/reuse-result.json';prepared=None
        if prep.exists():
            receipt=read(prep)
            if (receipt['status']!='exact_accepted_prepared_Linalg_copied_NOT_pipeline_or_DSP_acceptance'
                    or receipt['query_bounds']!=bounds or receipt['current_frontend']!=entry(frontend_path)
                    or receipt['current_recipe']!=entry(recipe) or receipt['DSP_calls']
                    or receipt['new_prepare_optimizer_invocations'] or receipt['new_frontend_invocations']):
                raise ValueError('Actual complete preparation-stage reuse receipt required')
            prepared=dict(receipt=entry(prep),actual_reuse_seconds=receipt['reuse_seconds'],
                historical_preparation_seconds=receipt['historical_preparation_seconds'])
        else:
            miss=read(folder/'prepared-reuse-miss/reuse-result.json')
            if miss['status']!='prepared_stage_cache_miss':raise ValueError('Explicit legitimate original preparation fallback required')
        rows.append(dict(query_bounds=bounds,machine_key=capture['machine_key'],DSO=linked['library'],
            primary_object_reuse=entry(folder/'objects/reuse-result.json'),prepared_stage=prepared))
    if len(rows)!=2 or {tuple(r['query_bounds']) for r in rows}!={(1,1),(2,32)}:
        raise ValueError('Exactly both original cold query graphs required')
    return dict(rows=rows,new_primary_model_codegen_invocations=0,
        new_SDK_caller_link_pipeline_invocations=2,actual_fresh_frontend_invocations=2,
        both_prepared_stages_actually_reused=all(r['prepared_stage'] is not None for r in rows),DSP_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-closure','transport-closure','transport-source','q1-closure',
                 'numerical-source','original-state-suite','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-only',action='store_true',help='Recheck only completed transport native64; never claim full model acceptance')
    a=p.parse_args()
    if not a.output.is_absolute() or a.output.exists() or a.output.is_symlink():p.error('Fresh absolute audit output required')
    if subprocess.check_output(['git','-C',str(AUDIT_ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Committed clean audit source required')
    sys.path[:0]=[str(a.numerical_source/'tools/qwen3'),str(a.numerical_source/'qwen3/runtime/arm')]
    from run_regression import require_configured_python
    from supervise_q1_runtime_closure import finished_process
    from run_integer_regressions import candidate_libraries,compare_integer,validate_float_policy,validate_parameter_mode
    from compare_performance import accepted
    from run_helper_effects_iteration import positive_native_gain
    require_configured_python(read(a.config))
    # Transport-only helpers live in their separately frozen checkout. Keep
    # numerical validators first so recipe ownership remains the actual source.
    sys.path.append(str(a.transport_source/'tools/qwen3'))
    spec=importlib.util.spec_from_file_location('release_transport_validator',a.transport_source/'tools/qwen3/run_transport_iteration.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    if a.native_only:
        proof=native_abba(a.transport_closure)
        result=dict(status='original_native_ABBA64_recalculated_without_new_DSP_or_full_closure_acceptance',
            audit_source_sha256=sha(__file__),revisited_unix=time.time(),native=proof,
            model_inference_acceptance=False,new_DSP_calls=0)
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status']);return
    transport.baseline(a.baseline_closure,a.baseline_closure.parent/'controller-process.json',
        sha(a.baseline_closure.parent/'controller-process.json'),a.original_state_suite)
    base_native=native_abba(a.baseline_closure)
    rt=read(a.transport_closure/'result.json')
    require_complete(rt,'unchanged_model_exact_block_transport_native_timing_and_original_states_accepted',384)
    rt_revision=subprocess.check_output(['git','-C',str(a.transport_source),'rev-parse','HEAD'],text=True).strip()
    finished_process(a.transport_closure.parent/'controller-process.json',rt_revision)
    if subprocess.check_output(['git','-C',str(a.transport_source),'status','--porcelain'],text=True).strip():
        raise ValueError('Frozen transport source changed')
    _,old_libraries=candidate_libraries(a.baseline_closure/'cold')
    _,old_warm,_=accepted(a.baseline_closure/'warm16');_,rt_warm,_=accepted(a.transport_closure/'warm16')
    compare_integer(old_warm,rt_warm,old_libraries)
    rt_native=native_abba(a.transport_closure)
    if not transport.positive_callback_gain(rt_native['raw_native_measurements']):raise ValueError('Positive original transport gain required')
    rt_states=states(a.transport_closure,a.original_state_suite,old_libraries)
    q1=read(a.q1_closure/'result.json')
    require_complete(q1,'exact_query1_integer_tiles_fixed512_native_timing_and_original_states_accepted',393)
    q1_process=finished_process(a.q1_closure.parent/'controller-process.json',q1['source_revision'])
    base,libraries=candidate_libraries(a.q1_closure/'cold')
    if Path(base['source_root'])!=a.numerical_source or base['revision']!=q1['source_revision']:
        raise ValueError('Actual frozen numerical source and controller must agree')
    cold=read(a.q1_closure/'cold/native-result.json');original=read(a.baseline_closure/'cold/native-result.json')
    if cold['DSP_calls']!=9:raise ValueError('Original fresh cold9 required')
    sdk=read(a.q1_closure/'cold/SDK-verification.json')
    lock=a.numerical_source/'tools/qwen3/dependencies.lock.json'
    if sdk!=dict(passed=True,files=len(read(lock)['files']),lock_sha256=sha(lock)):
        raise ValueError('Complete original pinned SDK verification required')
    framework=read(a.q1_closure/'cold/board-framework-verification.json')
    if framework['passed'] is not True or framework['actual']!=read(a.numerical_source/'tools/qwen3/board-framework.lock.json'):
        raise ValueError('Original ARM PyTorch and framework source required')
    compare_integer(original,cold,libraries);validate_float_policy(cold,True)
    if libraries[(2,32)]!=old_libraries[(2,32)] or libraries[(1,1)]==old_libraries[(1,1)]:
        raise ValueError('Changed single-token runtime and unchanged multi-token DSO required')
    early_path=Path(q1['early_model_revisit']['path']);early=read(early_path)
    from revisit_q1_model_runtime import revisit
    if (entry(early_path)!=q1['early_model_revisit']
            or revisit(Path(early['run']['path']),Path(early['plan']['path']),a.baseline_closure/'cold')!=early
            or not early['positive_early_model_gain']
            or libraries[(1,1)]!=read(early['build']['path'])['library']['sha256']):
        raise ValueError('The same original early-tested q1 runtime DSO required')
    _,warm,_=accepted(a.q1_closure/'warm16')
    if warm['DSP_calls']!=16 or rt_warm['DSP_calls']!=16:raise ValueError('Original warm16 required')
    compare_integer(rt_warm,warm,libraries);validate_parameter_mode(warm,'exact-bytes');validate_float_policy(warm,True)
    catalog=read(a.transport_closure/'warm16/commands.json')['runtime_override']
    if (read(a.q1_closure/'warm16/commands.json')['runtime_override']!=catalog
            or sha(catalog['path'])!=catalog['sha256']):
        raise ValueError('The actually accepted transport catalog must be preserved')
    q1_native=native_abba(a.q1_closure)
    if not positive_native_gain(q1_native['raw_native_measurements']):raise ValueError('Positive original q1 native gain required')
    q1_states=states(a.q1_closure,a.original_state_suite,libraries)
    stage=cold_cache(a.q1_closure/'cold',base)
    result=dict(status='actual_dynamic_transport384_q1_full393_and_compiler_stages_independently_revisited',
        revisited_unix=time.time(),audit_source_revision=subprocess.check_output(['git','-C',str(AUDIT_ROOT),'rev-parse','HEAD'],text=True).strip(),
        audit_source_sha256=sha(__file__),model_inference_acceptance=True,new_DSP_calls=0,
        historical_DSP_calls_rechecked=1170,baseline393=dict(result=entry(a.baseline_closure/'result.json'),native=base_native),
        transport384=dict(result=entry(a.transport_closure/'result.json'),native=rt_native,states=rt_states),
        q1_full393=dict(result=entry(a.q1_closure/'result.json'),process=q1_process,native=q1_native,states=q1_states),
        compiler_stages=stage,whole_cold_startup_speedup_claimed=False,production_TUI_speedup_claimed=False,
        preserved_scope='Original dynamic query1/query2..32, physicalKV1024, complete packets, tokens, termination, parameters, float owners, and native no-CPU-observer timing')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])


if __name__=='__main__':main()
