"""Independently recheck a complete original GQA copy closure without DSP work."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def require_complete_closure(result,kind):
    status={'gqa-value-view-q1':'query1_GQA_value_view_native_timing_and_original_states_accepted',
            'gqa-value-forward-q1':'query1_GQA_value_forward_native_timing_and_original_states_accepted',
            'gqa-key-compact-q1':'query1_GQA_key_compact_native_timing_and_original_states_accepted',
            'gqa-dead-repeat-q1':'query1_GQA_dead_repeat_native_timing_and_original_states_accepted',
            'gqa-key-tile-q1':'query1_GQA_key_tile_native_timing_and_original_states_accepted',
            'gqa-key-scalar-forward-q1':'query1_GQA_key_scalar_forward_native_timing_and_original_states_accepted',
            'empty-tensor-cse-q2to32':'query2to32_empty_tensor_CSE_native_timing_and_original_states_accepted',
            'dynamic-qk-invariant-vector-q2to32':'query2to32_dynamic_QK_invariant_vector_native_timing_and_original_states_accepted'}
    if (kind not in status or result.get('status')!=status[kind] or result.get('candidate_kind')!=kind or
            any(type(result.get(k)) is not int or result[k]!=n for k,n in
                (('DSP_calls',393),('candidate_DSP_calls',361),('paired_baseline_DSP_calls',32))) or
            result.get('model_inference_acceptance') is not True):
        raise ValueError('Actual complete native393 and matching selected copy contract required')


def require_native_precheck(result,kind):
    positive='query2to32_dynamic_QK_invariant_vector_original_native89_precheck_completed_NOT_state_acceptance'
    negative='helper_effect_candidate_deferred_after_nonpositive_native_model_timing'
    if (kind!='dynamic-qk-invariant-vector-q2to32' or result.get('candidate_kind')!=kind
            or result.get('status') not in (positive,negative) or result.get('native_precheck_only') is not True
            or result.get('model_inference_acceptance') is not False
            or any(type(result.get(k)) is not int or result[k]!=n for k,n in
                (('DSP_calls',89),('candidate_DSP_calls',57),('paired_baseline_DSP_calls',32)))
            or (result['status']==positive and (result.get('positive_native_precheck') is not True
                or result.get('original_state304_executed') is not False))):
        raise ValueError('Distinct actual native89 precheck must remain unaccepted and cannot borrow state evidence')


def require_cold_exit(record):
    # run_native deliberately leaves numerical acceptance to the independent
    # packet audit; its exit receipt therefore carries an explicit false flag.
    keys=set(record)
    if (keys not in ({'board_returncode','worker_returncode'},
                     {'board_returncode','worker_returncode','numerical_acceptance'})
            or any(type(record[k]) is not int or record[k]!=0
                   for k in ('board_returncode','worker_returncode'))
            or ('numerical_acceptance' in record and record['numerical_acceptance'] is not False)):
        raise ValueError('Normal actual cold exit required')


def raw_invocations(events,forwards):
    actual=[event for event in events if event.get('stage')=='invoke']
    expected=[row['invoke'] for row in forwards]
    if (actual!=expected or not events or events[-1].get('stage')!='close' or
            any(event.get(k,0) for event in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
        raise ValueError('Complete original raw invokes and normal close required')


def owner_gone(path):
    for pid in read(path).values():
        if type(pid) is not int or Path('/proc',str(pid)).exists():
            raise ValueError('An actual inference owner remains active')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','numerical-source','closure','controller-process','baseline-cold',
                 'baseline-warm','original-state-suite','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate-kind',choices=('gqa-key-tile-q1','gqa-key-scalar-forward-q1','empty-tensor-cse-q2to32','dynamic-qk-invariant-vector-q2to32'),default='gqa-key-tile-q1')
    p.add_argument('--native-precheck-only',action='store_true')
    a=p.parse_args()
    if not a.output.is_absolute() or a.output.exists():p.error('Fresh absolute independent audit required')
    audit_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed independent auditor required')
    sys.path[:0]=[str(a.numerical_source/'tools/qwen3'),str(a.numerical_source/'qwen3/runtime/arm')]
    from build import entry,sha
    from run_regression import require_configured_python
    from run_integer_regressions import candidate_libraries,compare_integer,validate_float_policy,validate_parameter_mode
    from compare_performance import accepted
    from revisit_performance_release import native_abba,states
    from revisit_key_tile_model import revisit
    dynamic_qk=a.candidate_kind=='dynamic-qk-invariant-vector-q2to32'
    dynamic_cse=a.candidate_kind in ('empty-tensor-cse-q2to32','dynamic-qk-invariant-vector-q2to32')
    scalar_forward=a.candidate_kind=='gqa-key-scalar-forward-q1'
    from run_helper_effects_iteration import positive_native_gain, require_common_runtime
    from reuse_accepted_prepared_linalg import frontend_entry
    from reuse_accepted_compiler_objects import ReadOnlyCompilerCache
    from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
    config=read(a.config);require_configured_python(config)
    result=read(a.closure/'result.json');process=read(a.controller_process)
    if a.native_precheck_only:require_native_precheck(result,a.candidate_kind)
    else:require_complete_closure(result,a.candidate_kind)
    dead=True
    compact=True
    forward=True
    revision=subprocess.check_output(['git','-C',str(a.numerical_source),'rev-parse','HEAD'],text=True).strip()
    if (result['source_revision']!=revision
            or process['status']!='completed' or process['returncode']!=0 or process['source_revision']!=revision
            or subprocess.check_output(['git','-C',str(a.numerical_source),'status','--porcelain'],text=True).strip()):
        raise ValueError('Actual terminal complete native393 and unchanged numerical source required')
    for key in ('controller_pid','child_pid'):
        if type(process[key]) is not int or Path('/proc',str(process[key])).exists():
            raise ValueError('Completed controller owners must be gone')
    base,libraries=candidate_libraries(a.closure/'cold')
    old_base,old_libraries=candidate_libraries(a.baseline_cold)
    cold=read(a.closure/'cold/native-result.json')
    if (Path(base['source_root'])!=a.numerical_source or base['revision']!=revision or cold['DSP_calls']!=9
            or (libraries[(1,1)]!=old_libraries[(1,1)] or libraries[(2,32)]==old_libraries[(2,32)] if dynamic_cse else
                libraries[(2,32)]!=old_libraries[(2,32)] or libraries[(1,1)]==old_libraries[(1,1)])
            or result['candidate_libraries']!={str(k):v for k,v in libraries.items()}):
        raise ValueError('Exactly changed q1 and unchanged original q2..32 required')
    compare_integer(read(a.baseline_cold/'native-result.json'),cold,libraries);validate_float_policy(cold,True)
    require_cold_exit(read(a.closure/'cold/run-exit.json'))
    sdk_lock=a.numerical_source/'tools/qwen3/dependencies.lock.json'
    if read(a.closure/'cold/SDK-verification.json')!=dict(passed=True,files=len(read(sdk_lock)['files']),lock_sha256=sha(sdk_lock)):
        raise ValueError('Complete SDK pin verification changed')
    framework=read(a.closure/'cold/board-framework-verification.json')
    if framework['passed'] is not True or framework['actual']!=read(a.numerical_source/'tools/qwen3/board-framework.lock.json'):
        raise ValueError('Original ARM framework pin verification changed')
    early_path=Path(result['early_model_revisit']['path']);early=read(early_path)
    if (entry(early_path)!=result['early_model_revisit'] or
            early['candidate_kind']!=a.candidate_kind or
            revisit(Path(early['run']['path']),Path(early['plan']['path']),Path(early['accepted_cold']['path']).parent)!=early
            or libraries[(2,32) if dynamic_cse else (1,1)]!=read(early['build']['path'])['library']['sha256']):
        raise ValueError('The same original ARM bitexact early-tested model required')
    recipe_path=Path(base['recipe']['path']);recipe=read(recipe_path)
    if (recipe.get('gqa_value_view_query1') is not True or recipe.get('share_immutable_prepared_artifacts') is not True
            or recipe.get('gqa_value_view_contract')!='original_Qwen_query1_identity_V_shape_view_v1'):
        raise ValueError('Explicit original identity graph and immutable preparation contract required')
    if forward and (recipe.get('gqa_value_forward_query1') is not True or
            recipe.get('gqa_value_forward_contract')!='original_Qwen_query1_component_slices_through_proved_V_repeat_v1'):
        raise ValueError('Explicit proved original V repeat-slice contract required')
    if compact and (recipe.get('gqa_key_compact_query1') is not True or
            recipe.get('gqa_key_compact_contract')!='original_Qwen_query1_eight_head_contiguous_K_SCF_copy_v1'):
        raise ValueError('Explicit proved original SCF K copy contract required')
    if dead and (recipe.get('gqa_dead_repeat_query1') is not True or
            recipe.get('gqa_dead_repeat_contract')!='original_Qwen_query1_proved_closed_unused_V_repeat_after_CSE_v1'):
        raise ValueError('Explicit late closed-unused V copy contract required')
    if scalar_forward:
        from key_forward_model_contract import CONTRACT, validate_key_forward as validate_key_copy
        contract_flag='gqa_key_scalar_forward_query1';contract_key='gqa_key_scalar_forward_contract'
        contract_input='query1_gqa_key_scalar_forward_contract_source';contract_file='key_forward_model_contract.py'
        object_key='query1_gqa_key_scalar_forward'
    else:
        from key_tile_model_contract import CONTRACT, validate_key_tile as validate_key_copy
        contract_flag='gqa_key_tile_query1';contract_key='gqa_key_tile_contract'
        contract_input='query1_gqa_key_tile_contract_source';contract_file='key_tile_model_contract.py'
        object_key='query1_gqa_key_tile'
    from profile_model_memory import checked
    expected_contract_source=(read(old_base['recipe']['path'])['inputs'][contract_input] if dynamic_cse else
        entry(a.numerical_source/'tools/qwen3'/contract_file))
    if dynamic_cse and sha(a.numerical_source/'tools/qwen3'/contract_file)!=expected_contract_source['sha256']:
        raise ValueError('Unchanged original N32 contract validator bytes required for q2 CSE')
    if (recipe.get(contract_flag) is not True or recipe.get(contract_key) != CONTRACT
            or recipe['inputs'][contract_input] != expected_contract_source
            or (scalar_forward and (recipe.get('gqa_key_tile_query1') is not False
                or 'gqa_key_tile_contract' in recipe or 'query1_gqa_key_tile_contract_source' in recipe['inputs']))):
        raise ValueError('Distinct authenticated final K contract required')
    model_build=read(early['build']['path'])
    q1_sdk=recipe['inputs']['query1_gqa_dead_repeat_sdk_receipt'] if dynamic_cse else model_build['compiler_receipt']
    q1_code=recipe['inputs']['query1_gqa_dead_repeat_objects_receipt'] if dynamic_cse else model_build['objects_receipt']
    validate_key_copy(read(checked(q1_sdk)),checked(read(checked(q1_code))['compiler_diagnostics']).read_text())
    dynamic_prefix='query2_qk_invariant' if dynamic_qk else 'query2_cse'
    if dynamic_cse and (recipe.get('empty_tensor_cse_query2_to32') is not True
            or recipe.get('empty_tensor_cse_dynamic_contract')!='original_Qwen_query2to32_CSE_after_empty_tensor_v1'
            or recipe['inputs'][dynamic_prefix+'_sdk_receipt']!=model_build['compiler_receipt']
            or recipe['inputs'][dynamic_prefix+'_objects_receipt']!=model_build['objects_receipt']):
        raise ValueError('Distinct exact original dynamic CSE recipe required')
    if dynamic_qk:
        from dynamic_qk_invariant_contract import require_reference
        from build_dynamic_qk_invariant_recipe import CONTRACT as QK_CONTRACT,require_precision
        prior=model_build['accepted_q2_baseline393'];prior_audit=model_build['accepted_model_audit']
        if entry(prior['path'])!=prior or entry(prior_audit['path'])!=prior_audit:
            raise ValueError('Own accepted CSE baseline and independent audit changed')
        require_reference(read(prior['path']),read(prior_audit['path']),prior)
        precision=result['precision32_independent_revisit']
        if (entry(precision['path'])!=precision or recipe.get('dynamic_qk_invariant_query2_to32') is not True
                or recipe.get('dynamic_qk_invariant_contract')!=QK_CONTRACT
                or recipe['inputs']['query2_qk_invariant_contract_source']!=entry(a.numerical_source/'tools/qwen3/dynamic_qk_invariant_contract.py')):
            raise ValueError('Distinct QK recipe and actual independently closed precision32 required')
        checked_precision=read(precision['path']);ref=checked_precision['run']
        if entry(ref['path'])!=ref:raise ValueError('Actual precision32 run changed')
        require_precision(checked_precision,read(ref['path']),read(model_build['compiler_receipt']['path'])['candidate']['sha256'])
    for row in recipe['inputs'].values():
        if sha(row['path'])!=row['sha256']:raise ValueError('Pinned dependency bytes changed')
    compiler_stages=[]
    for capture in cold['captures']:
        folder=a.closure/'cold';frontend_path=folder/'frontend-cache'/(capture['frontend_key']+'.json')
        frontend=frontend_entry(frontend_path)
        cache=CapturedQwenMachineCache(folder/'machine-cache',recipe_path,sha(recipe_path),folder/'frontend-cache')
        cache.cache=ReadOnlyCompilerCache(cache.cache,capture['machine_key'])
        replay=cache.get_or_compile(frontend,runtime_contract=capture['contract']);machine=Path(replay['directory'])
        bounds=capture['contract']['variables']['query'];objects=read(machine/'objects/reuse-result.json')
        prepared=read(machine/'prepared/reuse-result.json');linked=read(machine/'linked/result.json')
        if (not replay['cache_hit'] or replay['machine_code_compilations'] or replay['model_object_codegen_invocations']
                or objects['status']!='authenticated_exact_compiler_stage_objects_reused_NOT_model_acceptance'
                or objects['new_machine_code_compilations'] or objects['DSP_calls'] or objects['query_bounds']!=bounds
                or objects['original_complete_policy_key_and_all_retained_artifacts_rehashed'] is not True
                or sha(linked['library']['path'])!=capture['DSO_sha256']
                or prepared['status']!='exact_accepted_prepared_Linalg_staged_NOT_pipeline_or_DSP_acceptance'
                or prepared['shared_immutable_artifacts'] is not True or prepared['query_bounds']!=bounds
                or prepared['current_frontend']!=entry(frontend_path) or prepared['current_recipe']!=entry(recipe_path)
                or any(prepared[key] for key in ('DSP_calls','new_prepare_optimizer_invocations','new_frontend_invocations'))):
            raise ValueError('Actual exact original prepared/object stages required')
        for row in prepared['files']:
            if sha(row['path'])!=row['sha256']:raise ValueError('Authenticated preparation source changed')
        for row in prepared['staged_artifacts']:
            source=Path(row['source']);target=Path(row['target'])
            # result.json is intentionally rewritten into independent metadata.
            if source.name not in ('result.json','reuse-result.json') and sha(target)!=row['sha256']:
                raise ValueError('Staged graph or numerical policy artifact changed')
            if row['mode']=='hardlink':
                if (source.name not in ('hexagon-linalg.mlirbc','adapted-source.mlir.gz') or
                        source.stat().st_ino!=target.stat().st_ino or source.stat().st_dev!=target.stat().st_dev):
                    raise ValueError('Immutable shared artifact identity changed')
            elif row['mode']!='copy' or source.stat().st_ino==target.stat().st_ino:
                raise ValueError('Mutable stage metadata must remain independently owned')
        prefix='query1_gqa_dead_repeat' if dead else 'query1_gqa_key_compact' if compact else 'query1_gqa_value_forward' if forward else 'query1_gqa_value_view'
        selected=prefix in objects
        if (selected!=(bounds==[1,1]) or (object_key in objects)!=(bounds==[1,1])
                or (scalar_forward and 'query1_gqa_key_tile' in objects)
                or (not scalar_forward and 'query1_gqa_key_scalar_forward' in objects) or any(other in objects for other in
                ('query1_gqa_dead_repeat','query1_gqa_key_compact','query1_gqa_value_forward','query1_gqa_value_view') if other!=prefix)):
            raise ValueError('Selected GQA object escaped its original query1 route')
        if selected:
            qualification=objects.get(object_key,{})
            if (qualification.get('contract') != CONTRACT
                    or qualification.get('all28_original_K_scalar_sources_proved' if scalar_forward else
                        'all28_complete_K_copy_tiles_proved') is not True
                    or (scalar_forward and qualification.get('complete_K_copy_producers_removed')!=28)):
                raise ValueError('Actual q1 reuse must carry its distinct final K qualification')
            from types import SimpleNamespace
            from reuse_accepted_compiler_objects import cse_producer
            producer=objects[prefix]
            for key in ('original_producer','compiler_receipt','objects_receipt'):
                if entry(producer[key]['path'])!= {k:producer[key][k] for k in ('path','sha256')}:
                    raise ValueError('Pinned original GQA compiler producer changed')
            sdk=producer['compiler_receipt'];code=producer['objects_receipt']
            cse_producer(SimpleNamespace(query_bounds=[1,1],gqa_key_tile_n32=not scalar_forward,
                gqa_key_scalar_forward=scalar_forward,
                gqa_key_tile_contract_source_sha256=sha(a.numerical_source/'tools/qwen3'/contract_file) if not scalar_forward else None,
                gqa_key_scalar_forward_contract_source_sha256=sha(a.numerical_source/'tools/qwen3'/contract_file) if scalar_forward else None,
                cse_sdk_receipt=Path(sdk['path']),
                cse_sdk_receipt_sha256=sdk['sha256'],cse_objects_receipt=Path(code['path']),
                cse_objects_receipt_sha256=code['sha256']),read(producer['original_producer']['path']),
                variant='gqa-dead-repeat' if dead else 'gqa-key-compact' if compact else 'gqa-value-forward' if forward else 'gqa-value-view')
        if dynamic_cse:
            from types import SimpleNamespace
            from reuse_accepted_compiler_objects import cse_producer
            dynamic_object='query2_QK_invariant_vector' if dynamic_qk else 'query2_cse'
            if (dynamic_object in objects)!=(bounds==[2,32]) or dynamic_qk and 'query2_cse' in objects:
                raise ValueError('Dynamic CSE object escaped query2..32')
            if bounds==[2,32]:
                producer=objects[dynamic_object]
                for key in ('original_producer','compiler_receipt','objects_receipt'):
                    if entry(producer[key]['path'])!={k:producer[key][k] for k in ('path','sha256')}:
                        raise ValueError('Pinned dynamic CSE producer changed')
                sdk=producer['compiler_receipt'];code=producer['objects_receipt']
                cse_producer(SimpleNamespace(query_bounds=[2,32],dynamic_cse=not dynamic_qk,dynamic_qk_invariant=dynamic_qk,
                    dynamic_qk_invariant_contract_source_sha256=sha(a.numerical_source/'tools/qwen3/dynamic_qk_invariant_contract.py') if dynamic_qk else None,
                    cse_sdk_receipt=Path(sdk['path']),cse_sdk_receipt_sha256=sdk['sha256'],
                    cse_objects_receipt=Path(code['path']),cse_objects_receipt_sha256=code['sha256']),
                    read(producer['original_producer']['path']))
        elif 'query2_cse' in objects:
            raise ValueError('Original unchanged multi-token route cannot borrow dynamic CSE evidence')
        compiler_stages.append(dict(query_bounds=bounds,objects=entry(machine/'objects/reuse-result.json'),
            prepared=entry(machine/'prepared/reuse-result.json'),DSO=linked['library'],
            actual_prepare_reuse_seconds=prepared['reuse_seconds']))
    runtime=result['common_native_runtime_override']
    require_common_runtime(read(a.baseline_warm/'commands.json'),Path(runtime['path']),Path(runtime['board_path']))
    require_common_runtime(read(a.closure/'warm16/commands.json'),Path(runtime['path']),Path(runtime['board_path']))
    _,old_warm,_=accepted(a.baseline_warm);acceptance,warm,_=accepted(a.closure/'warm16')
    if acceptance['numerical_revision']!=revision or warm['DSP_calls']!=16:raise ValueError('Original native warm16 required')
    compare_integer(old_warm,warm,libraries);validate_parameter_mode(warm,'exact-bytes');validate_float_policy(warm,True)
    native=native_abba(a.closure)
    positive=positive_native_gain(native['raw_native_measurements'],multi=dynamic_cse)
    if a.native_precheck_only:
        if positive!=(result['status']=='query2to32_dynamic_QK_invariant_vector_original_native89_precheck_completed_NOT_state_acceptance'):
            raise ValueError('Raw native timing does not support the actual precheck decision')
        state_proof=None
    else:
        if not positive:raise ValueError('Actual native benefit and multi-token nonregression required')
        state_proof=states(a.closure,a.original_state_suite,libraries)
    folders=[a.closure/'cold',a.closure/'warm16']+[a.closure/('native-abba-'+str(i)+'-'+kind)
        for i,kind in enumerate(('baseline','candidate','candidate','baseline'))]
    if not a.native_precheck_only:folders += [a.closure/'state304'/case for case in ('tails','eos','old-long32','prefill1024','decode993','code64')]
    raw_proofs=[]
    for folder in folders:
        owner_gone(folder/'active-processes.json');actual=read(folder/'native-result.json')
        command=read(folder/'commands.json')['board'] if (folder/'commands.json').exists() else read(folder/'result.json')['board']
        if folder!=a.closure/'cold' and (command[command.index('--catalog')+1]!=runtime['board_path'] or command[command.index('--catalog-sha256')+1]!=runtime['sha256']):
            raise ValueError('Every actual native timing and state arm must use the same runtime')
        output=Path(command[command.index('--output')+1]);observation=Path(config['observations'])/output.name
        raw=observation/'stream.stdout.txt';events=[json.loads(line) for line in raw.read_text().splitlines()]
        raw_invocations(events,actual['forwards']);raw_proofs.append(entry(raw))
    if (subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=audit_revision or
            subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()):
        raise ValueError('Independent audit source changed')
    proof=dict(status=('query2to32_dynamic_QK_invariant_vector_original_native89_raw_and_owners_independently_revisited_NOT_state_acceptance' if a.native_precheck_only else
                       'query2to32_dynamic_QK_invariant_vector_native393_independently_revisited' if dynamic_qk else
                       'query2to32_empty_tensor_CSE_native393_independently_revisited' if dynamic_cse else
                       'query1_GQA_key_scalar_forward_native393_independently_revisited' if scalar_forward else
                       'query1_GQA_key_tile_native393_independently_revisited' if dead else
                       'query1_GQA_key_compact_native393_independently_revisited' if compact else
                       'query1_GQA_value_forward_native393_independently_revisited' if forward else
                       'query1_GQA_value_view_native393_independently_revisited'),model_inference_acceptance=not a.native_precheck_only,
        numerical_source_revision=revision,audit_source_revision=audit_revision,audit_source_sha256=sha(__file__),
        closure=entry(a.closure/'result.json'),controller=entry(a.controller_process),DSP_calls=89 if a.native_precheck_only else 393,new_DSP_calls=0,
        candidate_DSP_calls=57 if a.native_precheck_only else 361,paired_baseline_DSP_calls=32,candidate_libraries={str(k):v for k,v in libraries.items()},
        native_precheck_only=a.native_precheck_only,positive_native_precheck=positive,original_full393_minimum_remaining_seconds=14400,
        early_original_ARM_gate=entry(early_path),native=native,states=state_proof,compiler_stages=compiler_stages,
        raw_invocation_traces=raw_proofs,actual_fresh_frontend_invocations=2,new_primary_model_codegen_invocations=0,
        numerical_or_prompt_specific_graph_changes=False,whole_cold_startup_speedup_claimed=False,
        production_TUI_speedup_claimed=False,completed_unix=time.time())
    a.output.write_text(json.dumps(proof,indent=2)+'\n');print(proof['status'],proof['DSP_calls'])


if __name__=='__main__':main()
