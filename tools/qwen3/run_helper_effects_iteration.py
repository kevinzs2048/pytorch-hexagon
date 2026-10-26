"""Close the guarded helper contract through actual capture, timing and states.

An early same-input gate precedes this controller. Reject nonpositive complete
native model timings before the expensive unchanged original state suite.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from build import ROOT,sha,write
from compare_performance import accepted
from run_integer_regressions import candidate_libraries,compare_integer,validate_parameter_mode,validate_float_policy


def gain(timing):
    for k in ('kernel_seconds','callback_seconds'):
        if any(not math.isfinite(timing[k][n]) or timing[k][n]<=0 for n in ('baseline','candidate')):
            raise ValueError('Positive finite actual model timing required')
    return all(timing[k]['candidate']/timing[k]['baseline']<0.995 for k in ('kernel_seconds','callback_seconds'))


def positive_native_gain(timing,multi=False):
    if not multi:
        return gain(timing['1']) and all(r['ratio']<=1.02 for q,t in timing.items() if q!='1' for r in t.values())
    if set(timing)!={'1','13','14'}:
        raise ValueError('Unchanged original native query1, query13 and query14 coverage required')
    q1=timing['1']
    for key in ('kernel_seconds','callback_seconds'):
        if any(not math.isfinite(q1[key][n]) or q1[key][n]<=0 for n in ('baseline','candidate')):
            raise ValueError('Finite positive unchanged query1 timing required')
    return (gain(timing['13']) and gain(timing['14'])
        and all(q1[k]['candidate']/q1[k]['baseline']<=1.02 for k in ('kernel_seconds','callback_seconds')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','recipe-build','early-model-revisit','baseline-cold','baseline-warm',
                 'original-state-suite','runtime-catalog','board-runtime-catalog','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--early-model-revisit-sha256',required=True)
    p.add_argument('--candidate-kind',choices=('silu-effects','guarded-copy','multi-integer','multi-silu','multi-fixed512','q1-runtime-combined','empty-tensor-cse-q1','gqa-value-view-q1','gqa-value-forward-q1'),default='silu-effects')
    p.add_argument('--prior-transport-closure',type=Path)
    p.add_argument('--prior-transport-process',type=Path)
    p.add_argument('--prior-transport-process-sha256')
    p.add_argument('--name',required=True);p.add_argument('--deadline-unix',type=int,required=True)
    a=p.parse_args()
    combined=a.candidate_kind=='q1-runtime-combined'
    cse=a.candidate_kind=='empty-tensor-cse-q1'
    forward=a.candidate_kind=='gqa-value-forward-q1'
    view=a.candidate_kind in ('gqa-value-view-q1','gqa-value-forward-q1')
    if (not a.output.is_absolute() or a.output.exists() or a.deadline_unix-time.time()<(4 if combined or view else 6)*3600
            or not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.name)):
        p.error('Fresh scoped output and sufficient authorized time for full closure required')
    if sha(a.early_model_revisit)!=a.early_model_revisit_sha256:raise ValueError('Early real model evidence changed')
    early=json.loads(a.early_model_revisit.read_text())
    if combined or cse or view:
        from revisit_q1_model_runtime import revisit
        early_cold=Path(early['accepted_cold']['path']).parent if view else a.baseline_cold
        if (early.get('candidate_kind')!=a.candidate_kind
                or revisit(Path(early['run']['path']),Path(early['plan']['path']),early_cold)!=early):
            raise ValueError('Actual scoped complete original q1 gate required')
    multi_candidate=a.candidate_kind in ('multi-integer','multi-silu','multi-fixed512')
    multi_silu=a.candidate_kind=='multi-silu'
    fixed512=a.candidate_kind=='multi-fixed512'
    early_status=('actual_q1_original_ARM_complete_packet_unwrapped_gqa-value-forward-q1_ABBA8_revisited' if forward else
                  'actual_q1_original_ARM_complete_packet_unwrapped_GQA_value_view_ABBA8_revisited' if view else
                  'actual_q1_original_ARM_complete_packet_unwrapped_CSE_ABBA8_revisited' if cse else
                  'actual_q1_original_ARM_complete_packet_unwrapped_runtime_ABBA8_revisited' if combined else
                  'same_input_full_dynamic_output_and_unwrapped_fixed512_copy_ABBA_revisited' if fixed512 else
                  'same_input_full_q1_output_and_unwrapped_guarded_copy_ABBA_revisited' if a.candidate_kind=='guarded-copy'
                  else ('same_input_full_dynamic_output_and_unwrapped_SiLU_effect_ABBA_revisited' if multi_silu else
                        'same_input_full_dynamic_output_and_unwrapped_integer_ABBA_revisited' if multi_candidate else
                        'same_input_full_q1_output_and_unwrapped_model_ABBA_revisited'))
    if (early['status']!=early_status
            or not early.get('full_original_logits_and_56_KV_byteexact',combined) or early['DSP_calls']<8
            or sha(early['run']['path'])!=early['run']['sha256']
            or any(early['measurements'][k]['ratio']>=0.995 for k in ('kernel_us','callback_seconds'))):
        raise ValueError('Positive actual complete-model early gate required')
    config=json.loads(a.config.read_text());board=config['board'];recipe=a.recipe_build/'machine-recipe.json'
    recipe_sha=sha(recipe);candidate_recipe=json.loads(recipe.read_text())
    base,baseline_libraries=candidate_libraries(a.baseline_cold)
    original_recipe=json.loads(Path(base['recipe']['path']).read_text())
    if combined:
        if (not a.prior_transport_closure or not a.prior_transport_process
                or sha(a.prior_transport_process)!=a.prior_transport_process_sha256):
            raise ValueError('Final unchanged-model transport phase must finish normally before q1 closure')
        prior=json.loads((a.prior_transport_closure/'result.json').read_text())
        process=json.loads(a.prior_transport_process.read_text())
        accepted_transport=prior['status']=='unchanged_model_exact_block_transport_native_timing_and_original_states_accepted'
        deferred_transport=prior['status']=='transport_candidate_deferred_after_nonpositive_native_callback'
        expected_warm=a.prior_transport_closure/'warm16' if accepted_transport else a.baseline_cold.parent/'warm16'
        baseline_final=json.loads((a.baseline_cold.parent/'result.json').read_text())
        warm_command=json.loads((expected_warm/'commands.json').read_text())
        if (not (accepted_transport or deferred_transport) or prior['DSP_calls']!=(384 if accepted_transport else 80)
                or prior['model_inference_acceptance'] is not accepted_transport
                or prior['baseline_closure']!=str(a.baseline_cold.parent)
                or prior['numerical_revision']!=base['revision'] or prior['model_DSOs_and_recipe_unchanged'] is not True
                or process.get('status') not in ('complete','completed') or process.get('returncode')!=0
                or process.get('source_revision')!=prior['source_revision']
                or type(process.get('controller_pid')) is not int or Path('/proc',str(process['controller_pid'])).exists()
                or a.baseline_warm!=expected_warm or warm_command['runtime_override']['sha256']!=sha(a.runtime_catalog)
                or baseline_final['DSP_calls']!=393 or baseline_final['model_inference_acceptance'] is not True):
            raise ValueError('Original final393 and completed transport selection with unchanged numerical source are required')
    copy_candidate=a.candidate_kind=='guarded-copy'
    required=({'fixed512_copy_runtime','fixed512_copy_source','fixed512_copy_nm'} if fixed512 else
              {'multi_silu_effects_optimizer'} if multi_silu else
              {'packed_multi_i8_optimizer','packed_multi_i8_runtime','packed_multi_i8_nm'} if multi_candidate else
              ({'guarded_copy_runtime','guarded_copy_source'} if copy_candidate else {'silu_effects_optimizer','silu_effects_guard'}))
    if combined:required={'integer_tiles_source','integer_tiles_header','integer_tiles_original_header'}
    if cse:required={'query1_cse_compiler','query1_cse_sdk_receipt','query1_cse_objects_receipt'}
    if view:required={'query1_gqa_value_view_compiler','query1_gqa_value_view_sdk_receipt','query1_gqa_value_view_objects_receipt'}
    if forward:required={'query1_gqa_value_forward_compiler','query1_gqa_value_forward_sdk_receipt','query1_gqa_value_forward_objects_receipt'}
    stage_reuse=candidate_recipe.get('reuse_exact_compiler_objects',False)
    if type(stage_reuse) is not bool or (stage_reuse and not (fixed512 or combined or cse or view)):
        raise ValueError('Exact compiler-stage reuse is scoped to the final guarded fixed512 candidate')
    prepared_reuse=candidate_recipe.get('reuse_exact_prepared_linalg',False)
    if type(prepared_reuse) is not bool or (prepared_reuse and not ((combined or cse or view) and stage_reuse)):
        raise ValueError('Exact preparation reuse requires the original combined q1 compiler-stage route')
    if prepared_reuse:
        required.add('prepared_stage_reuse_source')
        owned=ROOT/'tools/qwen3/reuse_accepted_prepared_linalg.py'
        if candidate_recipe['inputs']['prepared_stage_reuse_source']!={'path':str(owned),'sha256':sha(owned)}:
            raise ValueError('The owned exact preparation authenticator must match the current source')
    if stage_reuse:
        required|={'compiler_stage_reuse_source','compiler_stage_reuse_cold','compiler_stage_reuse_config'}
        reuse_inputs=candidate_recipe['inputs']
        expected_cold=(original_recipe['inputs']['compiler_stage_reuse_cold'] if view else
                       {'path':str(a.baseline_cold/'acceptance.json'),'sha256':sha(a.baseline_cold/'acceptance.json')})
        if (reuse_inputs['compiler_stage_reuse_cold']!=expected_cold
                or reuse_inputs['compiler_stage_reuse_config']!={'path':str(a.config),'sha256':sha(a.config)}
                or sha(ROOT/'tools/qwen3/reuse_accepted_compiler_objects.py')!=reuse_inputs['compiler_stage_reuse_source']['sha256']):
            raise ValueError('Reuse source, accepted baseline cold and coordinator configuration must be pinned')
    selected_flag='gqa_value_forward_query1' if forward else 'gqa_value_view_query1' if view else 'empty_tensor_cse_query1' if cse else 'fixed512_copy_query1' if combined else 'fixed512_copy_query2_to32' if fixed512 else 'silu_effects_query2_to32' if multi_silu else 'packed_multi_i8_query2_to32' if multi_candidate else ('guarded_copy_query1' if copy_candidate else 'silu_effects_query1')
    replaced={'packed_i8_runtime','compiler_stage_reuse_source','compiler_stage_reuse_cold'} if combined else set()
    if cse:replaced={'compiler_stage_reuse_source','compiler_stage_reuse_cold','prepared_stage_reuse_source'}
    if view:replaced={'compiler_stage_reuse_source','prepared_stage_reuse_source','compiler_stage_reuse_config'}
    if (candidate_recipe.get(selected_flag) is not True
            or set(candidate_recipe['inputs'])!=set(original_recipe['inputs'])|required
            or any(candidate_recipe['inputs'][k]!=v for k,v in original_recipe['inputs'].items() if k not in replaced)
            or any(candidate_recipe[k]!=v for k,v in original_recipe.items() if k!='inputs')):
        raise ValueError('Original numerical recipe plus only explicit selected helper/runtime required')
    if view:
        i=candidate_recipe['inputs'];model_build=json.loads(Path(early['build']['path']).read_text())
        prefix='query1_gqa_value_forward' if forward else 'query1_gqa_value_view'
        sdk=json.loads(Path(i[prefix+'_sdk_receipt']['path']).read_text())
        prior=json.loads(Path(model_build['accepted_baseline_closure']['path']).read_text())
        if (not stage_reuse or not prepared_reuse or not original_recipe.get('empty_tensor_cse_query1')
                or candidate_recipe.get('gqa_value_view_contract')!='original_Qwen_query1_identity_V_shape_view_v1'
                or i[prefix+'_sdk_receipt']!=model_build['compiler_receipt']
                or i[prefix+'_objects_receipt']!=model_build['objects_receipt']
                or i[prefix+'_compiler']!={k:sdk['candidate'][k] for k in ('path','sha256')}
                or i['compiler']['sha256']!=sdk['accepted_sdk']['sha256']
                or prior['status']!=('query1_GQA_value_view_native_timing_and_original_states_accepted' if forward else
                                    'query1_empty_tensor_CSE_resumed_native_timing_and_original_states_accepted')
                or prior['DSP_calls']!=393 or not prior['model_inference_acceptance']
                or prior['candidate_libraries']!={str(k):v for k,v in baseline_libraries.items()}
                or sha(model_build['accepted_baseline_closure']['path'])!=model_build['accepted_baseline_closure']['sha256']):
            raise ValueError('Only the actually accepted CSE baseline plus proved original V view object required')
        if forward and (not original_recipe.get('gqa_value_view_query1') or
                candidate_recipe.get('gqa_value_forward_contract')!='original_Qwen_query1_component_slices_through_proved_V_repeat_v1'):
            raise ValueError('Accepted identity-view baseline and explicit repeat-slice contract required')
    if cse:
        i=candidate_recipe['inputs'];model_build=json.loads(Path(early['build']['path']).read_text())
        sdk=json.loads(Path(i['query1_cse_sdk_receipt']['path']).read_text())
        if (not stage_reuse or not prepared_reuse
                or candidate_recipe.get('empty_tensor_cse_contract')!='original_Qwen_query1_CSE_after_empty_tensor_v1'
                or i['query1_cse_sdk_receipt']!=model_build['compiler_receipt']
                or i['query1_cse_objects_receipt']!=model_build['objects_receipt']
                or i['query1_cse_compiler']!={k:sdk['candidate'][k] for k in ('path','sha256')}
                or i['compiler']['sha256']!=sdk['accepted_sdk']['sha256']):
            raise ValueError('Only the actual exact-output q1 CSE candidate may change the compiler')
    if combined:
        sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_fixed512_memcpy import QUERY1_CONTRACT
        from hexagon_q1_integer_tiles import CONTRACT,validate_runtime
        i=candidate_recipe['inputs']; model_build=json.loads(Path(early['build']['path']).read_text())
        if (not stage_reuse or candidate_recipe.get('integer_tiles_query1') is not True
                or candidate_recipe.get('integer_tiles_contract')!=CONTRACT
                or candidate_recipe.get('fixed512_copy_query1_contract')!=QUERY1_CONTRACT
                or i['packed_i8_runtime']['sha256']!=model_build['runtime']['sha256']
                or i['fixed512_copy_runtime']['sha256']!=model_build['fixed512_runtime']['sha256']):
            raise ValueError('Only the two actually accepted exact q1 runtimes may replace the original baseline')
        validate_runtime(i['packed_i8_runtime'],i['packed_i8_nm'],i['integer_tiles_source'],i['integer_tiles_header'],i['integer_tiles_original_header'])
    if multi_candidate:
        sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_multi_integer import CONTRACT
        from hexagon_guarded_memref_copy import DYNAMIC_CONTRACT
        if (candidate_recipe.get('guarded_copy_query2_to32') is not True
                or candidate_recipe.get('packed_multi_i8_contract')!=CONTRACT
                or candidate_recipe.get('guarded_copy_dynamic_contract')!=DYNAMIC_CONTRACT
                or any(original_recipe.get(k) is not True for k in ('packed_i8_query1','silu_effects_query1','guarded_copy_query1'))):
            raise ValueError('Original combined query1 policy plus exact bounded multi-token contract required')
        if multi_silu:
            from hexagon_silu_helper_effects import DYNAMIC_CONTRACT as SILU_DYNAMIC_CONTRACT
            if (original_recipe.get('packed_multi_i8_query2_to32') is not True
                    or original_recipe.get('guarded_copy_query2_to32') is not True
                    or candidate_recipe.get('silu_effects_dynamic_contract')!=SILU_DYNAMIC_CONTRACT):
                raise ValueError('Original exact integer/copy recipe plus guarded bounded multi-token SiLU required')
        if fixed512:
            from hexagon_fixed512_memcpy import CONTRACT as FIXED512_CONTRACT
            if (not original_recipe.get('silu_effects_query2_to32') is True
                    or candidate_recipe.get('fixed512_copy_contract')!=FIXED512_CONTRACT):
                raise ValueError('Original guarded multi-SiLU recipe plus explicit exact512 byte-copy contract required')
    _,baseline,_=accepted(a.baseline_warm);validate_parameter_mode(baseline,'exact-bytes');validate_float_policy(baseline,True)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    a.output.mkdir(parents=True)
    record=dict(status='helper_effect_iteration_preconditions_verified',source_revision=revision,
        candidate_kind=a.candidate_kind,
        source_sha256=sha(__file__),recipe_sha256=recipe_sha,early_model_revisit=dict(path=str(a.early_model_revisit),sha256=a.early_model_revisit_sha256),
        original_numerical_dependencies_unchanged=True,model_inference_acceptance=False,deadline_unix=a.deadline_unix,steps=[])
    if prepared_reuse:record['exact_preparation_stage_reuse_enabled']=True
    if combined:
        record.update(original_numerical_dependencies_unchanged=False,all_original_FP_and_multi_token_dependencies_unchanged=True,
            exact_integer_runtime_replaced_with_same_INT32_ring=True,
            prior_transport_process=dict(path=str(a.prior_transport_process),sha256=a.prior_transport_process_sha256))
    def save():write(a.output/'result.json',record)
    def pinned():
        if (subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=revision
                or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip() or sha(recipe)!=recipe_sha):
            raise ValueError('Pinned candidate source or recipe changed')
    def run(stage,command):
        pinned()
        if time.time()>=a.deadline_unix:raise TimeoutError('Authorized deadline reached before '+stage)
        owner=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target'],'pgrep -x buffer_stream'],capture_output=True,timeout=15)
        if owner.returncode!=1:raise ValueError('Existing DSP owner or owner inspection failed')
        record['status']='running_'+stage;save();write(a.output/(stage+'-command.json'),command)
        with (a.output/(stage+'.log')).open('xb') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        pinned();record['steps'].append(dict(stage=stage,completed_unix=time.time()));save()
    save()
    try:
        python=config['python'];tools=ROOT/'tools/qwen3';cold=a.output/'cold'
        run('cold',[python,str(tools/'run_native.py'),'--config',str(a.config),'--build',str(a.recipe_build),
            '--output',str(cold),'--name',a.name+'-cold','--specialize-floats'])
        _,libraries=candidate_libraries(cold)
        original=json.loads((a.baseline_cold/'native-result.json').read_text());actual=json.loads((cold/'native-result.json').read_text())
        if actual['DSP_calls']!=9:raise ValueError('Original actual cold9 required')
        if multi_candidate:
            if libraries[(2,32)]==baseline_libraries[(2,32)]:
                raise ValueError('Actual new selected multi-token library required')
        elif (libraries[(2,32)]!=baseline_libraries[(2,32)] or libraries[(1,1)]==baseline_libraries[(1,1)]):
            raise ValueError('Two actual graphs and unchanged q2..32 library required')
        compare_integer(original,actual,libraries);validate_float_policy(actual,True)
        record['candidate_libraries']={str(k):v for k,v in libraries.items()};save()
        warm=a.output/'warm16'
        run('warm16',[python,str(tools/'run_online_regression.py'),'--config',str(a.config),'--accepted-run',str(cold),
            '--output',str(warm),'--name',a.name+'-warm16','--observations',config['observations'],
            '--board-observations',board['observations'],'--prompts-json',json.dumps([
                'The capital of France is Paris. The capital of Germany is Berlin.',
                'Complete this Python function:\ndef add(a, b):\n    return']),
            '--max-new-tokens','8','--chunk-size','32','--runtime-catalog',str(a.runtime_catalog),
            '--board-runtime-catalog',str(a.board_runtime_catalog),'--generation-timing',
            '--parameter-check','exact-bytes','--specialize-floats'])
        _,native,_=accepted(warm)
        if native['DSP_calls']!=16:raise ValueError('Original warm invocation count changed')
        compare_integer(baseline,native,libraries);validate_parameter_mode(native,'exact-bytes');validate_float_policy(native,True)
        rows=[];all_queries={}
        for order,kind in enumerate(('baseline','candidate','candidate','baseline')):
            folder=a.output/('native-abba-'+str(order)+'-'+kind);oracle=a.baseline_warm if kind=='baseline' else warm
            run('native-abba-'+str(order),[python,str(tools/'benchmark_accepted_generation.py'),'--config',str(a.config),
                '--oracle',str(oracle),'--output',str(folder),'--name',a.name+'-native-abba-'+str(order)+'-'+kind])
            proof=json.loads((folder/'result.json').read_text());native=json.loads((folder/'native-result.json').read_text())
            if (proof['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
                    or proof['DSP_calls']!=16 or proof['inline_CPU_observer'] is not False
                    or proof['full_input_output_model_token_FNV_equality'] is not True):raise ValueError('Complete unwrapped native timing required')
            compare_integer(baseline,native,libraries if kind=='candidate' else baseline_libraries)
            validate_parameter_mode(native,'exact-bytes');validate_float_policy(native,True)
            for query,measure in proof['measurement']['query_groups'].items():
                med=measure['median_seconds'];all_queries.setdefault(query,[]).append(dict(order=order,kind=kind,
                    kernel_seconds=med['DSP_kernel_seconds'],callback_seconds=med['ARM_callback_seconds']))
            rows.append(dict(order=order,kind=kind,proof=dict(path=str(folder/'result.json'),sha256=sha(folder/'result.json'))))
        timing={}
        for query,measured in all_queries.items():
            if len(measured)!=4:raise ValueError('Original query coverage changed across native ABBA')
            timing[query]={}
            for key in ('kernel_seconds','callback_seconds'):
                old=statistics.mean(r[key] for r in measured if r['kind']=='baseline')
                new=statistics.mean(r[key] for r in measured if r['kind']=='candidate')
                timing[query][key]=dict(baseline=old,candidate=new,ratio=new/old)
        record['native_model_timing']=timing;save()
        write(a.output/'native-ABBA-result.json',dict(status='complete_selected_helper_native_model_ABBA',candidate_kind=a.candidate_kind,rows=rows,
            measurements=timing,DSP_calls=64,inline_CPU_observer=False,model_inference_acceptance=False))
        if not positive_native_gain(timing,multi_candidate):
            record.update(status='helper_effect_candidate_deferred_after_nonpositive_native_model_timing',
                DSP_calls=89,candidate_DSP_calls=57,paired_baseline_DSP_calls=32,
                long_state_suite_skipped_to_avoid_wasted_work=True);return
        states=a.output/'state304'
        run('state304',[python,str(tools/'run_integer_regressions.py'),'--baseline-suite',str(a.original_state_suite),
            '--accepted-run',str(cold),'--runtime-catalog',str(a.runtime_catalog),'--board-runtime-catalog',str(a.board_runtime_catalog),
            '--output',str(states),'--name',a.name+'-state','--deadline-unix',str(a.deadline_unix),
            '--parameter-check','exact-bytes','--specialize-floats'])
        result=json.loads((states/'result.json').read_text())
        if result['status']!='all_integer_candidate_state_regressions_passed' or result['DSP_calls']!=304:
            raise ValueError('Complete original state acceptance required')
        record.update(status='guarded_query1_SiLU_effects_native_timing_and_original_states_accepted',
            model_inference_acceptance=True,DSP_calls=393,candidate_DSP_calls=361,paired_baseline_DSP_calls=32,
            state_result=dict(path=str(states/'result.json'),sha256=sha(states/'result.json')),completed_unix=time.time())
        if copy_candidate:
            record['status']='guarded_query1_byte_copy_native_timing_and_original_states_accepted'
        if combined:
            record['status']='exact_query1_integer_tiles_fixed512_native_timing_and_original_states_accepted'
        elif cse:
            record['status']='query1_empty_tensor_CSE_native_timing_and_original_states_accepted'
        elif forward:
            record['status']='query1_GQA_value_forward_native_timing_and_original_states_accepted'
        elif view:
            record['status']='query1_GQA_value_view_native_timing_and_original_states_accepted'
        elif fixed512:
            record['status']='guarded_query2_to32_fixed512_byte_copy_native_timing_and_original_states_accepted'
        elif multi_silu:
            record['status']='guarded_query2_to32_SiLU_effects_native_timing_and_original_states_accepted'
        elif multi_candidate:
            record['status']='exact_query2_to32_integer_copy_native_timing_and_original_states_accepted'
    except BaseException as e:record.update(status='failed',error=str(e));raise
    finally:save()
    print(record['status'])


if __name__=='__main__':main()
