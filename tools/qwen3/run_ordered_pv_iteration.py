"""Run original cold9, warm16, native ABBA64 and state304 for ordered q1 PV.

The positive packet8 is a prerequisite, never generation acceptance. No
native89-only entry is exposed; admission always requires four hours.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from build import ROOT, entry, sha, write
from build_ordered_pv_recipe import CONTRACT, validate_transition
from ordered_pv_model_contract import KIND, require_reference, require_compiler
from revisit_key_tile_model import revisit
from compare_performance import accepted
from run_integer_regressions import candidate_libraries, compare_integer, validate_float_policy, validate_parameter_mode
from bounded_loader_policy import configured_policy, verify_native_layout
from run_helper_effects_iteration import positive_native_gain, require_common_runtime
from run_regression import require_configured_python


def require_admission(deadline, now):
    if deadline-now < 14400:
        raise ValueError('Original PV393 requires the unchanged four-hour admission budget')


def require_early(proof, owner):
    if (proof['candidate_kind'] != KIND or type(proof['DSP_calls']) is not int or proof['DSP_calls'] != 8
            or proof['positive_early_model_gain'] is not True
            or proof['model_inference_acceptance'] is not False
            or proof['full_original_logits_and_56_KV_byteexact'] is not True
            or owner['candidate_kind'] != KIND or type(owner['DSP_calls']) is not int or owner['DSP_calls'] != 8
            or owner['status'] != 'ordered_PV_register_original_QK393_q1_packet8_raw_FNV_and_normal_owners_independently_revisited'
            or owner['positive_early_model_gain'] is not True
            or owner['model_inference_acceptance'] is not False
            or owner['complete_input_FNV']['all_three_FNV_recomputed'] is not True):
        raise ValueError('Own positive PV packet8 and complete independent FNV/owner proof required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'recipe-build', 'early-model-revisit', 'owner-revisit',
                 'baseline-closure', 'baseline-independent-revisit', 'original-state-suite',
                 'runtime-catalog', 'board-runtime-catalog', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--name', required=True); p.add_argument('--deadline-unix', type=int, required=True)
    a = p.parse_args(); read = lambda path: json.loads(Path(path).read_text())
    require_admission(a.deadline_unix, time.time())
    if (not a.output.is_absolute() or a.output.exists() or not a.name
            or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.name)):
        raise ValueError('Fresh named original PV closure required')
    git = lambda *x: subprocess.check_output(['git', '-C', str(ROOT), *x], text=True).strip()
    revision = git('rev-parse', 'HEAD')
    if git('status', '--porcelain'): raise ValueError('Frozen clean PV source required')
    prior = read(a.baseline_closure/'result.json'); reference = entry(a.baseline_closure/'result.json')
    require_reference(prior, read(a.baseline_independent_revisit), reference)
    baseline_cold = a.baseline_closure/'cold'; baseline_warm = a.baseline_closure/'warm16'
    base, baseline_libraries = candidate_libraries(baseline_cold)
    early = read(a.early_model_revisit); owners = read(a.owner_revisit)
    require_early(early, owners)
    if (owners['packet_revisit'] != entry(a.early_model_revisit)
            or Path(early['accepted_cold']['path']).parent != baseline_cold
            or revisit(Path(early['run']['path']), Path(early['plan']['path']), baseline_cold) != early):
        raise ValueError('Exact own PV8 bound to the current original QK cold required')
    build = read(early['build']['path']); sdk = read(build['compiler_receipt']['path'])
    code = read(build['objects_receipt']['path'])
    require_compiler(sdk, Path(code['compiler_diagnostics']['path']).read_text())
    if (build['accepted_baseline_closure'] != reference
            or build['original_model_library']['sha256'] != baseline_libraries[(1, 1)]):
        raise ValueError('PV producer lineage differs from the current baseline')
    recipe = a.recipe_build/'machine-recipe.json'; recipe_sha = sha(recipe)
    candidate_recipe = read(recipe); original_recipe = read(base['recipe']['path'])
    validate_transition(original_recipe, candidate_recipe)
    expected = copy.deepcopy(original_recipe)
    expected.update(ordered_pv_register_query1=True, ordered_pv_register_contract=CONTRACT)
    expected['inputs'].update(
        query1_ordered_pv_compiler={k: sdk['candidate'][k] for k in ('path', 'sha256')},
        query1_ordered_pv_sdk_receipt=build['compiler_receipt'], query1_ordered_pv_objects_receipt=build['objects_receipt'],
        query1_ordered_pv_contract_source=entry(ROOT/'tools/qwen3/ordered_pv_model_contract.py'),
        compiler_stage_reuse_source=entry(ROOT/'tools/qwen3/reuse_accepted_compiler_objects.py'),
        prepared_stage_reuse_source=entry(ROOT/'tools/qwen3/reuse_accepted_prepared_linalg.py'),
        compiler_stage_reuse_config=entry(a.config))
    if candidate_recipe != expected: raise ValueError('Exact owned PV producer dependencies required')
    for row in candidate_recipe['inputs'].values():
        if sha(row['path']) != row['sha256']: raise ValueError('Pinned native recipe dependency changed')
    config = read(a.config); require_configured_python(config); board = config['board']
    loader_policy = configured_policy(config)
    if not loader_policy: raise ValueError('Common original payload-preserving loader required')
    require_common_runtime(read(baseline_warm/'commands.json'), a.runtime_catalog, a.board_runtime_catalog)
    _, baseline, _ = accepted(baseline_warm)
    validate_parameter_mode(baseline, 'exact-bytes'); validate_float_policy(baseline, True)
    a.output.mkdir()
    record = dict(status='ordered_PV_original393_admitted', candidate_kind=KIND,
        source_revision=revision, source_sha256=sha(__file__), recipe_sha256=recipe_sha,
        baseline_closure=reference, baseline_independent_revisit=entry(a.baseline_independent_revisit),
        early_model_revisit=entry(a.early_model_revisit), owner_revisit=entry(a.owner_revisit),
        original_numerical_dependencies_unchanged=True, original_full393_minimum_remaining_seconds=14400,
        new_primary_model_codegen_invocations=0, model_inference_acceptance=False,
        deadline_unix=a.deadline_unix, steps=[], common_payload_preserving_loader_policy=loader_policy,
        common_native_runtime_override=dict(path=str(a.runtime_catalog), board_path=str(a.board_runtime_catalog), sha256=sha(a.runtime_catalog)))
    def save(): write(a.output/'result.json', record)
    def run(stage, command):
        if (git('rev-parse', 'HEAD') != revision or git('status', '--porcelain')
                or sha(recipe) != recipe_sha or time.time() >= a.deadline_unix):
            raise ValueError('Pinned source, recipe or authorized budget changed')
        owner = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', board['ssh_target'],
            'pgrep -x buffer_stream'], capture_output=True, timeout=15)
        if owner.returncode != 1: raise ValueError('Existing DSP owner or inspection failed')
        record['status'] = 'running_'+stage; save(); write(a.output/(stage+'-command.json'), command)
        with (a.output/(stage+'.log')).open('xb') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        record['steps'].append(dict(stage=stage, completed_unix=time.time())); save()
    save()
    # The remaining commands intentionally replay the original suite unchanged.
    try:
        python=config['python'];tools=ROOT/'tools/qwen3';cold=a.output/'cold'
        run('cold',[python,str(tools/'run_native.py'),'--config',str(a.config),'--build',str(a.recipe_build),
            '--output',str(cold),'--name',a.name+'-cold','--specialize-floats'])
        _,libraries=candidate_libraries(cold)
        original=json.loads((baseline_cold/'native-result.json').read_text());actual=json.loads((cold/'native-result.json').read_text())
        if actual['DSP_calls']!=9:raise ValueError('Original actual cold9 required')
        loader_fingerprint=verify_native_layout(actual) if loader_policy else None
        if libraries[(2,32)]!=baseline_libraries[(2,32)] or libraries[(1,1)]!=build['library']['sha256']:
            raise ValueError('Actual unchanged QK graph and exact early-tested PV q1 DSO required')
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
        if loader_policy and verify_native_layout(native)!=loader_fingerprint:
            raise ValueError('Warm model constant loader policy changed')
        compare_integer(baseline,native,libraries);validate_parameter_mode(native,'exact-bytes');validate_float_policy(native,True)
        rows=[];all_queries={}
        for order,kind in enumerate(('baseline','candidate','candidate','baseline')):
            folder=a.output/('native-abba-'+str(order)+'-'+kind);oracle=baseline_warm if kind=='baseline' else warm
            run('native-abba-'+str(order),[python,str(tools/'benchmark_accepted_generation.py'),'--config',str(a.config),
                '--oracle',str(oracle),'--output',str(folder),'--name',a.name+'-native-abba-'+str(order)+'-'+kind])
            proof=json.loads((folder/'result.json').read_text());native=json.loads((folder/'native-result.json').read_text())
            if (proof['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
                    or proof['DSP_calls']!=16 or proof['inline_CPU_observer'] is not False
                    or proof.get('timings_eligible_for_performance_acceptance') is not True
                    or proof['full_input_output_model_token_FNV_equality'] is not True):raise ValueError('Complete unwrapped native timing required')
            if loader_policy and verify_native_layout(native)!=loader_fingerprint:
                raise ValueError('Native ABBA baseline and candidate loader policies differ')
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
            for metric in ('kernel_seconds','callback_seconds'):
                old=statistics.mean(r[metric] for r in measured if r['kind']=='baseline')
                new=statistics.mean(r[metric] for r in measured if r['kind']=='candidate')
                timing[query][metric]=dict(baseline=old,candidate=new,ratio=new/old)
        record['native_model_timing']=timing;save()
        write(a.output/'native-ABBA-result.json',dict(status='complete_selected_helper_native_model_ABBA',candidate_kind=KIND,rows=rows,
            measurements=timing,DSP_calls=64,inline_CPU_observer=False,model_inference_acceptance=False))
        if set(timing)!={'1','13','14'}:
            raise ValueError('Complete original q1/q13/q14 timing coverage required')
        if not positive_native_gain(timing,False):
            record.update(status='ordered_PV_deferred_after_nonpositive_original_native89',
                DSP_calls=89,candidate_DSP_calls=57,paired_baseline_DSP_calls=32,
                original_state304_executed=False,completed_unix=time.time(),
                long_state_suite_skipped_to_avoid_wasted_work=True);return
        states=a.output/'state304'
        run('state304',[python,str(tools/'run_integer_regressions.py'),'--baseline-suite',str(a.original_state_suite),
            '--accepted-run',str(cold),'--runtime-catalog',str(a.runtime_catalog),'--board-runtime-catalog',str(a.board_runtime_catalog),
            '--output',str(states),'--name',a.name+'-state','--deadline-unix',str(a.deadline_unix),
            '--parameter-check','exact-bytes','--specialize-floats'])
        result=json.loads((states/'result.json').read_text())
        if result['status']!='all_integer_candidate_state_regressions_passed' or result['DSP_calls']!=304:
            raise ValueError('Complete original state acceptance required')
        record.update(status='ordered_PV_original393_complete_NOT_independently_accepted',
            model_inference_acceptance=False,original_state304_executed=True,DSP_calls=393,candidate_DSP_calls=361,paired_baseline_DSP_calls=32,
            state_result=dict(path=str(states/'result.json'),sha256=sha(states/'result.json')),completed_unix=time.time())
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));raise
    finally:save()
    print(record['status'])


if __name__=='__main__':main()
