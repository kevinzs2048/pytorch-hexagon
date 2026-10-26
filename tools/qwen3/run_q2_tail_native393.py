"""Run the original full393 numerical and state closure for the guarded dynamic prefill QK tail."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from build import ROOT, entry, sha, write
from revisit_release_index import indexed_files
from revisit_native_compile_session_abba import require_normal_owner
from run_dynamic_pv_native393 import require_admission
from build_caller_copy_native_recipe import validate_transition
from run_integer_regressions import candidate_libraries, compare_integer, validate_parameter_mode, validate_float_policy
from compare_performance import accepted
from bounded_loader_policy import configured_policy, verify_native_layout
from q2_dispatch_native_gain import positive_native_gain


def require_common_runtime(oracle_command, runtime_catalog, board_catalog):
    expected = dict(path=str(runtime_catalog), board_path=str(board_catalog), sha256=sha(runtime_catalog))
    command = oracle_command['board']
    if (oracle_command.get('runtime_override') != expected or command.count('--catalog') != 1
            or command.count('--catalog-sha256') != 1
            or command[command.index('--catalog')+1] != str(board_catalog)
            or command[command.index('--catalog-sha256')+1] != expected['sha256']):
        raise ValueError('Actual baseline and candidate must use the exact same runtime')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'recipe-build', 'preflight', 'preflight-controller', 'baseline-independent-revisit',
                 'original-state-suite', 'runtime-catalog', 'board-runtime-catalog', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--verified-cold', type=Path, required=True)
    p.add_argument('--cold-independent-revisit', type=Path, required=True)
    p.add_argument('--cold-review-controller', type=Path, required=True)
    p.add_argument('--q2-identical-tail',dest='q2_identical_tail',action='store_true',required=True,help='Qualify the separately proven exact dynamic prefill tail')
    p.add_argument('--name', required=True)
    p.add_argument('--deadline-unix', type=int, required=True)
    a = p.parse_args(); require_admission(a.deadline_unix, time.time())
    read = lambda path: json.loads(Path(path).read_text())
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if (not a.output.is_absolute() or a.output.exists() or git('status', '--porcelain')
            or not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.name)):
        raise ValueError('Fresh named closure and clean source required')
    prefix = 'q2_tail'
    revision = git('rev-parse', 'HEAD'); prior = read(a.baseline_independent_revisit)
    prior_status='global_copy_original393_independently_revisited'
    if (prior['status'] != prior_status or prior['DSP_calls'] != 393
            or prior['model_inference_acceptance'] is not True):
        raise ValueError('Qualified dynamic PV393 baseline required')
    refs = []; indexed_files(prior, refs)
    baseline = read(prior['closure']['path'])
    cold0, warm0 = Path(baseline['cold_run']), Path(baseline['warm_run'])
    base0, original_libraries = candidate_libraries(cold0)
    if prior['candidate_libraries'] != {str(k):v for k,v in original_libraries.items()}:
        raise ValueError('Baseline closure and actual original libraries differ')
    build = read(a.recipe_build/'result.json'); pre = read(a.preflight)
    indexed_files(build, refs); indexed_files(pre, refs)
    preowner = read(a.preflight_controller)
    require_normal_owner(a.preflight_controller, preowner['authorized_deadline_unix'])
    from revisit_q2_dispatch_preflight import actual_owner
    actual_owner(a.preflight_controller, pre, 'preflight_q2_tail_native_recipe.py', a.preflight.parent)
    recipe = a.recipe_build/'machine-recipe.json'; recipe_sha = sha(recipe)
    from build_q2_tail_native_recipe import validate_transition as copy_transition
    from q2_tail_release import tested_model, require_cold_review, require_reviewer_owner
    copy_transition(read(base0['recipe']['path']), read(recipe))
    model, expected = tested_model(build)
    review = read(a.cold_independent_revisit)
    numerical = require_cold_review(review, cold=a.verified_cold, recipe=recipe, libraries=expected, baseline=cold0)
    require_reviewer_owner(review, a.cold_review_controller)
    numerical_root, numerical_revision = numerical['source_root'], numerical['revision']
    if (pre['status'] != 'Q2_identical_tail_actual_public_two_domain_exact_tested_DSOs_NOT_native_acceptance'
            or pre['source_revision'] != preowner['source_revision'] or preowner['source_root'] != numerical_root
            or pre['recipe'] != build['recipe']
            or pre['new_model_PRIMARY_compilations'] or pre['DSP_calls']
            or {tuple(r['query_bounds']):r['DSO']['sha256'] for r in pre['rows']} != expected):
        raise ValueError('Own exact public two-graph preflight required')
    c = read(a.config); policy = configured_policy(c)
    if not policy:
        raise ValueError('Common bounded constant loader required')
    require_common_runtime(read(warm0/'commands.json'), a.runtime_catalog, a.board_runtime_catalog)
    _, old_warm, _ = accepted(warm0)
    validate_parameter_mode(old_warm, 'exact-bytes'); validate_float_policy(old_warm, True)
    a.output.mkdir()
    record = dict(status=prefix+'_original393_admitted', candidate_kind='ordered-q2-identical-tail-row-guard',
        source_root=str(ROOT), source_revision=revision, numerical_source_root=numerical_root, numerical_source_revision=numerical_revision,
        source=entry(__file__), recipe=entry(recipe), recipe_build=entry(a.recipe_build/'result.json'),
        preflight=entry(a.preflight), preflight_controller=entry(a.preflight_controller),
        baseline_closure=prior['closure'], baseline_independent_revisit=entry(a.baseline_independent_revisit),
        original_state_suite=str(a.original_state_suite), baseline_cold=str(cold0), baseline_warm=str(warm0),
        new_primary_model_codegen_invocations=0, model_inference_acceptance=False, performance_acceptance=False,
        actual_start_minimum_remaining_seconds=14400, deadline_unix=a.deadline_unix, steps=[],
        common_payload_preserving_loader_policy=policy, common_native_runtime_override=dict(
            path=str(a.runtime_catalog), board_path=str(a.board_runtime_catalog), sha256=sha(a.runtime_catalog)))
    record['fresh_cold9_independently_reviewed']=True
    record['q2_tail_candidate']=True
    record.update(reused_cold_review=entry(a.cold_independent_revisit),
        reused_cold_reviewer_controller=entry(a.cold_review_controller))
    save = lambda: write(a.output/'result.json', record)
    def run(stage, command):
        if (git('rev-parse', 'HEAD') != revision or git('status', '--porcelain')
                or sha(recipe) != recipe_sha or time.time() >= a.deadline_unix):
            raise ValueError('Pinned source, recipe or budget changed')
        owner = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            c['board']['ssh_target'], 'pgrep -x buffer_stream'], capture_output=True, timeout=15)
        if owner.returncode != 1:
            raise ValueError('DSP must be free before each original stage')
        record['status'] = 'running_'+stage; save(); write(a.output/(stage+'-command.json'), command)
        with (a.output/(stage+'.log')).open('xb') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        record['steps'].append(dict(stage=stage, completed_unix=time.time())); save()
    save(); py = c['python']; tools = ROOT/'tools/qwen3'; cold = a.verified_cold; warm = a.output/'warm16'
    try:
        _, libraries = candidate_libraries(cold); actual = read(cold/'native-result.json')
        if libraries != expected or actual['DSP_calls'] != 9:
            raise ValueError('Own original cold9 and exact candidate libraries required')
        compare_integer(read(cold0/'native-result.json'), actual, libraries); validate_float_policy(actual, True)
        fingerprint = verify_native_layout(actual)
        record.update(candidate_libraries={str(k):v for k,v in libraries.items()}, cold_run=str(cold), warm_run=str(warm)); save()
        old_command = read(warm0/'commands.json')['board']
        prompts_json = old_command[old_command.index('--prompts-json')+1]
        run('warm16', [py, str(tools/'run_online_regression.py'), '--config', str(a.config), '--accepted-run', str(cold),
            '--output', str(warm), '--name', a.name+'-warm16', '--observations', c['observations'],
            '--board-observations', c['board']['observations'], '--prompts-json', prompts_json,
            '--max-new-tokens', '8', '--chunk-size', '32', '--runtime-catalog', str(a.runtime_catalog),
            '--board-runtime-catalog', str(a.board_runtime_catalog), '--generation-timing',
            '--parameter-check', 'exact-bytes', '--specialize-floats'])
        _, actual, _ = accepted(warm)
        if actual['DSP_calls'] != 16 or verify_native_layout(actual) != fingerprint:
            raise ValueError('Original warm16 and exact common loader required')
        compare_integer(old_warm, actual, libraries); validate_parameter_mode(actual, 'exact-bytes'); validate_float_policy(actual, True)
        groups = {}; rows = []
        for i, kind in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
            folder = a.output/('native-abba-'+str(i)+'-'+kind); oracle = warm0 if kind == 'baseline' else warm
            run('native-abba-'+str(i), [py, str(tools/'benchmark_accepted_generation.py'), '--config', str(a.config),
                '--oracle', str(oracle), '--output', str(folder), '--name', a.name+'-native-abba-'+str(i)+'-'+kind])
            proof = read(folder/'result.json'); native = read(folder/'native-result.json')
            if (proof['DSP_calls'] != 16 or proof['inline_CPU_observer'] is not False
                    or proof.get('timings_eligible_for_performance_acceptance') is not True
                    or proof['full_input_output_model_token_FNV_equality'] is not True
                    or verify_native_layout(native) != fingerprint):
                raise ValueError('Native unwrapped timing and all original packets required')
            compare_integer(old_warm, native, libraries if kind == 'candidate' else original_libraries)
            validate_parameter_mode(native, 'exact-bytes'); validate_float_policy(native, True)
            for query, measurement in proof['measurement']['query_groups'].items():
                med = measurement['median_seconds']; groups.setdefault(query, []).append(dict(kind=kind,
                    kernel_seconds=med['DSP_kernel_seconds'], callback_seconds=med['ARM_callback_seconds']))
            rows.append(dict(order=i, kind=kind, proof=entry(folder/'result.json')))
        timing = {}
        for query, measured in groups.items():
            if len(measured) != 4:
                raise ValueError('Identical query coverage in all timing arms required')
            timing[query] = {}
            for metric in ('kernel_seconds', 'callback_seconds'):
                old = statistics.mean(r[metric] for r in measured if r['kind'] == 'baseline')
                new = statistics.mean(r[metric] for r in measured if r['kind'] == 'candidate')
                timing[query][metric] = dict(baseline=old, candidate=new, ratio=new/old)
        record['native_model_timing'] = timing
        write(a.output/'native-ABBA-result.json', dict(status='complete_selected_helper_native_model_ABBA',
            rows=rows, measurements=timing, DSP_calls=64, model_inference_acceptance=False)); save()
        if not positive_native_gain(timing):
            record.update(status=prefix+'_deferred_after_nonpositive_original_native89', DSP_calls=89,
                candidate_DSP_calls=57, paired_baseline_DSP_calls=32, original_state304_executed=False,
                newly_executed_DSP_calls=80,
                reused_independently_verified_DSP_calls=9, completed_unix=time.time())
            return
        states = a.output/'state304'
        run('state304', [py, str(tools/'run_integer_regressions.py'), '--baseline-suite', str(a.original_state_suite),
            '--accepted-run', str(cold), '--runtime-catalog', str(a.runtime_catalog), '--board-runtime-catalog',
            str(a.board_runtime_catalog), '--output', str(states), '--name', a.name+'-state',
            '--deadline-unix', str(a.deadline_unix), '--parameter-check', 'exact-bytes', '--specialize-floats'])
        state = read(states/'result.json')
        if state['status'] != 'all_integer_candidate_state_regressions_passed' or state['DSP_calls'] != 304:
            raise ValueError('All six original states and304 actual calls required')
        record.update(status=prefix+'_original393_complete_NOT_independently_accepted', DSP_calls=393,
            candidate_DSP_calls=361, paired_baseline_DSP_calls=32, original_state304_executed=True,
            state_result=entry(states/'result.json'), newly_executed_DSP_calls=384,
            reused_independently_verified_DSP_calls=9, completed_unix=time.time())
    except BaseException as error:
        record.update(status='failed', error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
