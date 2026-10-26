"""Close one selected model's AOT and warm deployment using original contracts.

All DSP stages are serial. Each stage keeps its committed producer, independent
reviewer, full-output oracle and actual normal process exit. A failure stops the
queue with its evidence intact; no rollback is substituted into the measurements.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from build import ROOT, HERE, entry, sha, write
from native_model_acceptance import require_model_acceptance
from native_warm_deployment import require_warm_deployment, LIVE, WARM
from run_native_aot_state_suite import EXPECTED


def require_completed_prefix(directory, controller, index, oracle):
    """Reuse the original 400 completed calls, retaining the failed queue record."""
    from generate_native_aot import admit
    from revisit_native_compile_session_abba import require_normal_owner
    read = lambda p: json.loads(Path(p).read_text())
    old = read(directory/'result.json'); parent = read(controller)
    if (old['status'] != 'failed' or old['current_stage'] != 'warm-seed16'
            or parent['status'] != 'failed' or parent['returncode'] == 0
            or parent['source_root'] != str(ROOT) or parent['source_revision'] != old['source_revision']
            or parent['command'][1] != str(HERE/'run_native_warm_release.py')
            or parent['command'][parent['command'].index('--output')+1] != str(directory)
            or any(Path('/proc',str(parent[k])).exists() for k in ('controller_pid','child_pid'))):
        raise ValueError('Exact closed failed warm queue with its untouched completed prefix required')
    expected = ['native16','artifact-catalog','cache-seed16','cache-seed16-review']
    for i in range(4): expected += ['cache-abba-'+str(i),'cache-abba-'+str(i)+'-review']
    expected += ['cache-ABBA64-review','cache-state304']
    if [row['stage'] for row in old['steps']] != expected:
        raise ValueError('All original completed 400-call prefix stages required')
    failed = directory/'warm-seed16'
    if (not (failed/'run-exit.json').is_file() or (failed/'native-result.json').exists()
            or read(failed/'run-exit.json') != dict(board_returncode=1,worker_returncode=None)
            or 'recording-space margin required' not in (failed/'board.log').read_text()):
        raise ValueError('Only the exact pre-DSP recording-space refusal may resume this prefix')
    for row in old['steps']:
        if entry(row['controller']['path']) != row['controller']:
            raise ValueError('Retained normal stage controller changed')
        require_normal_owner(row['controller']['path'],old['deadline_unix'])
    base = directory/'native16'; artifact = directory/'artifact-catalog'
    cache_state = directory/'cache-state304'; cache_abba = directory/'cache-ABBA64-review.json'
    model, manifest, baseline, prior = admit(index,base,cache_state,cache_abba)
    if (old['accepted_model'] != model or baseline['oracle_acceptance_sha256'] != sha(oracle/'acceptance.json')
            or read(artifact/'result.json')['accepted_model'] != model
            or read(directory/'namespace.json') != manifest):
        raise ValueError('Retained prefix must bind the same accepted model, namespace and complete oracle')
    old_name = parent['command'][parent['command'].index('--name')+1]
    return dict(old=old,base=base,artifact=artifact,namespace=directory/'namespace.json',cache_name=old_name+'-live',
        cache_state=cache_state,cache_abba=cache_abba,artifact_owner=directory/'artifact-catalog-process.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'index', 'oracle', 'online-source', 'baseline-state-suite', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--deadline-unix', type=int, required=True)
    p.add_argument('--completed-duration-reference',type=Path,action='append',default=[],
        help='Optional two original completed800-call results for an evidence-based reserve')
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--resume-controller',type=Path)
    a = p.parse_args()
    if bool(a.resume_from) != bool(a.resume_controller):
        p.error('Original completed prefix and its failed queue controller are required together')
    read = lambda path: json.loads(Path(path).read_text())
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if (a.output.exists() or not a.output.is_absolute() or git('status', '--porcelain')
            or not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.name)):
        raise ValueError('Fresh frozen serial release queue required')
    index = read(a.index); model, _ = require_model_acceptance(index)
    from native_warm_release_budget import admission_budget
    budget=admission_budget(index,a.completed_duration_reference,a.deadline_unix,time.time())
    config = read(a.config); revision = git('rev-parse', 'HEAD')
    a.output.mkdir(); steps = []
    record = dict(status='running', source_root=str(ROOT), source_revision=revision,
                  index=entry(a.index), accepted_model=model, steps=steps,
                  deadline_unix=a.deadline_unix, new_model_PRIMARY_compilations=0,
                  budget_admission=budget,
                  reused_completed_DSP_calls=0,
                  production_launcher_switched=False, performance_acceptance=False)
    def save(): write(a.output/'result.json', record)
    def own(script, flags, stem, minimum=300):
        if git('rev-parse', 'HEAD') != revision or git('status', '--porcelain'):
            raise ValueError('Release producer source changed during execution')
        command = [sys.executable, str(HERE/script), *map(str, flags)]
        plan = a.output/(stem+'-plan.json'); owner = a.output/(stem+'-process.json')
        write(plan, dict(source_root=str(ROOT), source_revision=revision,
                        deadline_unix=a.deadline_unix, minimum_remaining_seconds=minimum, command=command))
        record['current_stage'] = stem; save()
        subprocess.run([sys.executable, str(HERE/'run_owned_command.py'),
                        '--plan', str(plan), '--controller', str(owner)], check=True)
        steps.append(dict(stage=stem, controller=entry(owner), completed_unix=time.time())); save()
        return owner
    base = a.output/'native16'; artifact = a.output/'artifact-catalog'
    namespace = a.output/'namespace.json'; cache_name = a.name+'-live'
    save()
    try:
        if a.resume_from is not None:
            retained = require_completed_prefix(a.resume_from,a.resume_controller,index,a.oracle)
            base,artifact,namespace = retained['base'],retained['artifact'],retained['namespace']
            cache_state,cache_abba = retained['cache_state'],retained['cache_abba']
            artifact_owner = retained['artifact_owner']; cache_name = retained['cache_name']
            record.update(retained_prefix=entry(a.resume_from/'result.json'),
                retained_failed_queue_controller=entry(a.resume_controller),reused_completed_DSP_calls=400,
                original_failed_queue_record_preserved=True)
            save()
        else:
            own('benchmark_accepted_generation.py', ['--config', a.config, '--oracle', a.oracle,
             '--output', base, '--name', a.name+'-native16', '--input-packer', 'joined',
             '--startup-timing'], 'native16', 1800)
            native = read(base/'native-result.json')
            context = read(read(base/'result.json')['worker'][read(base/'result.json')['worker'].index('--context')+1])
            framework = read(HERE/'board-framework.lock.json')
            # The actual coordinator verifies the pinned board framework first;
            # namespace platform fields are overwritten from the actual ARM process.
            manifest = dict(adapter_sha256=sha(ROOT/'qwen3/runtime/support/native_aot_output.py'),
                native_source_sha256=native['source_sha256'], numerical_revision=model['source_revision'],
                recipe_sha256=index['machine_recipe']['sha256'], runtime_catalog_sha256=index['runtime_catalog']['sha256'],
                model_parameter_sha256=context['historical_parameter_sha256'],
                module_constants_sha256=native['parameter_validation']['guarded_python_float_constants']['ledger_sha256'],
                torch_version=framework['versions']['torch'], torch_git=framework['torch_git'],
                python_version=framework['python'], architecture=framework['architecture'])
            write(namespace, manifest)
            artifact_owner = own('build_native_artifact_catalog.py', ['--index', a.index, '--output', artifact,
                '--board-library-root', '/dev/shm/'+a.name+'-model-libraries'], 'artifact-catalog')
            def cache_arm(mode, stem, case):
                folder = a.output/stem
                owner = own('probe_native_aot_cached_session.py', [
                    '--config', a.config, '--baseline-native', base, '--output', folder, '--name', a.name+'-'+stem,
                    '--namespace-template', namespace, '--aot-mode', mode, '--cache-name', cache_name,
                    '--oracle', a.oracle, '--numerical-source', model['source_root'], '--worker-mode', 'session',
                    '--deadline-unix', a.deadline_unix], stem, 1800)
                proof = a.output/(stem+'-review.json')
                reviewer = own('revisit_native_aot_cached_session.py', ['--config', a.config, '--oracle', a.oracle,
                    '--native', folder, '--controller', owner, '--numerical-source', model['source_root'],
                    '--cache-case', case, '--output', proof], stem+'-review')
                return dict(native=str(folder), controller=entry(owner), independent_review=entry(proof),
                            reviewer_controller=entry(reviewer))
            cache_arm('cache', 'cache-seed16', 'seed')
            arms = []
            for order, mode in enumerate(('original', 'cache', 'cache', 'original')):
                row = cache_arm(mode, 'cache-abba-'+str(order), 'original' if mode == 'original' else 'hit')
                arms.append(dict(order=order, aot_mode=mode, **row))
            arms_path = a.output/'cache-arms.json'; write(arms_path, dict(arms=arms, deadline_unix=a.deadline_unix))
            cache_abba = a.output/'cache-ABBA64-review.json'
            own('revisit_native_aot_startup_abba.py', ['--arms-manifest', arms_path, '--output', cache_abba], 'cache-ABBA64-review')
            cache_state = a.output/'cache-state304'
            cold = Path(read(model['acceptance']['path'])['cold_run'])
            own('run_native_aot_state_suite.py', ['--config', a.config, '--baseline-state-suite', a.baseline_state_suite,
                '--online-source', a.online_source, '--accepted-run', cold, '--namespace-template', namespace,
                '--output', cache_state, '--name', a.name+'-cache-state', '--cache-name', cache_name,
                '--deadline-unix', a.deadline_unix, '--reuse-baseline-references'], 'cache-state304', 14400)
        def generation(stem, oracle, mode, flags, expected):
            folder = a.output/stem
            options = ['--config', a.config, '--index', a.index, '--baseline-native', base,
                '--cache-state304', cache_state, '--cache-abba', cache_abba,
                '--cache-name', a.name+'-warm' if mode == WARM else cache_name,
                '--name', a.name+'-'+stem, '--deadline-unix', a.deadline_unix, '--output', folder, *flags]
            if mode == WARM: options += ['--artifact-catalog', artifact]
            owner = own('generate_native_aot.py', options, stem, 1800)
            review = a.output/(stem+'-review.json')
            options = ['--config', a.config, '--index', a.index, '--oracle', oracle,
                       '--native', folder, '--controller', owner, '--output', review, '--compiler-mode', mode]
            if mode == WARM: options += ['--artifact-catalog', artifact, '--artifact-controller', artifact_owner]
            reviewer = own('revisit_native_artifact_generation.py', options, stem+'-review')
            proof = read(review)
            if proof['DSP_calls'] != expected: raise ValueError('Original transition count changed')
            return dict(case=stem, DSP_calls=expected, native=entry(folder/'result.json'), controller=entry(owner),
                        independent_review=entry(review), reviewer_controller=entry(reviewer),
                        cache_protocol=proof['cache_protocol'], generation_timings=proof['generation_timings'])
        board_args = read(base/'result.json')['board']
        warm_flags = []
        for flag in ('--prompts-json', '--max-new-tokens', '--chunk-size'):
            warm_flags.extend([flag, board_args[board_args.index(flag)+1]])
        generation('warm-seed16', a.oracle, WARM, warm_flags, 16)
        generation('warm-hit16', a.oracle, WARM, warm_flags, 16)
        state_dir = a.output/'warm-state304'; state_dir.mkdir(); rows = []
        for case in read(a.baseline_state_suite/'plan.json')['cases']:
            flags = []
            for flag in ('--prompts-json', '--requests-json', '--max-new-tokens', '--chunk-size'):
                if flag in case['command']: flags.extend([flag, case['command'][case['command'].index(flag)+1]])
            if '--chat-template' in case['command']: flags.append('--chat-template')
            row = generation('artifact-state-'+case['case'], cache_state/case['case'], WARM, flags, EXPECTED[case['case']])
            rows.append(row)
            write(state_dir/'result.json', dict(status='running', rows=rows, deadline_unix=a.deadline_unix))
        state = dict(status='original_warm_catalog_state304_completed_DIAGNOSTIC', rows=rows, DSP_calls=304,
            source_revision=revision, deadline_unix=a.deadline_unix, CPU_reference_executed=False,
            production_launcher_switched=False, model_inference_acceptance=False, performance_acceptance=False)
        write(state_dir/'result.json', state)
        pair_dir = a.output/'warm-ABBA64'; pair_dir.mkdir(); paired = []
        for order, mode in enumerate((LIVE, WARM, WARM, LIVE)):
            row = generation('artifact-abba-'+str(order), a.oracle, mode, warm_flags, 16)
            paired.append(dict(order=order, compiler_mode=mode, **row))
        pair = dict(status='same_input_live_warm_warm_live_ABBA64_completed_DIAGNOSTIC', rows=paired,
            DSP_calls=64, startup_samples_per_mode=2, source_revision=revision, deadline_unix=a.deadline_unix,
            CPU_reference_executed=False, production_launcher_switched=False,
            model_inference_acceptance=False, performance_acceptance=False)
        write(pair_dir/'result.json', pair)
        deployment = dict(schema=1, status='accepted_dynamic_model_native_AOT_opt_in_deployment',
            config=str(a.config), index=str(a.index), baseline_native=str(base), cache_state304=str(cache_state),
            cache_abba=str(cache_abba), cache_name=a.name+'-warm', results_root=str(a.output/'public-results'),
            compiler_mode=WARM, artifact_catalog=str(artifact), entry_source=entry(HERE/'qwen3_native_infer.py'),
            warm_validation=dict(state304=entry(state_dir/'result.json'), ABBA64=entry(pair_dir/'result.json'),
                artifact_preparation=entry(artifact/'result.json'), artifact_controller=entry(artifact_owner)))
        require_warm_deployment(deployment)
        write(a.output/'warm-deployment.json', deployment)
        timings = {}
        for label, select in (
            ('first_token_seconds', lambda r: r['generation_timings'][0]['first_generated_token_ready_ns']/1e9),
            ('first_8_tokens_seconds', lambda r: r['generation_timings'][0]['full_generate_ns']/1e9),
            ('second_8_tokens_seconds', lambda r: r['generation_timings'][1]['full_generate_ns']/1e9)):
            means = {mode: statistics.mean(select(r) for r in paired if r['compiler_mode'] == mode) for mode in (LIVE, WARM)}
            timings[label] = dict(baseline=means[LIVE], candidate=means[WARM], ratio=means[WARM]/means[LIVE])
        record.update(status='qualified_dynamic_PV_warm_deployment_closed', deployment=entry(a.output/'warm-deployment.json'),
                      timing_diagnostic=timings, qualified_DSP_calls=800,
                      newly_executed_DSP_calls=800-record['reused_completed_DSP_calls'],completed_unix=time.time())
    except BaseException as error:
        record.update(status='failed', error=repr(error), completed_unix=time.time()); raise
    finally: save()


if __name__ == '__main__': main()
