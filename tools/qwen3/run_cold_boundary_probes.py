"""Run bounded recorded probes after a pinned cold child finishes and exits.

Pause only the explicit iteration parent. Its current cold child keeps running;
no probe starts until full cold acceptance, child exit and no board DSP owner.
The parent resumes in finally, before its warm/native timing runs can begin.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from build import ROOT, sha, write


def identity(pid):
    root = Path('/proc') / str(pid)
    fields = (root/'stat').read_text().split(') ', 1)[1].split()
    argv = (root/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    return dict(start_tick=fields[19], parent=int(fields[1]), state=fields[0], argv=argv)


def cold_acceptance(cold):
    acceptance = json.loads((cold/'acceptance.json').read_text())
    audit_path = Path(acceptance['audit']['path'])
    if sha(audit_path) != acceptance['audit']['sha256']:
        raise ValueError('Completed cold audit changed')
    audit = json.loads(audit_path.read_text())
    native = json.loads((cold/'native-result.json').read_text())
    exit_record = json.loads((cold/'run-exit.json').read_text())
    if (acceptance['status'] != 'clean_checkout_cold_compile_ARM_W8A8_acceptance_passed'
            or audit['status'] != 'framework_native_numerical_acceptance_passed'
            or not audit['numerical_acceptance'] or audit['run_sha256'] != sha(cold/'native-result.json')
            or native['DSP_calls'] != 9 or native['process_exit_code'] or native['CPU_model_fallback']
            or exit_record['board_returncode'] or exit_record['worker_returncode']):
        raise ValueError('Complete original cold9 and full numerical audit required')
    return dict(path=str(cold/'acceptance.json'), sha256=sha(cold/'acceptance.json'), DSP_calls=9)


def final_native_acceptance(run):
    from run_helper_effects_iteration import gain
    rows=[];queries={}
    for index,kind in enumerate(('baseline','candidate','candidate','baseline')):
        folder=run/('native-abba-'+str(index)+'-'+kind)
        proof=json.loads((folder/'result.json').read_text())
        native=json.loads((folder/'native-result.json').read_text())
        if (proof['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
                or proof['DSP_calls']!=16 or proof['inline_CPU_observer'] is not False
                or proof['full_input_output_model_token_FNV_equality'] is not True
                or proof['native_sha256']!=sha(folder/'native-result.json')
                or json.loads((folder/'run-exit.json').read_text())!={'board_returncode':0,'worker_returncode':0}
                or native['DSP_calls']!=16 or native['process_exit_code'] or native['CPU_model_fallback']):
            raise ValueError('Completed exact native ABBA64 and normal owner exits required')
        for query,measure in proof['measurement']['query_groups'].items():
            values=measure['median_seconds'];queries.setdefault(query,[]).append(dict(kind=kind,
                kernel_seconds=values['DSP_kernel_seconds'],callback_seconds=values['ARM_callback_seconds']))
        rows.append(dict(kind=kind,proof=dict(path=str(folder/'result.json'),sha256=sha(folder/'result.json'))))
    measurements={}
    for query,values in queries.items():
        if len(values)!=4:raise ValueError('Native ABBA query coverage differs')
        measurements[query]={}
        for key in ('kernel_seconds','callback_seconds'):
            old=statistics.mean(r[key] for r in values if r['kind']=='baseline')
            new=statistics.mean(r[key] for r in values if r['kind']=='candidate')
            if old<=0 or new<=0:raise ValueError('Actual positive native timings required')
            measurements[query][key]=dict(baseline=old,candidate=new,ratio=new/old)
    positive=gain(measurements['1']) and all(r['ratio']<=1.02 for q,t in measurements.items() if q!='1' for r in t.values())
    return dict(status='actual_native_ABBA64_revisited_before_combined_state_regression',DSP_calls=64,
        positive_native_model_gain=positive,rows=rows,measurements=measurements,model_inference_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'controller-process', 'controller-command', 'controller-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cold-child-pid', type=int)
    parser.add_argument('--boundary',choices=('cold','native-abba-final'),default='cold')
    parser.add_argument('--superseding-recipe',type=Path)
    parser.add_argument('--superseding-recipe-sha256')
    parser.add_argument('--superseding-model-build',type=Path)
    parser.add_argument('--superseding-model-build-sha256')
    parser.add_argument('--plan', type=Path, action='append', required=True)
    parser.add_argument('--deadline-unix', type=int, required=True)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.exists() or not 1 <= len(args.plan) <= 3
            or not 0 < args.deadline_unix-time.time() <= 3600):
        parser.error('Fresh bounded cold-boundary probe plan required')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Clean committed boundary scheduler required')
    process = json.loads(args.controller_process.read_text())
    command = json.loads(args.controller_command.read_text())
    pid = process['controller_pid']
    parent = identity(pid)
    cold = args.controller_run/'cold'
    stage = json.loads((args.controller_run/'result.json').read_text())
    if (process['status'] != 'running' or parent['start_tick'] != process['process_start_tick']
            or parent['argv'] != command or Path(command[1]).name != 'run_helper_effects_iteration.py'
            or command.count('--output') != 1 or Path(command[command.index('--output')+1]) != args.controller_run):
        raise ValueError('Only pinned active helper iteration parent may be scheduled')
    native_boundary=args.boundary=='native-abba-final'
    multi_followup=bool(args.superseding_model_build)
    if native_boundary:
        if args.cold_child_pid:
            raise ValueError('Final native boundary discovers its actual current child')
        if multi_followup:
            if (args.superseding_recipe or args.superseding_recipe_sha256
                    or not args.superseding_model_build_sha256
                    or sha(args.superseding_model_build)!=args.superseding_model_build_sha256):
                raise ValueError('Pinned multi-token model build required without a recipe substitution')
            superseding=json.loads(args.superseding_model_build.read_text())
            if (superseding['status']!='exact_query2_to_32_integer_model_built_NOT_device_or_state_acceptance'
                    or superseding.get('query_bounds')!=[2,32] or not superseding['guarded_copy_included']
                    or not superseding['all_surviving_original_operations_properties_and_SSA_preserved']
                    or not superseding['original_caller_unchanged'] or not superseding['constant_objects_identical']
                    or superseding['FP_evaluation_order_changed'] or not superseding['test_hooks_absent_from_production_runtime']
                    or sha(superseding['library']['path'])!=superseding['library']['sha256']):
                raise ValueError('Original bounded multi-token integer/copy candidate required')
            if command.count('--recipe-build')!=1:
                raise ValueError('Current combined query1 recipe must remain explicitly bound')
            current_recipe=Path(command[command.index('--recipe-build')+1])/'machine-recipe.json'
            current_recipe_sha256=sha(current_recipe)
            q1=json.loads(current_recipe.read_text())
            if q1.get('guarded_copy_query1') is not True or q1.get('silu_effects_query1') is not True:
                raise ValueError('Only the current combined query1 candidate may share final states with multi-token work')
        else:
            if (args.superseding_model_build_sha256 or not args.superseding_recipe or not args.superseding_recipe_sha256
                    or sha(args.superseding_recipe)!=args.superseding_recipe_sha256):
                raise ValueError('Explicit combined helper/copy recipe required for final native boundary')
            superseding=json.loads(args.superseding_recipe.read_text())
            if superseding.get('guarded_copy_query1') is not True or superseding.get('silu_effects_query1') is not True:
                raise ValueError('Only combined, already probed helper/copy follow-up may replace duplicate long states')
        while stage['status']!='running_native-abba-3':
            if (time.time()>=args.deadline_unix or identity(pid)['start_tick']!=parent['start_tick']
                    or not stage['status'].startswith('running_native-abba-')):
                raise ValueError('Bounded final-native boundary advanced or was not active')
            time.sleep(5);stage=json.loads((args.controller_run/'result.json').read_text())
        candidates=[]
        for root in Path('/proc').iterdir():
            if not root.name.isdecimal():continue
            try:
                row=identity(int(root.name))
                if row['parent']==pid and len(row['argv'])>1 and Path(row['argv'][1]).name=='benchmark_accepted_generation.py':
                    candidates.append((int(root.name),row))
            except (FileNotFoundError,ProcessLookupError):pass
        if len(candidates)!=1:raise ValueError('Exactly one current final native benchmark child required')
        child_pid,child=candidates[0]
        child_output=args.controller_run/'native-abba-3-baseline'
        child_script='benchmark_accepted_generation.py';expected_stage='running_native-abba-3'
    else:
        if (not args.cold_child_pid or stage['status']!='running_cold' or stage['steps'] or args.superseding_recipe
                or args.superseding_recipe_sha256 or args.superseding_model_build or args.superseding_model_build_sha256):
            raise ValueError('Only current cold child without a superseding recipe may use the cold boundary')
        child_pid=args.cold_child_pid;child=identity(child_pid)
        child_output=cold;child_script='run_native.py';expected_stage='running_cold'
    if (child['parent'] != pid or Path(child['argv'][1]).name != child_script
            or child['argv'].count('--output') != 1
            or Path(child['argv'][child['argv'].index('--output')+1]) != child_output):
        raise ValueError('Only pinned active iteration and its selected boundary child may be scheduled')
    plans = [dict(path=str(path), sha256=sha(path)) for path in args.plan]
    for path in args.plan:
        if json.loads(path.read_text())['status'] != 'staged_NOT_executed':
            raise ValueError('Fresh explicit staged diagnostic plan required')
    config = json.loads(args.config.read_text())
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', config['board']['ssh_target']]
    args.output.mkdir(parents=True)
    record = dict(status='waiting_for_completed_cold_child', source_sha256=sha(__file__),
        source_revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        controller_pid=pid, controller_start_tick=parent['start_tick'], cold_child_pid=child_pid,boundary=args.boundary,
        cold_child_start_tick=child['start_tick'], plans=plans, deadline_unix=args.deadline_unix, rows=[])
    def save():
        write(args.output/'result.json', record)
    def interrupted(signum, frame):
        raise InterruptedError('Cold boundary interrupted: '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(max(1, int(args.deadline_unix-time.time())))
    paused = False
    superseded=False
    save()
    try:
        if identity(pid)['start_tick'] != parent['start_tick']:
            raise ValueError('Controller process identity changed')
        os.kill(pid, signal.SIGSTOP)
        paused = True
        # Recheck after STOP to catch a cold/warm transition race before pausing.
        if (json.loads((args.controller_run/'result.json').read_text())['status'] != expected_stage
                or identity(child_pid)['start_tick'] != child['start_tick']):
            raise ValueError('Cold boundary advanced before controller pause')
        record['parent_paused_before_next_stage'] = True
        save()
        while True:
            if time.time() >= args.deadline_unix:
                raise TimeoutError('Bounded cold-boundary deadline reached')
            try:
                current = identity(child_pid)
                done = current['state'] == 'Z'
                if current['start_tick'] != child['start_tick']:
                    raise ValueError('Cold child process identity changed')
            except FileNotFoundError:
                done = True
            if done:
                record['cold_acceptance'] = cold_acceptance(cold)
                if native_boundary:record['native_ABBA64']=final_native_acceptance(args.controller_run)
                break
            time.sleep(5)
        owner = subprocess.run(ssh+['pgrep -x buffer_stream'], capture_output=True, timeout=15)
        if owner.returncode != 1:
            raise ValueError('DSP owner remains or owner inspection failed')
        if native_boundary and not record['native_ABBA64']['positive_native_model_gain']:
            record.update(status='native_gain_nonpositive_original_controller_will_handle_deferral',DSP_calls=0)
            return
        for index, row in enumerate(plans):
            if sha(row['path']) != row['sha256']:
                raise ValueError('Pinned stage plan changed')
            folder = args.output/('probe-'+str(index))
            argv = [config['python'], str(ROOT/'tools/qwen3/run_staged_probe.py'), '--config', str(args.config),
                    '--plan', row['path'], '--output', str(folder)]
            record['status'] = 'running_probe_'+str(index)
            save()
            with (args.output/('probe-'+str(index)+'.log')).open('xb') as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            proof = json.loads((folder/'process.json').read_text())
            if result.returncode or proof['status'] != 'complete':
                raise ValueError('Actual boundary probe failed: '+str(index))
            record['rows'].append(dict(plan=row, result=proof['result'], DSP_calls=proof['actual_DSP_calls']))
            save()
        record.update(status='completed_boundary_probes_NOT_native_model_acceptance',
                      DSP_calls=sum(row['DSP_calls'] for row in record['rows']), completed_unix=time.time())
        if native_boundary:
            from summarize_helper_effects import model
            gate_status=('actual_same_input_dynamic_integer_model_full_packet_gate_passed' if multi_followup else
                         'actual_same_input_q1_guarded_copy_model_full_packet_gate_passed')
            gate=next((json.loads(Path(row['result']['path']).read_text()) for row in record['rows']
                if json.loads(Path(row['result']['path']).read_text())['status']==gate_status),None)
            if gate is None:raise ValueError('Complete combined original-input model ABBA gate required')
            gate_proof=model(gate,not multi_followup,multi_followup,multi_followup);record['combined_early_model_gate']=gate_proof
            if multi_followup and (gate['candidate_library_sha256']!=superseding['library']['sha256']
                    or sha(args.superseding_model_build)!=args.superseding_model_build_sha256
                    or sha(current_recipe)!=current_recipe_sha256):
                raise ValueError('Actual model gate candidate or current query1 recipe changed')
            if any(gate_proof['measurements'][k]['ratio']>=0.995 for k in ('kernel_us','callback_seconds')):
                record['combined_gate_nonpositive_original_controller_will_finish_states']=True
                return
            if identity(pid)['start_tick']!=parent['start_tick']:raise ValueError('Controller identity changed before supersession')
            intent=dict(status='combined_helper_copy_positive_gates_long_states_will_run_once',
                native_ABBA64=record['native_ABBA64'],combined_early_model_gate=gate_proof,
                model_inference_acceptance=False,reason='Both mathematical paths remain fixed; validate the combined candidate against the original complete state suite')
            if multi_followup:
                intent.update(status='combined_query1_native_and_multi_integer_copy_gate_states_will_run_once',
                    preserved_query1_recipe=dict(path=str(current_recipe),sha256=sha(current_recipe)),
                    superseding_model_build=dict(path=str(args.superseding_model_build),sha256=args.superseding_model_build_sha256))
            else:
                intent['superseding_recipe']=dict(path=str(args.superseding_recipe),sha256=args.superseding_recipe_sha256)
            write(args.controller_run/'combined-supersession.json',intent)
            # A pending TERM is delivered on CONT before the paused Python
            # parent can launch its next stage. The completed child is untouched.
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);superseded=True
            for _ in range(50):
                try:dead=identity(pid)['state']=='Z'
                except FileNotFoundError:dead=True
                if dead:break
                time.sleep(.1)
            if not dead:raise ValueError('Superseded controller has not exited')
            terminal=json.loads((args.controller_run/'result.json').read_text())
            terminal.update(status=('combined_q1_native_timing_passed_states_superseded_by_multi_integer_copy_candidate' if multi_followup else
                                    'SiLU_native_timing_passed_states_superseded_by_combined_copy_candidate'),
                native_model_timing=record['native_ABBA64']['measurements'],DSP_calls=89,candidate_DSP_calls=57,
                paired_baseline_DSP_calls=32,model_inference_acceptance=False,
                supersession=dict(path=str(args.controller_run/'combined-supersession.json'),sha256=sha(args.controller_run/'combined-supersession.json')))
            write(args.controller_run/'result.json',terminal)
            record.update(status='positive_native_and_combined_gates_original_long_states_superseded',original_controller_superseded=True)
    except BaseException as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        signal.alarm(0)
        if paused and not superseded:
            try:
                if identity(pid)['start_tick'] == parent['start_tick']:
                    os.kill(pid, signal.SIGCONT)
                    record['original_controller_resumed'] = True
                else:
                    record['controller_resume_error'] = 'Process identity changed; no signal sent'
            except BaseException as error:
                record['controller_resume_error'] = str(error)
        save()
    print(record['status'])


if __name__ == '__main__':
    main()
