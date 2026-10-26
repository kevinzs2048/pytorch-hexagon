"""Exercise PID pinning and unconditional resume with real disposable processes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = ROOT/'tools/qwen3/run_cold_boundary_probes.py'
sys.path.insert(0,str(ROOT/'tools/qwen3'))
spec=importlib.util.spec_from_file_location('cold_boundary_probe',SCHEDULER)
boundary=importlib.util.module_from_spec(spec);spec.loader.exec_module(boundary)


def state(pid):
    return (Path('/proc')/str(pid)/'stat').read_text().split(') ', 1)[1].split()


@pytest.fixture
def owned_controller(tmp_path):
    # The scheduler requires committed source. Exercise the current files in a
    # disposable clean repository, independent of edits in the developer tree.
    checkout = tmp_path/'scheduler'
    scripts = checkout/'tools/qwen3'
    scripts.mkdir(parents=True)
    for name in ('build.py', 'run_cold_boundary_probes.py'):
        shutil.copyfile(ROOT/'tools/qwen3'/name, scripts/name)
    for command in (['init', '-q'], ['add', 'tools'],
                    ['-c', 'user.name=Qwen test', '-c', 'user.email=test@example.invalid',
                     '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'Test fixture']):
        subprocess.run(['git', '-C', str(checkout), *command], check=True, capture_output=True)
    run = tmp_path/'run'
    run.mkdir()
    (run/'cold').mkdir()
    (run/'result.json').write_text(json.dumps(dict(status='running_cold', steps=[])))
    # The child deliberately finishes without cold acceptance. No SSH or DSP
    # operation is reachable: the scheduler must fail and resume the parent.
    child_script = tmp_path/'run_native.py'
    child_script.write_text('import time\ntime.sleep(2)\n')
    controller = tmp_path/'run_helper_effects_iteration.py'
    controller.write_text('''import json,subprocess,sys,time
from pathlib import Path
run=Path(sys.argv[sys.argv.index('--output')+1])
child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run_native.py')),'--output',str(run/'cold')])
(run/'child.json').write_text(json.dumps({'pid':child.pid}))
child.wait()
(run/'resumed.marker').write_text('child reaped after parent resumed')
time.sleep(20)
''')
    command = [sys.executable, str(controller), '--output', str(run)]
    parent = subprocess.Popen(command)
    limit = time.monotonic()+10
    while not (run/'child.json').exists():
        if time.monotonic() >= limit:
            raise TimeoutError('Disposable controller failed to start')
        time.sleep(0.01)
    child = json.loads((run/'child.json').read_text())['pid']
    process_path, command_path = tmp_path/'process.json', tmp_path/'command.json'
    process_path.write_text(json.dumps(dict(status='running', controller_pid=parent.pid, process_start_tick=state(parent.pid)[19])))
    command_path.write_text(json.dumps(command))
    config, plan = tmp_path/'config.json', tmp_path/'plan.json'
    config.write_text(json.dumps(dict(python=sys.executable, board={'ssh_target':'invalid-unused-board'})))
    plan.write_text(json.dumps(dict(status='staged_NOT_executed')))
    output = tmp_path/'probe-result'
    argv = [sys.executable, str(scripts/SCHEDULER.name), '--config', str(config), '--controller-process', str(process_path),
        '--controller-command', str(command_path), '--controller-run', str(run), '--cold-child-pid', str(child),
        '--plan', str(plan), '--deadline-unix', str(int(time.time()+20)), '--output', str(output)]
    try:
        yield dict(parent=parent, child=child, argv=argv, process=process_path, run=run, output=output)
    finally:
        if parent.poll() is None:
            os.kill(parent.pid, 18)  # SIGCONT even if an assertion failed.
            parent.terminate()
            parent.wait(timeout=5)
        try:
            os.kill(child, 15)
        except ProcessLookupError:
            pass


def test_failed_cold_acceptance_always_resumes_original_parent(owned_controller):
    owned = owned_controller
    result = subprocess.run(owned['argv'], capture_output=True, timeout=15,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode
    record = json.loads((owned['output']/'result.json').read_text())
    assert record['parent_paused_before_next_stage'] is True
    assert record['original_controller_resumed'] is True
    assert record['status'] == 'failed' and record['rows'] == []
    assert 'acceptance.json' in record['error']
    limit = time.monotonic()+5
    while not (owned['run']/'resumed.marker').exists():
        assert time.monotonic() < limit
        time.sleep(0.01)
    assert state(owned['parent'].pid)[0] != 'T'


def test_wrong_process_start_tick_is_rejected_before_any_signal(owned_controller):
    owned = owned_controller
    record = json.loads(owned['process'].read_text())
    record['process_start_tick'] = '0'
    owned['process'].write_text(json.dumps(record))
    result = subprocess.run(owned['argv'], capture_output=True, timeout=5,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode and b'Only pinned active helper' in result.stderr
    assert not owned['output'].exists()
    assert state(owned['parent'].pid)[0] != 'T'


@pytest.mark.parametrize('ratio,positive',[(.98,True),(1.001,False)])
def test_native_gain_boundary_requires_actual_four_ordered_sessions(tmp_path,ratio,positive):
    for index,kind in enumerate(('baseline','candidate','candidate','baseline')):
        folder=tmp_path/('native-abba-'+str(index)+'-'+kind);folder.mkdir()
        native=dict(DSP_calls=16,process_exit_code=0,CPU_model_fallback=False)
        native_path=folder/'native-result.json';native_path.write_text(json.dumps(native))
        (folder/'run-exit.json').write_text(json.dumps(dict(board_returncode=0,worker_returncode=0)))
        factor=ratio if kind=='candidate' else 1
        proof=dict(status='native_generation_timing_verified_against_preaccepted_full_packet_oracle',DSP_calls=16,
            inline_CPU_observer=False,full_input_output_model_token_FNV_equality=True,
            native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),measurement={'query_groups':{
                '1':{'median_seconds':{'DSP_kernel_seconds':6*factor,'ARM_callback_seconds':9*factor}},
                '32':{'median_seconds':{'DSP_kernel_seconds':20,'ARM_callback_seconds':23}}}})
        (folder/'result.json').write_text(json.dumps(proof))
    result=boundary.final_native_acceptance(tmp_path)
    assert result['positive_native_model_gain'] is positive and result['DSP_calls']==64
    assert result['model_inference_acceptance'] is False
    (tmp_path/'native-abba-3-baseline/native-result.json').write_text('{}')
    with pytest.raises(ValueError,match='Completed exact native ABBA64'):
        boundary.final_native_acceptance(tmp_path)
