#!/usr/bin/env python3
"""Deploy a clean checkout and run cold ARM capture -> compiler -> DSP -> controls.

All machine caches and deployment names must be new. Numerical acceptance is
recorded only by the independent raw-packet audit after both references finish.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from build import ROOT, HERE, entry, sha, write, verify_dependencies
from verify_board import inspect
from run_regression import require_configured_python
from terminal_reports import read_terminal_report
from bounded_loader_policy import configured_policy, apply_arguments, verify_native_layout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--specialize-floats', action='store_true')
    args = p.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.name):
        raise ValueError('A simple unique deployment name is required')
    c = json.loads(args.config.read_text())
    loader_policy = configured_policy(c)
    require_configured_python(c)
    b = c['board']
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', b['ssh_target']]
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Run from a clean Git checkout; generated files belong outside it')
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    sdk_lock = json.loads((HERE/'dependencies.lock.json').read_text())
    checked = verify_dependencies(c, sdk_lock)
    write(output/'SDK-verification.json', dict(passed=True, files=len(checked),
        lock_sha256=sha(HERE/'dependencies.lock.json')))
    # Both cache classes deliberately require existing, absolute directories.
    # Initialize fresh roots and validate local dependencies before ARM startup.
    frontend = output/'frontend-cache'
    machine = output/'machine-cache'
    frontend.mkdir()
    machine.mkdir()
    scripts = ROOT/'qwen3/runtime/arm'
    sys.path[:0] = [str(scripts), str(ROOT)]
    from hexagon_captured_linalg_cache import CapturedLinalgCache
    from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
    recipe = entry(args.build/'machine-recipe.json')
    CapturedLinalgCache(frontend)
    CapturedQwenMachineCache(machine, recipe['path'], recipe['sha256'], frontend)
    if list(frontend.iterdir()) or list(machine.iterdir()):
        raise ValueError('Cold validation requires empty caches')
    actual_framework = inspect(c)
    expected_framework = json.loads((HERE/'board-framework.lock.json').read_text())
    write(output/'board-framework-verification.json', dict(passed=actual_framework == expected_framework,
                                                         actual=actual_framework))
    if actual_framework != expected_framework:
        raise ValueError('Pinned ARM W8A8 framework/source dependencies changed')
    observations = Path(c['observations'])
    board_observations = Path(b['observations'])
    mailbox_name = 'live-'+args.name+'-mailbox'
    mailbox = observations/mailbox_name
    mailbox.mkdir(exist_ok=False)
    deployment = board_observations/(args.name+'-sources')
    deploy = ['mkdir', str(deployment)]
    subprocess.run(ssh+[shlex.join(deploy)], check=True)
    archive = subprocess.Popen(['git', '-C', str(ROOT), 'archive', revision], stdout=subprocess.PIPE)
    with archive.stdout:
        extract = subprocess.run(ssh+[shlex.join(['tar', '-C', str(deployment), '-xf', '-'])],
                                 stdin=archive.stdout)
    if archive.wait() or extract.returncode:
        raise RuntimeError('Clean source deployment failed')
    catalog_sha = sha(c['catalog'])
    worker = [c['python'], str(scripts/'serve_qwen_compile_mailbox.py'),
        '--mailbox', str(mailbox), '--trusted-local-producer', '--until-session-close',
        '--timeout', '3600', '--context', c['context'], '--catalog', c['catalog'],
        '--catalog-sha256', catalog_sha, '--frontend-cache', str(output/'frontend-cache'),
        '--machine-cache', str(output/'machine-cache'), '--machine-recipe', recipe['path'],
        '--machine-recipe-sha256', recipe['sha256']]
    board = ['env', 'PYTHONPATH='+b['pythonpath'], b['python'],
        str(deployment/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py'),
        '--catalog', b['catalog'], '--catalog-sha256', catalog_sha,
        '--context', b['context'], '--hf', b['hf'], '--decompositions', b['decompositions'],
        '--prompts-json', json.dumps(c['prompts'], ensure_ascii=False),
        '--max-new-tokens', str(c.get('max_new_tokens', 4)), '--chunk-size', '8',
        '--workdir', '/dev/shm/qwen3-arm-dynamic-'+args.name,
        '--output', str(board_observations/args.name), '--packet-recording', 'full',
        '--local-packet-transport', 'shared-memory', '--output-decoder', 'owned-once',
        '--compiler-mailbox', str(board_observations/mailbox_name)]
    if args.specialize_floats:board.append('--specialize-floats')
    if loader_policy:
        board = apply_arguments(board, loader_policy)
    write(output/'commands.json', dict(revision=revision, source_root=str(ROOT),
        source_deployment=str(deployment), worker=worker, board=board, recipe=recipe,
        frontend_cache_initial_entries=[], machine_cache_initial_entries=[], graph_whitelist=False))
    env = dict(os.environ, PYTHONPATH=str(scripts)+':'+str(ROOT),
               LD_LIBRARY_PATH=':'.join(c['library_path']), PYTHONDONTWRITEBYTECODE='1')
    with (output/'worker.log').open('xb') as wlog, (output/'board.log').open('xb') as blog:
        wp = subprocess.Popen(worker, env=env, stdout=wlog, stderr=subprocess.STDOUT, start_new_session=True)
        bp = subprocess.Popen(ssh+[shlex.join(board)], stdout=blog, stderr=subprocess.STDOUT, start_new_session=True)
        write(output/'active-processes.json', dict(coordinator=os.getpid(), worker=wp.pid, board_ssh=bp.pid))
        board_rc = bp.wait()
        try:
            worker_rc = wp.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if board_rc == 0:
                raise
            os.killpg(wp.pid, signal.SIGTERM)
            worker_rc = wp.wait(timeout=30)
    write(output/'run-exit.json', dict(board_returncode=board_rc, worker_returncode=worker_rc,
        numerical_acceptance=False))
    if board_rc or worker_rc:
        raise RuntimeError('Cold native session failed; logs and artifacts retained')
    run = observations/args.name/'result.json'
    native,run_sha=read_terminal_report(ssh,board_observations/args.name/'result.json',
        output/'native-result.json')
    if loader_policy:
        verify_native_layout(native)
    controls = ['env', 'PYTHONPATH='+b['pythonpath'], 'PYTHONDONTWRITEBYTECODE=1', b['python'],
        str(deployment/'tools/qwen3/reference_native_handoff_arm.py'), '--sources', str(deployment),
        '--run', str(board_observations/args.name/'result.json'), '--run-sha256', run_sha,
        '--context', b['context'], '--hf', b['hf'],
        '--output', str(board_observations/(args.name+'-ARM-controls'))]
    write(output/'controls-command.json', controls)
    with (output/'controls.log').open('xb') as log:
        subprocess.run(ssh+[shlex.join(controls)], stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = [c['python'], str(HERE/'audit_native_run.py'), '--sources', str(ROOT),
        '--observations', str(observations), '--board-observations', str(board_observations),
        '--hf', c['hf'], '--output', str(output/'audit'), '--name', args.name,
        '--ssh-target', b['ssh_target']]
    write(output/'audit-command.json', audit)
    with (output/'audit.log').open('xb') as log:
        subprocess.run(audit, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output/'audit'/(args.name+'-independent-audit.json')).read_text())
    if not result['numerical_acceptance']:
        raise RuntimeError('Independent numerical audit failed')
    write(output/'acceptance.json', dict(status='clean_checkout_cold_compile_ARM_W8A8_acceptance_passed',
        revision=revision, recipe=recipe, audit=entry(output/'audit'/(args.name+'-independent-audit.json'))))
    print('Clean checkout cold compile and ARM W8A8 numerical acceptance passed', flush=True)


if __name__ == '__main__':
    main()
