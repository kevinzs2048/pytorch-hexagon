#!/usr/bin/env python3
"""Repeat native generation/state checks using an accepted cold run's caches.

The accepted numerical checkout, deployment and recipe remain authoritative.
This tool changes request parameters only; every actual capture still goes to
the compiler and every retained packet goes through both ARM controls/audit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def replace(command, option, value):
    if command.count(option) != 1:
        raise ValueError('Expected exactly one command option: '+option)
    command[command.index(option)+1] = str(value)


def require_configured_python(config):
    if Path(sys.executable).resolve() != Path(config['python']).resolve():
        raise ValueError('Use the pinned coordinator Python: ' + config['python'])


def accepted_base(path):
    """Admit only a cold run whose independent acceptance is still intact."""
    commands = json.loads((path/'commands.json').read_text())
    acceptance = json.loads((path/'acceptance.json').read_text())
    audit_record = acceptance['audit']
    audit = json.loads(Path(audit_record['path']).read_text())
    if (acceptance['status'] != 'clean_checkout_cold_compile_ARM_W8A8_acceptance_passed'
            or sha(audit_record['path']) != audit_record['sha256']
            or audit.get('numerical_acceptance') is not True
            or acceptance['revision'] != commands['revision']
            or acceptance['recipe'] != commands['recipe']
            or sha(commands['recipe']['path']) != commands['recipe']['sha256']):
        raise ValueError('Accepted cold run, numerical audit or recipe changed')
    root = Path(commands['source_root'])
    if (subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip() != commands['revision']
            or subprocess.check_output(['git','-C',str(root),'status','--porcelain'], text=True).strip()):
        raise ValueError('Accepted numerical checkout must be unchanged and clean')
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--accepted-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--prompts-json', required=True)
    p.add_argument('--max-new-tokens', type=int, required=True)
    p.add_argument('--chunk-size', type=int, default=8)
    p.add_argument('--chat-template', action='store_true')
    p.add_argument('--require-warm', action='store_true', help='Reject observed recompilation, never alter cache keys')
    p.add_argument('--require-eos', action='store_true', help='Require an actually generated native EOS before the token budget')
    p.add_argument('--observations', type=Path, help='Optional host view of a dedicated evidence parent')
    p.add_argument('--board-observations', type=Path, help='Matching board evidence parent; may be a mounted volume')
    a = p.parse_args()
    if (not a.name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for x in a.name)
            or not a.output.is_absolute() or a.output.exists() or a.output.is_symlink()
            or not 1 <= a.max_new_tokens <= 1024 or not 1 <= a.chunk_size <= 32
            or bool(a.observations) != bool(a.board_observations)):
        p.error('Unique name, fresh absolute output, bounded generation and paired evidence paths required')
    prompts = json.loads(a.prompts_json)
    if (not isinstance(prompts, list) or not 1 <= len(prompts) <= 8
            or any(type(x) is not str or not x or len(x.encode('utf-8')) > 65536 for x in prompts)):
        p.error('One to eight bounded nonempty prompts required')
    base = accepted_base(a.accepted_run)
    c = json.loads(a.config.read_text()); b = c['board']; root = Path(base['source_root'])
    require_configured_python(c)
    O = a.observations or Path(c['observations'])
    BO = a.board_observations or Path(b['observations'])
    if not O.is_absolute() or not BO.is_absolute() or not O.is_dir():
        raise ValueError('Existing absolute paired evidence directories required')
    # The cold commands retain all compiler/cache and model dependency paths.
    # Config supplies connection/environment settings, not alternate numerics.
    ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',b['ssh_target']]
    env = dict(os.environ, PYTHONPATH=str(root/'qwen3/runtime/arm')+':'+str(root),
               LD_LIBRARY_PATH=':'.join(c['library_path']), PYTHONDONTWRITEBYTECODE='1')
    a.output.mkdir(parents=True)
    mailbox = O/('live-'+a.name+'-mailbox'); mailbox.mkdir(exist_ok=False)
    worker = list(base['worker']); replace(worker,'--mailbox',mailbox)
    board = list(base['board'])
    for flag,value in (('--prompts-json',a.prompts_json),('--max-new-tokens',a.max_new_tokens),
            ('--chunk-size',a.chunk_size),('--output',BO/a.name),
            ('--workdir','/dev/shm/qwen3-arm-dynamic-'+a.name),('--compiler-mailbox',BO/mailbox.name)):
        replace(board,flag,value)
    if a.chat_template: board.append('--chat-template')
    write(a.output/'commands.json',dict(coordinator_source_sha256=sha(__file__),
        source_revision=base['revision'],source_root=str(root),
        accepted_run=str(a.accepted_run),accepted_audit=json.loads((a.accepted_run/'acceptance.json').read_text())['audit'],
        recipe=base['recipe'],worker=worker,board=board,graph_whitelist=False,
        require_warm=a.require_warm,require_eos=a.require_eos))
    write(a.output/'progress.json',dict(stage='native_DSP'))
    with (a.output/'worker.log').open('xb') as wlog,(a.output/'board.log').open('xb') as blog:
        wp = subprocess.Popen(worker,env=env,stdout=wlog,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            bp = subprocess.Popen(ssh+[shlex.join(board)],stdout=blog,stderr=subprocess.STDOUT,start_new_session=True)
            write(a.output/'active-processes.json',dict(coordinator=os.getpid(),worker=wp.pid,board_ssh=bp.pid))
            brc = bp.wait()
            if brc: os.killpg(wp.pid,signal.SIGTERM)
            wrc = wp.wait(timeout=35)
        finally:
            if 'bp' in locals() and bp.poll() is None:
                os.killpg(bp.pid,signal.SIGTERM); bp.wait(timeout=35)
            if wp.poll() is None:
                os.killpg(wp.pid,signal.SIGTERM); wp.wait(timeout=35)
    write(a.output/'run-exit.json',dict(board_returncode=brc,worker_returncode=wrc))
    if brc or wrc: raise RuntimeError('Native DSP session failed; evidence retained')
    run = O/a.name/'result.json'; record = json.loads(run.read_text())
    if a.require_warm and (record['frontend_invocations'] or record['machine_code_compilations']):
        raise RuntimeError('Actual cache miss observed; retained, not accepted as a warm run')
    deployment = Path(base['source_deployment'])
    controls = ['env','PYTHONPATH='+b['pythonpath'],'PYTHONDONTWRITEBYTECODE=1',b['python'],
        str(deployment/'tools/qwen3/reference_native_handoff_arm.py'),'--sources',str(deployment),
        '--run',str(BO/a.name/'result.json'),'--run-sha256',sha(run),
        '--context',b['context'],'--hf',b['hf'],'--output',str(BO/(a.name+'-ARM-controls'))]
    write(a.output/'controls-command.json',controls)
    write(a.output/'progress.json',dict(stage='ARM_same_input_and_native_controls',DSP_calls=record['DSP_calls']))
    with (a.output/'controls.log').open('xb') as log:
        subprocess.run(ssh+[shlex.join(controls)],stdout=log,stderr=subprocess.STDOUT,check=True)
    audit = [c['python'],str(root/'tools/qwen3/audit_native_run.py'),'--sources',str(root),
        '--observations',str(O),'--board-observations',str(BO),'--hf',c['hf'],
        '--output',str(a.output/'audit'),'--name',a.name,'--ssh-target',b['ssh_target']]
    write(a.output/'audit-command.json',audit)
    write(a.output/'progress.json',dict(stage='independent_full_packet_audit'))
    with (a.output/'audit.log').open('xb') as log:
        subprocess.run(audit,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = a.output/'audit'/(a.name+'-independent-audit.json')
    if json.loads(result.read_text())['numerical_acceptance'] is not True:
        raise RuntimeError('Independent numerical audit failed')
    eos_requests = [r['index'] for r in record['requests']
        if r['terminal_observation']['last_token_is_native_eos']
        and not r['terminal_observation']['reached_requested_token_limit']]
    if a.require_eos and not eos_requests:
        raise RuntimeError('No actual early native EOS; numerical evidence retained, EOS branch not accepted')
    write(a.output/'acceptance.json',dict(status='native_state_regression_passed',
        numerical_source_revision=base['revision'],audit=dict(path=str(result),sha256=sha(result)),
        DSP_calls=record['DSP_calls'],frontend_invocations=record['frontend_invocations'],
        machine_code_compilations=record['machine_code_compilations'],early_native_EOS_requests=eos_requests))
    write(a.output/'progress.json',dict(stage='accepted'))
    print('Native state regression independently accepted',flush=True)


if __name__ == '__main__':
    main()
