"""Run unchanged native generation with an experimental cache-hit host worker.

Uses the exact existing board deployment, loader, math, input packer and oracle.
Only the host worker command changes. Results remain diagnostic until a paired
startup experiment and independent full closure, not a production switch.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from build import ROOT, entry, sha, write
from compare_performance import compare
from revisit_native_startup import startup_summary
from run_regression import replace, require_configured_python
from terminal_reports import read_terminal_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-native','oracle','numerical-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--name',required=True);parser.add_argument('--deadline-unix',type=int,required=True)
    parser.add_argument('--worker-mode',choices=('original','session'),default='session')
    args=parser.parse_args()
    git=lambda root,*words:subprocess.check_output(['git','-C',str(root),*words],text=True).strip()
    if (not args.output.is_absolute() or args.output.exists() or git(ROOT,'status','--porcelain')
            or not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.name)
            or args.deadline_unix-time.time()<1800):
        raise ValueError('Fresh admitted diagnostic, clean source and full startup budget required')
    read=lambda path:json.loads(Path(path).read_text())
    config=read(args.config);require_configured_python(config);board=config['board']
    base=read(args.baseline_native/'result.json');prior=read(args.baseline_native/'native-result.json')
    oracle=read(args.oracle/'native-result.json');accepted=read(args.oracle/'acceptance.json')
    if (base['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
            or accepted['numerical_acceptance'] is not True
            or base['oracle_acceptance_sha256']!=sha(args.oracle/'acceptance.json')
            or base['numerical_source_revision']!=accepted['numerical_revision']
            or git(args.numerical_source,'rev-parse','HEAD')!=accepted['numerical_revision']
            or git(args.numerical_source,'status','--porcelain')
            or prior['DSP_calls']!=16 or len(prior['requests'])!=2):
        raise ValueError('Own original startup16 baseline and accepted full-packet oracle required')
    compare(oracle,prior)
    # This diagnostic needs a complete startup ledger. Reject a legacy oracle
    # before opening a mailbox or starting ARM/DSP work, rather than discovering
    # the missing observation after a successful sixteen-call session.
    prior_startup = startup_summary(prior)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target']]
    if subprocess.run(ssh+['pgrep -x buffer_stream'],capture_output=True).returncode!=1:
        raise ValueError('Exclusive board DSP owner required')
    old_worker=list(base['worker']);old_board=list(base['board'])
    native_source=old_board[4]
    if subprocess.check_output(ssh+[shlex.join(['sha256sum',native_source])],text=True).split()[0]!=prior['source_sha256']:
        raise ValueError('Exact already tested native board source required')
    host_mailbox=Path(old_worker[old_worker.index('--mailbox')+1]).parent/('live-'+args.name+'-mailbox')
    board_mailbox=Path(old_board[old_board.index('--compiler-mailbox')+1]).parent/host_mailbox.name
    replace(old_worker,'--mailbox',host_mailbox)
    worker=old_worker if args.worker_mode=='original' else [old_worker[0],str(Path(__file__).with_name('native_compile_session_worker.py')),
        '--numerical-source',str(args.numerical_source),'--numerical-revision',accepted['numerical_revision'],
        '--deadline-unix',str(args.deadline_unix),*old_worker[2:]]
    native_output=Path(old_board[old_board.index('--output')+1]).parent/args.name
    for flag,value in (('--output',native_output),('--workdir','/dev/shm/qwen3-arm-dynamic-'+args.name),
                       ('--compiler-mailbox',board_mailbox)):
        replace(old_board,flag,value)
    # The immutable native entry itself takes this DSP lock for buffer_stream.
    # Holding the same lock around Python would make that inner acquisition fail.
    command=old_board
    args.output.mkdir();host_mailbox.mkdir(exist_ok=False)
    record=dict(status='running',source_revision=git(ROOT,'rev-parse','HEAD'),source_sha256=sha(__file__),
        worker_source=entry(Path(worker[1])),worker_mode=args.worker_mode,
        baseline=entry(args.baseline_native/'result.json'),oracle=entry(args.oracle/'acceptance.json'),
        worker=worker,board=command,host_mailbox=str(host_mailbox),native_output=str(native_output),
        original_board_source_sha256=prior['source_sha256'],new_model_PRIMARY_compilations=0,
        diagnostic_only=True,model_inference_acceptance=False,performance_acceptance=False,
        production_configuration_changed=False,TUI_startup_claimed=False,
        original_model_loader_and_all_parameter_checks_each_request=True)
    def save():write(args.output/'result.json',record)
    save();wp=None;bp=None
    def interrupted(signum,frame):
        raise InterruptedError('Owned native coordinator interrupted: '+str(signum))
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,interrupted)
    env=dict(os.environ,PYTHONPATH=str(args.numerical_source/'qwen3/runtime/arm')+':'+str(args.numerical_source),
             LD_LIBRARY_PATH=':'.join(config['library_path']),PYTHONDONTWRITEBYTECODE='1')
    try:
        with (args.output/'worker.log').open('xb') as wl,(args.output/'board.log').open('xb') as bl:
            wp=subprocess.Popen(worker,stdout=wl,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            bp=subprocess.Popen(ssh+[shlex.join(command)],stdout=bl,stderr=subprocess.STDOUT,start_new_session=True)
            record['active_processes']=dict(worker=wp.pid,board_ssh=bp.pid,coordinator=os.getpid());save()
            brc=bp.wait(timeout=max(1,args.deadline_unix-time.time()-60))
            if brc:os.killpg(wp.pid,signal.SIGTERM)
            wrc=wp.wait(timeout=35)
        write(args.output/'run-exit.json',dict(board_returncode=brc,worker_returncode=wrc))
        if brc or wrc:raise ValueError('Actual native or session worker failed')
        run,run_sha=read_terminal_report(ssh,native_output/'result.json',args.output/'native-result.json')
        if (run['status']!='ARM_native_dynamic_compiler_handoff_generate_completed'
                or run['process_exit_code']!=0 or run['CPU_model_fallback'] is not False
                or run['DSP_calls']!=16 or run['frontend_invocations']!=0 or run['machine_code_compilations']!=0
                or run['source_sha256']!=prior['source_sha256'] or run['input_packer']!=prior['input_packer']):
            raise ValueError('Exact unchanged native16, zero new frontend/model compilation required')
        compare(oracle,run)
        if [r['terminal_observation'] for r in oracle['requests']]!=[r['terminal_observation'] for r in run['requests']]:
            raise ValueError('Original native stopping behavior changed')
        sys.path[:0]=[str(args.numerical_source/'qwen3/runtime/arm'),str(args.numerical_source)]
        from audit_arm_dynamic_warm_run import audit_compile_handoff
        chain=audit_compile_handoff(run,host_mailbox,rebuilt=True)
        expected={tuple(row['contract']['variables']['query']):row['DSO_sha256'] for row in prior['captures']}
        actual={tuple(row['contract']['variables']['query']):row['DSO_sha256'] for row in run['captures']}
        if actual!=expected:
            raise ValueError('Original two full model libraries changed')
        record.update(status='host_worker_native16_full_packet_oracle_and_compile_chain_matched_DIAGNOSTIC',
            DSP_calls=16,native_sha256=run_sha,complete_compile_chain_rehashed=chain,
            full_original_packet_tokens_and_three_FNV_matched=True,
            startup=startup_summary(run),original_startup=prior_startup,completed_unix=time.time())
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        for process in (bp,wp):
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        save()


if __name__=='__main__':main()
