"""Complete two remaining owned startup arms after a recorded server interruption.

Only the first two already normal, SHA-bound arms are reusable. The interrupted
partial arm is retained and excluded. This continuation keeps the original
deadline and all full-packet checks; it does not claim the old supervisor exited
normally or restart a fresh four-arm experiment with a shorter budget.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

from build import ROOT, entry, sha, write
from run_native_compile_session_abba import summarize_abba


def require_completed_arm(row, deadline):
    for key in ('result','controller'):
        if sha(row[key]['path'])!=row[key]['sha256']:
            raise ValueError('Original normally completed arm changed')
    owner=json.loads(Path(row['controller']['path']).read_text())
    if (owner['status']!='completed' or type(owner['returncode']) is not int or owner['returncode']!=0
            or owner['authorized_deadline_unix']!=deadline or owner['completed_unix']>deadline):
        raise ValueError('Actual normal arm owner within original deadline required')
    result=json.loads(Path(row['result']['path']).read_text())
    if result['worker_mode']!=row['worker_mode'] or result['DSP_calls']!=16:
        raise ValueError('Original complete arm mode and forward count required')
    return result


def require_signal_cleanup_only(original, current):
    old=ast.parse(original);new=ast.parse(current)
    function=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    allowed=ast.parse("""def interrupted(signum,frame):
    raise InterruptedError('Owned native coordinator interrupted: '+str(signum))
for signum in (signal.SIGINT,signal.SIGTERM):
    signal.signal(signum,interrupted)
""").body
    shape=lambda node:ast.dump(node,include_attributes=False)
    pairs=[i for i in range(len(function.body)-1)
           if [shape(n) for n in function.body[i:i+2]]==[shape(n) for n in allowed]]
    if len(pairs)!=1:
        raise ValueError('Exactly the reviewed interruption handler and registration required')
    del function.body[pairs[0]:pairs[0]+2]
    if ast.dump(old,include_attributes=False)!=ast.dump(new,include_attributes=False):
        raise ValueError('Normal coordinator path changed beyond signal cleanup')


def unchanged_normal_coordinator(old_revision):
    relative='tools/qwen3/probe_native_compile_session.py'
    original=subprocess.check_output(['git','-C',str(ROOT),'show',old_revision+':'+relative],text=True)
    require_signal_cleanup_only(original,(ROOT/relative).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','config','baseline-native','oracle','numerical-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--interruption',type=Path,required=True)
    parser.add_argument('--deadline-unix',type=int,required=True);args=parser.parse_args()
    git=lambda *words:subprocess.check_output(['git','-C',str(ROOT),*words],text=True).strip()
    old=json.loads((args.experiment/'result.json').read_text());interruption=json.loads(args.interruption.read_text())
    if (not args.output.is_absolute() or args.output.exists() or git('status','--porcelain')
            or args.deadline_unix-time.time()<2100 or old['status']!='interrupted_by_server_restart'
            or old['deadline_unix']!=args.deadline_unix or len(old['rows'])!=2
            or [r['worker_mode'] for r in old['rows']]!=['original','session']
            or interruption['original_deadline_preserved']!=args.deadline_unix):
        raise ValueError('Recorded interrupted experiment, two normal arms and original remaining budget required')
    unchanged_normal_coordinator(old['source_revision'])
    rows=list(old['rows']);raw=[require_completed_arm(row,args.deadline_unix) for row in rows]
    for result in raw:
        if (result['oracle']['sha256']!=sha(args.oracle/'acceptance.json')
                or result['baseline']['sha256']!=sha(args.baseline_native/'result.json')):
            raise ValueError('Original complete arm baseline/oracle changed')
    revision=git('rev-parse','HEAD');args.output.mkdir()
    record=dict(status='running_continuation_after_recorded_server_restart',source_revision=revision,rows=rows,
        original_interrupted_experiment=entry(args.experiment/'result.json'),interruption=entry(args.interruption),
        original_deadline_preserved=args.deadline_unix,excluded_interrupted_DSP_calls=interruption['excluded_interrupted_DSP_calls'],
        DSP_calls=32,model_inference_acceptance=False,performance_acceptance=False,
        production_configuration_changed=False,TUI_startup_claimed=False)
    def save():write(args.output/'result.json',record)
    save()
    try:
        for index,mode in ((2,'session'),(3,'original')):
            if git('rev-parse','HEAD')!=revision or git('status','--porcelain'):
                raise ValueError('Frozen continuation source changed')
            arm=args.output/(str(index)+'-'+mode);name='perf20261010-session-worker-recovery-'+str(index)+'-'+mode
            command=[sys.executable,str(ROOT/'tools/qwen3/probe_native_compile_session.py'),
                '--name',name,'--worker-mode',mode,'--deadline-unix',str(args.deadline_unix),'--output',str(arm)]
            for key in ('config','baseline-native','oracle','numerical-source'):
                command.extend(['--'+key,str(getattr(args,key.replace('-','_')))])
            plan=args.output/(str(index)+'-owned-plan.json');controller=args.output/(str(index)+'-process.json')
            write(plan,dict(source_root=str(ROOT),source_revision=revision,deadline_unix=args.deadline_unix,
                minimum_remaining_seconds=1800,command=command))
            process=subprocess.run([sys.executable,str(ROOT/'tools/qwen3/run_owned_command.py'),
                                    '--plan',str(plan),'--controller',str(controller)])
            if process.returncode:raise ValueError('Owned continuation arm failed; evidence retained')
            row=dict(order=index,worker_mode=mode,result=entry(arm/'result.json'),controller=entry(controller))
            raw.append(require_completed_arm(row,args.deadline_unix));rows.append(row);record['DSP_calls']+=16;save()
        measurements=summarize_abba(raw)
        record.update(status='original_host_worker_vs_preloaded_session_ABBA64_completed_after_recorded_server_restart_DIAGNOSTIC',
            measurements=measurements,positive_early_startup_signal=(measurements['first_token_seconds']['ratio']<.98
                and measurements['host_wait_seconds']['ratio']<.9),completed_unix=time.time())
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:save()


if __name__=='__main__':main()
