"""Own original/session/session/original native startup observations.

Each arm uses a fresh process, the same accepted oracle and unchanged board
source. Full packet equality is required before reporting a startup signal.
This experiment never switches the production worker or grants performance
acceptance from a standalone measurement.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from build import ROOT, entry, write


def summarize_abba(rows):
    if len(rows)!=4 or [r['worker_mode'] for r in rows]!=['original','session','session','original']:
        raise ValueError('All four original/session/session/original arms required')
    for row in rows:
        if (row['status']!='host_worker_native16_full_packet_oracle_and_compile_chain_matched_DIAGNOSTIC'
                or type(row['DSP_calls']) is not int or row['DSP_calls']!=16
                or row['full_original_packet_tokens_and_three_FNV_matched'] is not True
                or row['production_configuration_changed'] is not False):
            raise ValueError('Every complete native16 arm must match the full original oracle')
    result={}
    for label,select in (
        ('first_token_seconds',lambda row:row['startup']['requests'][0]['first_token_seconds']),
        ('first_request_full_generate_seconds',lambda row:row['startup']['requests'][0]['full_generate_seconds']),
        ('host_wait_seconds',lambda row:row['startup']['requests'][0]['capture_stages_seconds']['wait_for_compiler_host'])):
        values={mode:[select(row) for row in rows if row['worker_mode']==mode] for mode in ('original','session')}
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for group in values.values() for v in group):
            raise ValueError('Complete positive finite real startup timings required')
        means={mode:statistics.mean(group) for mode,group in values.items()}
        result[label]=dict(original_samples=values['original'],session_samples=values['session'],
            baseline=means['original'],candidate=means['session'],ratio=means['session']/means['original'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-native','oracle','numerical-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-unix',type=int,required=True);args=parser.parse_args()
    git=lambda *words:subprocess.check_output(['git','-C',str(ROOT),*words],text=True).strip()
    if (not args.output.is_absolute() or args.output.exists() or git('status','--porcelain')
            or args.deadline_unix-time.time()<3000):
        raise ValueError('Fresh owned four-arm experiment, frozen source and full 50 minute budget required')
    revision=git('rev-parse','HEAD');args.output.mkdir();rows=[]
    record=dict(status='running',source_revision=revision,rows=rows,deadline_unix=args.deadline_unix,
        DSP_calls=0,model_inference_acceptance=False,performance_acceptance=False,
        production_configuration_changed=False,TUI_startup_claimed=False)
    def save():write(args.output/'result.json',record)
    save()
    try:
        for index,mode in enumerate(('original','session','session','original')):
            if git('rev-parse','HEAD')!=revision or git('status','--porcelain'):
                raise ValueError('Frozen experiment source changed')
            arm=args.output/(str(index)+'-'+mode);name='perf20261010-session-worker-abba-'+str(index)+'-'+mode
            command=[sys.executable,str(Path(__file__).with_name('probe_native_compile_session.py')),
                '--name',name,'--worker-mode',mode,'--deadline-unix',str(args.deadline_unix),'--output',str(arm)]
            for key in ('config','baseline-native','oracle','numerical-source'):
                command.extend(['--'+key,str(getattr(args,key.replace('-','_')))])
            plan=args.output/(str(index)+'-owned-plan.json');controller=args.output/(str(index)+'-process.json')
            write(plan,dict(source_root=str(ROOT),source_revision=revision,deadline_unix=args.deadline_unix,
                           minimum_remaining_seconds=1800,command=command))
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('run_owned_command.py')),
                                '--plan',str(plan),'--controller',str(controller)])
            if run.returncode:
                raise ValueError('Owned startup arm failed; all failure evidence retained')
            result=json.loads((arm/'result.json').read_text())
            rows.append(dict(order=index,worker_mode=mode,result=entry(arm/'result.json'),controller=entry(controller)))
            record['DSP_calls']=sum(json.loads(Path(row['result']['path']).read_text()).get('DSP_calls',0) for row in rows)
            save()
        raw=[json.loads(Path(row['result']['path']).read_text()) for row in rows];measurements=summarize_abba(raw)
        record.update(status='original_host_worker_vs_preloaded_session_ABBA64_complete_full_original_packet_equality_DIAGNOSTIC',
            measurements=measurements,positive_early_startup_signal=(measurements['first_token_seconds']['ratio']<.98
                and measurements['host_wait_seconds']['ratio']<.9),completed_unix=time.time())
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:save()


if __name__=='__main__':main()
