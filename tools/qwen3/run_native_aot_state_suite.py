"""Run the original six state boundaries through independently audited AOT reuse.

Each actual case retains its original prompt/request semantics and expected
count. No new prompt corpus, code generation or performance acceptance is
introduced. The first tail case is the integration pilot; failure stops the
queue before any later state is launched.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from build import ROOT,entry,sha,write


EXPECTED=dict(tails=6,eos=9,**{'old-long32':33},prefill1024=33,decode993=95,code64=128)


def preserved_case_args(command):
    values=list(command[2:]);result=[];removed=set();index=0
    while index<len(values):
        flag=values[index]
        if flag in ('--config','--accepted-run','--output','--name'):
            if flag in removed or index+1>=len(values):raise ValueError('Exactly one original coordinator option required')
            removed.add(flag);index+=2
        else:
            result.append(flag);index+=1
    if removed!={'--config','--accepted-run','--output','--name'}:
        raise ValueError('Complete original coordinator arguments required')
    if '--input-packer' in result or '--startup-timing' in result:
        raise ValueError('Original established state corpus required before explicit joined/timing opt-in')
    if (result.count('--parameter-check')!=1 or result[result.index('--parameter-check')+1]!='exact-bytes'
            or result.count('--specialize-floats')!=1):
        raise ValueError('All original full parameter and FP64 guards required')
    return [*result,'--input-packer','joined','--startup-timing']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-state-suite','online-source','accepted-run','namespace-template','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('name','cache-name'):p.add_argument('--'+name,required=True)
    p.add_argument('--reuse-baseline-references',action='store_true')
    p.add_argument('--deadline-unix',type=int,required=True);a=p.parse_args()
    read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if (not a.output.is_absolute() or a.output.exists() or git('status','--porcelain')
            or a.deadline_unix-time.time()<14400
            or any(not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name)
                   for name in (a.name,a.cache_name))):
        raise ValueError('Fresh frozen six-case queue and unchanged four-hour admission budget required')
    baseline=read(a.baseline_state_suite/'result.json');plan=read(a.baseline_state_suite/'plan.json')
    if (baseline['status']!='all_integer_candidate_state_regressions_passed' or baseline['DSP_calls']!=304
            or [(r['case'],r['expected_DSP_calls']) for r in plan['cases']]!=list(EXPECTED.items())):
        raise ValueError('Complete independently accepted original six-state304 corpus required')
    from compare_performance import accepted
    for row in plan['cases']:
        _,native,_=accepted(a.baseline_state_suite/row['case'])
        if native['DSP_calls']!=row['expected_DSP_calls']:
            raise ValueError('Original accepted state count changed')
        preserved_case_args(row['command'])
    revision=git('rev-parse','HEAD');a.output.mkdir();rows=[]
    record=dict(status='running',source_revision=revision,source_root=str(ROOT),rows=rows,DSP_calls=0,
        deadline_unix=a.deadline_unix,original_four_hour_admission_preserved=True,
        baseline_plan=entry(a.baseline_state_suite/'plan.json'),baseline_result=entry(a.baseline_state_suite/'result.json'),
        namespace_template=entry(a.namespace_template),new_model_PRIMARY_compilations=0,
        model_inference_acceptance=False,performance_acceptance=False,production_configuration_changed=False)
    def save():write(a.output/'result.json',record)
    def own(command,stem,minimum):
        if git('rev-parse','HEAD')!=revision or git('status','--porcelain') or time.time()>=a.deadline_unix:
            raise ValueError('Frozen state queue source or authorized deadline changed')
        plan_path=a.output/(stem+'-owned-plan.json');controller=a.output/(stem+'-process.json')
        write(plan_path,dict(source_root=str(ROOT),source_revision=revision,deadline_unix=a.deadline_unix,
            minimum_remaining_seconds=minimum,command=command))
        subprocess.run([sys.executable,str(ROOT/'tools/qwen3/run_owned_command.py'),
            '--plan',str(plan_path),'--controller',str(controller)],check=True)
        return controller
    save()
    try:
        for case in plan['cases']:
            name=case['case'];folder=a.output/name
            write(a.output/'progress.json',dict(stage='original_observed_state',case=name,completed_cases=len(rows)))
            command=[sys.executable,str(ROOT/'tools/qwen3/probe_native_aot_online_regression.py'),
                '--output',str(folder),'--name',a.name+'-'+name,'--cache-name',a.cache_name,
                '--deadline-unix',str(a.deadline_unix)]
            for key in ('config','online-source','accepted-run','namespace-template'):
                command.extend(['--'+key,str(getattr(a,key.replace('-','_')))])
            command+=preserved_case_args(case['command'])
            if a.reuse_baseline_references:
                command+=['--reference-reuse',str(a.baseline_state_suite/name)]
            controller=own(command,name,1800)
            review=a.output/(name+'-independent-review.json')
            review_command=[sys.executable,str(ROOT/'tools/qwen3/revisit_native_aot_online_regression.py'),
                '--config',str(a.config),'--baseline',str(a.baseline_state_suite/name),'--native',str(folder),
                '--controller',str(controller),'--output',str(review)]
            reviewer=own(review_command,name+'-review',300);actual=read(review)
            if (actual['status']!='native_AOT_original_online_state_packets_full_guards_raw_FNV_and_owners_independently_revisited_DIAGNOSTIC'
                    or actual['DSP_calls']!=case['expected_DSP_calls'] or actual['controller']!=entry(controller)):
                raise ValueError('Complete original case and independent actual owner evidence required')
            rows.append(dict(case=name,DSP_calls=actual['DSP_calls'],native=str(folder),controller=entry(controller),
                independent_review=entry(review),reviewer_controller=entry(reviewer),cache=actual['cache']))
            record['DSP_calls']=sum(row['DSP_calls'] for row in rows);save()
        record.update(status='all_original_native_AOT_state304_cases_individually_independently_revisited_DIAGNOSTIC',
            all_original_state_boundaries_and_counts_retained=True,completed_unix=time.time())
        write(a.output/'progress.json',dict(stage='complete',completed_cases=len(rows)))
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
