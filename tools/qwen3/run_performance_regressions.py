"""Repeat an accepted state-boundary suite with a pinned transport override."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from compare_performance import accepted, compare
from run_regression import replace, require_configured_python

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def derive_command(original, *, source, output, name, host_catalog, board_catalog):
    command = list(original)
    if (len(command) < 2 or Path(command[1]).name != 'run_online_regression.py'
            or Path(command[1]).parent.name != 'qwen3'
            or any(flag in command for flag in ('--runtime-catalog','--board-runtime-catalog','--generation-timing'))):
        raise ValueError('Original numerical suite coordinator command required')
    command[0] = sys.executable
    command[1] = str(source/'tools/qwen3/run_online_regression.py')
    replace(command, '--output', output)
    replace(command, '--name', name)
    return command + ['--runtime-catalog',str(host_catalog),
        '--board-runtime-catalog',str(board_catalog),'--generation-timing']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-suite','runtime-catalog','board-runtime-catalog','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--cases',nargs='+',default=['tails','eos','old-long32','prefill1024','decode993','code64'])
    parser.add_argument('--deadline-unix',type=int)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.exists() or args.output.is_symlink()
            or not args.name or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for x in args.name)
            or len(args.cases)!=len(set(args.cases))):
        parser.error('Fresh absolute suite output, scoped name and unique cases required')
    final=json.loads((args.baseline_suite/'final-result.json').read_text())
    if final['status']!='all_post_cleanup_regressions_passed_and_independently_rechecked':
        raise ValueError('Fully accepted baseline boundary suite required')
    original=json.loads((args.baseline_suite/'plan.json').read_text())
    available={case['case']:case for case in original['cases']}
    if not args.cases or set(args.cases)-available.keys():
        raise ValueError('Every requested case must already exist in accepted suite')
    baselines={}
    for name in args.cases:
        command=available[name]['command']
        if command.count('--config')!=1:
            raise ValueError('Original pinned configuration required')
        require_configured_python(json.loads(Path(command[command.index('--config')+1]).read_text()))
        acceptance,native,_=accepted(args.baseline_suite/name)
        if native['DSP_calls']!=available[name]['expected_DSP_calls']:
            raise ValueError('Baseline actual forward count differs from suite plan')
        baselines[name]=(acceptance,native)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed performance source required')
    args.output.mkdir(parents=True)
    rows=[]
    for name in args.cases:
        old=available[name]
        rows.append(dict(case=name,expected_DSP_calls=old['expected_DSP_calls'],
            baseline_acceptance_sha256=sha(args.baseline_suite/name/'acceptance.json'),
            command=derive_command(old['command'],source=ROOT,output=args.output/name,
                name=args.name+'-'+name,host_catalog=args.runtime_catalog,board_catalog=args.board_runtime_catalog)))
    plan=dict(revision=revision,source=str(ROOT),baseline_suite=str(args.baseline_suite),
        baseline_plan_sha256=sha(args.baseline_suite/'plan.json'),
        baseline_final_sha256=sha(args.baseline_suite/'final-result.json'),
        runtime_catalog_sha256=sha(args.runtime_catalog),deadline_unix=args.deadline_unix,cases=rows)
    save(args.output/'plan.json',plan)
    result=dict(status='performance_boundary_regressions_running',started_unix=time.time(),cases=[],DSP_calls=0)
    save(args.output/'result.json',result)
    for row in rows:
        if args.deadline_unix and time.time()>=args.deadline_unix:
            result['status']='deadline_reached_before_next_case';save(args.output/'result.json',result);return
        if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=revision:
            raise ValueError('Pinned queue source revision changed')
        save(args.output/'progress.json',dict(stage='real_board_case',case=row['case'],completed_cases=len(result['cases'])))
        with (args.output/(row['case']+'.driver.log')).open('xb') as log:
            process=subprocess.run(row['command'],stdout=log,stderr=subprocess.STDOUT)
        name=row['case'];directory=args.output/name
        if process.returncode:
            result.update(status='case_failed',failed_case=name,driver_returncode=process.returncode)
            save(args.output/'result.json',result);raise RuntimeError('Boundary regression failed: '+name)
        acceptance,native,_=accepted(directory)
        old_acceptance,old_native=baselines[name]
        if (native['DSP_calls']!=row['expected_DSP_calls']
                or acceptance['numerical_revision']!=old_acceptance['numerical_revision']):
            raise ValueError('Actual count or accepted numerical compiler revision changed')
        compare(old_native,native)
        proof={file_name:dict(path=str(directory/file_name),sha256=sha(directory/file_name)) for file_name in
            ('acceptance.json','native-result.json','online-audit.json','reference-result.json','run-exit.json')}
        result['cases'].append(dict(case=name,DSP_calls=native['DSP_calls'],
            same_full_buffers_model_DSOs_tokens_and_FNV_results=True,proof=proof))
        result['DSP_calls']+=native['DSP_calls'];save(args.output/'result.json',result)
    result.update(status='all_performance_boundary_regressions_passed',completed_unix=time.time(),
        numerical_acceptance=True,production_throughput_acceptance=False)
    save(args.output/'result.json',result)
    save(args.output/'progress.json',dict(stage='complete',completed_cases=len(result['cases'])))
    print(result['status'],result['DSP_calls'])


if __name__=='__main__':main()
