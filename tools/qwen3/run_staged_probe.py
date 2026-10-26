"""Stage a recorded diagnostic plan, run it once, and retain process/exit proof.

Plans contain explicit input hashes and an argv list. The board probe itself
owns transport checks and the exclusive DSP lock. No code is generated here.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from build import ROOT,sha,write


def predecessor_finished(spec):
    """Serialize independent probes using their real owner, never a delay."""
    try:
        record=json.loads(Path(spec['exit_record']).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None
    if record.get('source_revision')!=spec['source_revision']:
        raise ValueError('Prior probe source identity differs')
    if 'returncode' not in record:
        return None
    if record['returncode']!=0 or record.get('status') not in ('complete','completed'):
        raise ValueError('Prior probe did not exit normally; inspect its failure before more DSP work')
    for key in ('controller_pid','child_pid'):
        if record.get(key)!=spec[key]:
            raise ValueError('Prior probe owner identity differs')
        try:
            tick=Path('/proc',str(spec[key]),'stat').read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:
            continue
        if tick==spec[key+'_start_tick']:
            return None
    return dict(exit_record=dict(path=spec['exit_record'],sha256=sha(spec['exit_record'])),
                actual_normal_exit=0,original_processes_gone=True)


def prerequisite_accepted(spec):
    """Authenticate the finished producer, never infer completion from silence."""
    try:
        result=json.loads(Path(spec['result']).read_text())
        exit_record=json.loads(Path(spec['exit_record']).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None
    if exit_record.get('returncode',0)!=0 or result.get('status')=='failed':
        raise ValueError('Required model closure failed; no primitive may run')
    if 'returncode' not in exit_record:
        return None
    if (result.get('status')!=spec['expected_status']
            or result.get('source_revision')!=spec['source_revision']
            or result.get('DSP_calls')!=spec['DSP_calls']
            or result.get('model_inference_acceptance') is not True
            or result.get('candidate_libraries')!=spec['candidate_libraries']
            or exit_record.get('source_revision')!=spec['source_revision']):
        raise ValueError('Required completed closure contract differs')
    for key in ('controller_pid','child_pid'):
        if exit_record.get(key)!=spec[key]:
            raise ValueError('Required original process identity differs')
        path=Path('/proc',str(spec[key]))
        try:
            tick=(path/'stat').read_text().split(') ',1)[1].split()[19]
        except FileNotFoundError:
            continue
        if tick==spec[key+'_start_tick']:
            return None
    return dict(result=dict(path=spec['result'],sha256=sha(spec['result'])),
                exit_record=dict(path=spec['exit_record'],sha256=sha(spec['exit_record'])),
                actual_normal_exit=exit_record['returncode'],original_processes_gone=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not a.output.is_absolute() or a.output.exists():p.error('Fresh absolute run output required')
    config=json.loads(a.config.read_text());plan=json.loads(a.plan.read_text())
    root=Path(plan['board_root']);destination=Path(plan['board_output'])
    for path,base in ((root,Path('/mnt/nvme/qwen3-work/qwen3-performance')),
                      (destination,Path('/mnt/nvme/qwen3-work/qwen3-maintained/accepted-runs'))):
        if not path.is_absolute() or '..' in path.parts or path.parent!=base:
            raise ValueError('One fresh direct project staging/run directory required')
    if (plan['status']!='staged_NOT_executed' or not isinstance(plan['files'],list)
            or not 1<=len(plan['files'])<=20 or not isinstance(plan['command'],list)
            or not all(isinstance(x,str) and x for x in plan['command'])
            or plan['command'][-2:]!=['--output',str(destination)]):raise ValueError('Recorded bound diagnostic plan required')
    names=set()
    for row in plan['files']:
        if (set(row)!={'name','source','sha256'} or Path(row['name']).name!=row['name']
                or row['name'] in ('.','..') or row['name'] in names
                or not Path(row['source']).is_absolute() or sha(row['source'])!=row['sha256']):
            raise ValueError('Unique unchanged pinned file required')
        names.add(row['name'])
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed diagnostic runner required')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target']]
    a.output.mkdir(parents=True)
    record=dict(status='staging',plan=dict(path=str(a.plan),sha256=sha(a.plan)),source_sha256=sha(__file__),
        source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),start_unix=time.time())
    def save():write(a.output/'process.json',record)
    save()
    try:
        if 'prerequisite' in plan:
            spec=plan['prerequisite']
            deadline=spec['deadline_unix']
            if type(deadline) is not int or deadline<=time.time():
                raise ValueError('Explicit future prerequisite deadline required')
            record.update(status='waiting_for_accepted_model_prerequisite',controller_pid=os.getpid())
            save()
            while True:
                proof=prerequisite_accepted(spec)
                if proof:
                    record['prerequisite']=proof
                    save()
                    break
                if time.time()>=deadline:
                    raise TimeoutError('Model prerequisite deadline reached; no DSP probe was run')
                time.sleep(5)
        if 'predecessor' in plan:
            spec=plan['predecessor']
            deadline=spec['deadline_unix']
            if type(deadline) is not int or deadline<=time.time():
                raise ValueError('Explicit future predecessor deadline required')
            record.update(status='waiting_for_original_prior_probe_exit',controller_pid=os.getpid())
            save()
            while True:
                if time.time()>=deadline:
                    raise TimeoutError('Prior probe deadline reached; no new DSP probe was run')
                proof=predecessor_finished(spec)
                if proof:
                    record['predecessor']=proof
                    save()
                    break
                time.sleep(5)
        for row in plan['files']:
            if sha(row['source'])!=row['sha256']:
                raise ValueError('Pinned diagnostic source changed while waiting')
        check='test ! -e '+shlex.quote(str(root))+' && test ! -e '+shlex.quote(str(destination))+' && mkdir '+shlex.quote(str(root))
        subprocess.run(ssh+[check],check=True,timeout=30)
        for row in plan['files']:
            subprocess.run(['scp','-q',row['source'],config['board']['ssh_target']+':'+str(root/row['name'])],check=True,timeout=120)
        check='sha256sum -- '+shlex.join([str(root/row['name']) for row in plan['files']])
        actual=subprocess.check_output(ssh+[check],text=True,timeout=30)
        observed={line.split(maxsplit=1)[1].strip():line.split(maxsplit=1)[0] for line in actual.splitlines()}
        if observed!={str(root/r['name']):r['sha256'] for r in plan['files']}:raise ValueError('Deployed probe hashes differ')
        record['deployed_hashes_verified']=True
        with (a.output/'driver.log').open('xb') as log:
            process=subprocess.Popen(ssh+[shlex.join(plan['command'])],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(status='running',ssh_pid=process.pid,
                process_start_tick=Path('/proc',str(process.pid),'stat').read_text().split(') ',1)[1].split()[19]);save()
            record['returncode']=process.wait(timeout=1800)
        copied=subprocess.run(['scp','-q','-r',config['board']['ssh_target']+':'+str(destination),str(a.output/'board')],capture_output=True,timeout=120)
        record['output_copy_returncode']=copied.returncode
        if record['returncode'] or copied.returncode:raise ValueError('Actual board probe or result copy failed')
        result=a.output/'board/result.json';data=json.loads(result.read_text())
        if data.get('process_exit_code')!=0 or data['status']=='failed' or not data.get('DSP_calls'):
            raise ValueError('Incomplete device exit or execution evidence')
        record.update(status='complete',completed_unix=time.time(),actual_DSP_calls=data['DSP_calls'],
            result=dict(path=str(result),sha256=sha(result)))
    except BaseException as e:record.update(status='failed',error=str(e));raise
    finally:save()
    print(record['status'])


if __name__=='__main__':main()
