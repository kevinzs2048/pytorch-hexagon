"""Deploy a bound Q2 model comparison once the genuine preceding DSP owner exits."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from build import ROOT, entry, write
from bounded_loader_policy import configured_policy
from profile_model_memory import checked
from revisit_q2_dispatch_preflight import actual_owner
from native_model_acceptance import require_model_acceptance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','preparation','preparation-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    c = read(a.config); b = c['board']; policy = configured_policy(c)
    git = lambda *args: subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if (a.output.exists() or not a.output.is_absolute() or git('status','--porcelain') or not policy
            or a.deadline_unix-time.time()<1800 or not a.name or len(a.name)>80
            or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for ch in a.name)):
        raise ValueError('Fresh committed diagnostic and admitted half hour required')
    preparation = read(a.preparation)
    actual_owner(a.preparation_controller,preparation,'prepare_q2_tail_row_model.py',a.preparation.parent)
    manifest_path = checked(preparation['manifest']); m = read(manifest_path)
    model,_=require_model_acceptance(read(checked(m['index'])))
    if model!=m['qualified_baseline_model']:raise ValueError('Exact current qualified baseline required')
    if m['status'] != 'Q2_identical_tail_model_comparison_manifest_prepared_NOT_executed' or m['queries'] != [8,2,32]:
        raise ValueError('Full source-pinned comparison manifest required')
    ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',b['ssh_target']]
    if subprocess.run(ssh+['pgrep -x buffer_stream'],capture_output=True,timeout=15).returncode != 1:
        raise ValueError('Main model delivery and other DSP owners must actually finish first')
    subprocess.run(ssh+['flock -n /tmp/qwen3-qcs9075-probe.lock true'],check=True,timeout=15)
    digest = subprocess.check_output(ssh+[shlex.join(['sha256sum',m['board_catalog']])],text=True,timeout=15).split()[0]
    if digest != m['catalog_sha256']:
        raise ValueError('Original board runtime catalog changed')
    remote = Path('/dev/shm')/a.name
    a.output.mkdir()
    subprocess.run(ssh+[shlex.join(['mkdir',str(remote)])],check=True,timeout=15)
    subprocess.run(ssh+[shlex.join(['mkdir',str(remote/'sources'),str(remote/'inputs')])],check=True,timeout=15)
    revision = git('rev-parse','HEAD')
    archive = subprocess.Popen(['git','-C',str(ROOT),'archive',revision],stdout=subprocess.PIPE)
    with archive.stdout:
        extract = subprocess.run(ssh+[shlex.join(['tar','-C',str(remote/'sources'),'-xf','-'])],stdin=archive.stdout,timeout=60)
    if archive.wait(timeout=30) or extract.returncode:
        raise ValueError('Frozen source deployment failed')
    def copy(source,name,sparse=False):
        source = Path(source)
        want = entry(source)['sha256']
        if sparse:
            with subprocess.Popen(['tar','--sparse','-cf','-','-C',str(source.parent),source.name],stdout=subprocess.PIPE) as pack:
                with pack.stdout:
                    done = subprocess.run(ssh+[shlex.join(['tar','-xf','-','-C',str(remote/'inputs'),
                        '--transform=s|^'+source.name+'$|'+name+'|'])],stdin=pack.stdout,timeout=120)
                if pack.wait(timeout=30) or done.returncode:
                    raise ValueError('Sparse original packet deployment failed')
        else:
            subprocess.run(['scp','-q',str(source),b['ssh_target']+':'+str(remote/'inputs'/name)],check=True,timeout=120)
        actual = subprocess.check_output(ssh+[shlex.join(['sha256sum',str(remote/'inputs'/name)])],text=True,timeout=30).split()[0]
        if actual != want:
            raise ValueError('Deployed input bytes changed')
    copy(manifest_path,'manifest.json')
    for role in ('baseline','candidate'):
        copy(checked(m['libraries'][role]),role+'.so')
    for row in m['packets']:
        copy(checked(row['input']),'input-q'+str(row['query'])+'.bin',sparse=True)
        copy(checked(row['output']),'reference-q'+str(row['query'])+'.bin',sparse=True)
    workdir = Path('/dev/shm')/('qwen3-arm-dynamic-'+a.name)
    command = ['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+b['pythonpath'],b['python'],
               str(remote/'sources/tools/qwen3/probe_q2_tail_row_model_arm.py'),
               '--sources',str(remote/'sources'),'--manifest',str(remote/'inputs/manifest.json'),
               '--manifest-sha256',entry(manifest_path)['sha256'],'--catalog',m['board_catalog'],
               '--constant-load-split-proofs',policy['proofs'],'--workdir',str(workdir),
               '--deadline-unix',str(a.deadline_unix),'--output',str(remote/'board')]
    record = dict(status='Q2_identical_tail_model_comparison_running_NOT_accepted',source_root=str(ROOT),source_revision=revision,
                  source=entry(__file__),board_source=entry(ROOT/'tools/qwen3/probe_q2_tail_row_model_arm.py'),
                  preparation=entry(a.preparation),preparation_controller=entry(a.preparation_controller),
                  manifest=entry(manifest_path),board_command=command,board_root=str(remote),
                  deadline_unix=a.deadline_unix,DSP_calls=0,model_PRIMARY_compilations=0,
                  model_inference_acceptance=False,performance_acceptance=False)
    save = lambda: write(a.output/'result.json',record)
    save()
    try:
        with (a.output/'board.log').open('xb') as log:
            done = subprocess.run(ssh+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,
                                  timeout=min(1500,a.deadline_unix-time.time()-300))
        record['board_returncode'] = done.returncode; save()
        archive_path = remote/'board-evidence.tar'
        subprocess.run(ssh+[shlex.join(['tar','--sparse','-cf',str(archive_path),'-C',str(remote),'board'])],check=True,timeout=120)
        local = a.output/'board-evidence.tar'
        subprocess.run(['scp','-q',b['ssh_target']+':'+str(archive_path),str(local)],check=True,timeout=120)
        actual = subprocess.check_output(ssh+[shlex.join(['sha256sum',str(archive_path)])],text=True,timeout=30).split()[0]
        if entry(local)['sha256'] != actual:
            raise ValueError('Complete sparse evidence changed during transport')
        subprocess.run(['tar','-xf',str(local),'-C',str(a.output)],check=True,timeout=120)
        result = read(a.output/'board/result.json')
        record.update(board=entry(a.output/'board/result.json'),board_archive=entry(local),DSP_calls=result['DSP_calls'])
        if done.returncode or result['status'] != 'actual_Q2_identical_tail_model_ABBA12_full_logits_and_56KV_bitexact_NOT_generation_acceptance':
            raise ValueError('Full model comparison failed; exact input/output evidence retained')
        record.update(status='actual_Q2_identical_tail_model_ABBA12_completed_PENDING_independent_review',completed_unix=time.time()); save()
    except BaseException as exc:
        record.update(status='failed',error=repr(exc)); save(); raise


if __name__ == '__main__':
    main()
