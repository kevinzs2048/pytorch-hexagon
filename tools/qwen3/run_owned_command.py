"""Own one pinned project command and durably record its actual terminal exit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());source=plan['source_root'];command=plan['command']
    git=lambda *args:subprocess.check_output(['git','-C',source,*args],text=True).strip()
    if (not a.controller.is_absolute() or a.controller.exists()
            or plan['deadline_unix']-time.time()<plan['minimum_remaining_seconds']
            or git('rev-parse','HEAD')!=plan['source_revision'] or git('status','--porcelain')
            or not Path(command[1]).resolve().is_relative_to(Path(source).resolve())):
        raise ValueError('Fresh actual controller, full admitted budget and pinned clean project source required')
    record=dict(status='starting',controller_pid=os.getpid(),controller_start_ticks=Path('/proc/self/stat').read_text().split()[21],
        source_root=source,source_revision=plan['source_revision'],command=command,
        authorized_deadline_unix=plan['deadline_unix'],start_unix=time.time());child=None
    def save():
        tmp=a.controller.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(a.controller)
    def interrupted(signum,frame):raise InterruptedError('Owned supervisor interrupted: '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    save()
    try:
        with a.controller.with_suffix('.log').open('xb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(status='running',child_pid=child.pid,child_start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21]);save()
            rc=child.wait(timeout=max(1,plan['deadline_unix']-time.time()-60))
        record.update(status='completed' if rc==0 else 'failed',returncode=rc)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        record.update(status='failed',returncode=child.returncode if child is not None else None,error=repr(exc))
    finally:
        record['completed_unix']=time.time();save()
    if record['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
