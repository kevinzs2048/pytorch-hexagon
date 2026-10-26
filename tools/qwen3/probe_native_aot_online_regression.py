"""Retain the original online oracle/observer and record the actual AOT wrapper.

Only the exact original observer SSH launch and original compiler worker are
adapted. The cache-only session worker prevents an accidental model rebuild.
The committed original online coordinator, reference and observer own all
numerical acceptance; this additional protocol remains diagnostic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import sys
import time


def observed_command(command,prefix,observer,wrapper,wrapper_args):
    if not isinstance(command,(list,tuple)) or not command or Path(str(command[0])).name!='ssh':
        return command,False
    words=shlex.split(command[-1])
    if words[:len(prefix)]!=prefix or words[len(prefix):len(prefix)+1]!=[str(observer)]:
        return command,False
    return [*command[:-1],shlex.join([*prefix,str(wrapper),*wrapper_args,*words[len(prefix)+1:]])],True


def session_command(command,original_prefix,worker,numerical_source,revision,deadline):
    if not isinstance(command,(list,tuple)) or list(command[:2])!=original_prefix:
        return command,False
    return [command[0],str(worker),'--numerical-source',str(numerical_source),
        '--numerical-revision',revision,'--deadline-unix',str(deadline),*command[2:]],True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','online-source','accepted-run','namespace-template','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('name','cache-name'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    a,remaining=p.parse_known_args();own=Path(__file__).resolve().parents[2]
    git=lambda root,*words:subprocess.check_output(['git','-C',str(root),*words],text=True).strip()
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    entry=lambda path:dict(path=str(Path(path).absolute()),sha256=sha(path))
    def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
    if (not a.output.is_absolute() or a.output.exists() or a.deadline_unix-time.time()<1800
            or git(own,'status','--porcelain') or git(a.online_source,'status','--porcelain')
            or any(not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name)
                   for name in (a.name,a.cache_name))):
        raise ValueError('Fresh bounded diagnostic and both frozen clean sources required')
    # Original coordinator imports must bind its own committed ROOT.
    sys.path[:0]=[str(a.online_source/'tools/qwen3'),str(a.online_source/'qwen3/runtime/arm'),str(a.online_source)]
    from run_regression import accepted_base,require_configured_python
    base=accepted_base(a.accepted_run);config=json.loads(a.config.read_text());require_configured_python(config)
    numerical=Path(base['source_root']);revision=base['revision'];board=config['board']
    if git(numerical,'rev-parse','HEAD')!=revision or git(numerical,'status','--porcelain'):
        raise ValueError('Original numerical compiler must remain frozen')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target']]
    if subprocess.run(ssh+['pgrep -x buffer_stream'],capture_output=True).returncode!=1:
        raise ValueError('No concurrent DSP owner permitted')
    adapter=Path(__file__).with_name('native_aot_output.py');wrapper=Path(__file__).with_name('run_native_aot_observed_arm.py')
    delegate=a.online_source/'tools/qwen3/run_online_regression.py'
    native=a.online_source/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py'
    observer=a.online_source/'tools/qwen3/run_online_packet_trace_arm.py'
    manifest=json.loads(a.namespace_template.read_text())
    if (sha(adapter)!=manifest['adapter_sha256'] or sha(native)!=manifest['native_source_sha256']
            or manifest['numerical_revision']!=revision or sha(base['recipe']['path'])!=manifest['recipe_sha256']):
        raise ValueError('Original numerical/recipe/native identity and immutable adapter required')
    manifest.update(json.loads(subprocess.check_output(ssh+[shlex.join([board['python'],'-c',
        'import json,platform;print(json.dumps(dict(python_version=platform.python_version(),architecture=platform.machine())))'])],text=True)))
    deploy=Path(board['observations'])/(a.name+'-aot-wrapper-sources')
    subprocess.run(ssh+[shlex.join(['mkdir',str(deploy)])],check=True);deployed=[]
    for path in (adapter,wrapper):
        target=deploy/path.name
        subprocess.run(ssh+[shlex.join([board['python'],'-c',
            'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(target)])],
            input=path.read_bytes(),check=True)
        if subprocess.check_output(ssh+[shlex.join(['sha256sum',str(target)])],text=True).split()[0]!=sha(path):
            raise ValueError('Deployed experimental source differs')
        deployed.append(dict(source=entry(path),board_path=str(target)))
    board_manifest=deploy/'namespace.json'
    subprocess.run(ssh+[shlex.join([board['python'],'-c',
        'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(board_manifest)])],
        input=json.dumps(manifest,sort_keys=True).encode(),check=True)
    original_deploy=Path(board['observations'])/(a.name+'-sources')
    board_native=original_deploy/'qwen3/runtime/arm'/native.name
    board_observer=original_deploy/'tools/qwen3'/observer.name
    report=Path(board['observations'])/(a.name+'-AOT-cache-report.json')
    native_output=Path(board['observations'])/a.name
    wrapper_args=['--native-source',str(board_native),'--observer-source',str(board_observer),
        '--observer-sha256',sha(observer),'--namespace-manifest',str(board_manifest),
        '--cache-root',str(Path(board['observations'])/(a.cache_name+'-native-AOT-cache')),
        '--cache-report',str(report),'--native-output',str(native_output)]
    prefix=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+board['pythonpath'],board['python']]
    worker=Path(__file__).with_name('native_compile_session_worker.py')
    launches=[];workers=[];original_popen=subprocess.Popen;old_argv=sys.argv
    def intercept(command,*positional,**kwargs):
        actual,changed=observed_command(command,prefix,board_observer,deploy/wrapper.name,wrapper_args)
        if changed:launches.append(dict(original=command,actual=actual))
        else:
            actual,changed=session_command(command,list(base['worker'][:2]),worker,numerical,revision,a.deadline_unix)
            if changed:workers.append(dict(original=command,actual=actual))
        return original_popen(actual,*positional,**kwargs)
    sys.argv=[str(delegate),'--config',str(a.config),'--accepted-run',str(a.accepted_run),
        '--output',str(a.output),'--name',a.name,*remaining]
    record=dict(status='running',driver_source=entry(__file__),delegate_source=entry(delegate),
        online_source_root=str(a.online_source),online_source_revision=git(a.online_source,'rev-parse','HEAD'),
        original_native_source=entry(native),original_observer_source=entry(observer),worker_source=entry(worker),
        numerical_source_root=str(numerical),numerical_source_revision=revision,
        namespace_template=entry(a.namespace_template),manifest=manifest,deployed_sources=deployed,
        actual_observer_launches=launches,actual_worker_launches=workers,
        accepted_cold=entry(a.accepted_run/'acceptance.json'),deadline_unix=a.deadline_unix,
        model_inference_acceptance=False,performance_acceptance=False,
        production_configuration_changed=False,start_unix=time.time())
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(signum,frame):raise InterruptedError('Observed native coordinator interrupted: '+str(signum))
    for sig in previous_handlers:signal.signal(sig,interrupted)
    try:
        subprocess.Popen=intercept;runpy.run_path(str(delegate),run_name='__main__')
        if len(launches)!=1 or len(workers)!=1:
            raise ValueError('Exactly one original observer and one cache-only worker required')
        from terminal_reports import read_terminal_report
        cached,_=read_terminal_report(ssh,report,a.output/'AOT-cache-report.json')
        if (cached['status']!='original_native_observer_completed_with_recorded_AOT_cache_protocol_DIAGNOSTIC'
                or cached['factory_restored'] is not True or cached['original_configuration_restored'] is not True
                or cached['native_source_sha256']!=sha(native) or cached['observer_source_sha256']!=sha(observer)
                or cached['adapter_sha256']!=sha(adapter) or cached['wrapper_sha256']!=sha(wrapper)):
            raise ValueError('Actual original observer, adapter and restoration evidence required')
        record.update(status='original_online_numerical_acceptance_with_recorded_native_AOT_protocol_DIAGNOSTIC',
            original_online_acceptance=entry(a.output/'acceptance.json'),AOT_cache_report=entry(a.output/'AOT-cache-report.json'))
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        subprocess.Popen=original_popen;sys.argv=old_argv
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        record['completed_unix']=time.time()
        if a.output.exists():write(a.output/'AOT-online-protocol-controller.json',record)


if __name__=='__main__':main()
