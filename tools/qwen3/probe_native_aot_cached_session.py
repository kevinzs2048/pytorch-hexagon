"""Own native16 AOT-cache diagnostics through the unchanged native coordinator.

Only its exact native SSH launch is replaced by the recorded ARM wrapper.
The original coordinator still verifies every packet, token, DSO, original
compiler chain and process exit. Its unmodified source and actual wrapped
command are both retained; cache diagnostics remain outside model acceptance.
"""
import argparse
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import time

from build import ROOT,entry,sha,write
from run_regression import require_configured_python
from terminal_reports import read_terminal_report


def wrapped_native_command(command,original_prefix,wrapper,wrapper_args):
    if not isinstance(command,(list,tuple)) or not command or Path(str(command[0])).name!='ssh':
        return command,False
    words=shlex.split(command[-1])
    if words[:len(original_prefix)]!=original_prefix:
        return command,False
    native=words[len(original_prefix)-1]
    actual=words[:len(original_prefix)-1]+[str(wrapper),'--native-source',native,*wrapper_args]+words[len(original_prefix):]
    return [*command[:-1],shlex.join(actual)],True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--baseline-native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--namespace-template',type=Path,required=True)
    parser.add_argument('--aot-mode',choices=('original','cache'),required=True)
    parser.add_argument('--cache-name',required=True)
    args,remaining=parser.parse_known_args()
    if any(not value or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in value)
           for value in (args.name,args.cache_name)):
        raise ValueError('Scoped diagnostic and cache names required')
    if not args.output.is_absolute() or args.output.exists():raise ValueError('Fresh coordinator output required')
    config=json.loads(args.config.read_text());require_configured_python(config);board=config['board']
    base=json.loads((args.baseline_native/'result.json').read_text());prefix=base['board'][:5]
    prior=json.loads((args.baseline_native/'native-result.json').read_text())
    manifest=json.loads(args.namespace_template.read_text())
    if len(prefix)!=5 or prefix[3]!=board['python'] or manifest['native_source_sha256']!=prior['source_sha256']:
        raise ValueError('Template native source changed')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target']]
    if subprocess.run(ssh+['pgrep -x buffer_stream'],capture_output=True).returncode!=1:
        raise ValueError('No concurrent DSP owner permitted')
    actual_platform=json.loads(subprocess.check_output(ssh+[shlex.join([board['python'],'-c',
        'import json,platform;print(json.dumps(dict(python_version=platform.python_version(),architecture=platform.machine())))'])],text=True))
    if actual_platform['architecture']!='aarch64':raise ValueError('Actual ARM platform required')
    manifest.update(actual_platform)
    adapter=Path(__file__).with_name('native_aot_output.py');wrapper=Path(__file__).with_name('run_native_aot_cached_arm.py')
    if manifest['adapter_sha256']!=sha(adapter):raise ValueError('Exact experimental adapter template required')
    deploy=Path(board['observations'])/(args.name+'-aot-wrapper-sources')
    subprocess.run(ssh+[shlex.join(['mkdir',str(deploy)])],check=True)
    source_entries=[]
    for path in (adapter,wrapper):
        target=deploy/path.name
        with path.open('rb') as stream:
            subprocess.run(ssh+[shlex.join([board['python'],'-c',
                'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(target)])],stdin=stream,check=True)
        if subprocess.check_output(ssh+[shlex.join(['sha256sum',str(target)])],text=True).split()[0]!=sha(path):
            raise ValueError('Exact wrapper deployment bytes required')
        source_entries.append(dict(source=entry(path),board_path=str(target)))
    board_manifest=deploy/'namespace.json'
    subprocess.run(ssh+[shlex.join([board['python'],'-c',
        'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(board_manifest)])],
        input=json.dumps(manifest,sort_keys=True).encode(),check=True)
    cache_root=Path(board['observations'])/(args.cache_name+'-native-AOT-cache')
    report=Path(board['observations'])/(args.name+'-AOT-cache-report.json')
    launches=[];original_popen=subprocess.Popen;old_argv=sys.argv
    wrapper_args=['--namespace-manifest',str(board_manifest),'--cache-root',str(cache_root),
                  '--cache-report',str(report),'--aot-mode',args.aot_mode]
    def intercept(command,*positional,**kwargs):
        actual,changed=wrapped_native_command(command,prefix,deploy/wrapper.name,wrapper_args)
        if changed:launches.append(dict(original=command,actual=actual))
        return original_popen(actual,*positional,**kwargs)
    delegate=Path(__file__).with_name('probe_native_compile_session.py')
    sys.argv=[str(delegate),'--config',str(args.config),'--baseline-native',str(args.baseline_native),
              '--output',str(args.output),'--name',args.name,*remaining]
    record=dict(status='running',driver_source=entry(__file__),delegate_source=entry(delegate),
        aot_mode=args.aot_mode,namespace_template=entry(args.namespace_template),manifest=manifest,
        deployed_sources=source_entries,actual_native_launches=launches,
        compiler_and_worker_commands_unchanged=True,model_inference_acceptance=False,
        performance_acceptance=False,production_configuration_changed=False,start_unix=time.time())
    try:
        subprocess.Popen=intercept
        runpy.run_path(str(delegate),run_name='__main__')
        if len(launches)!=1:raise ValueError('Exactly one recorded immutable native launch required')
        cached,_=read_terminal_report(ssh,report,args.output/'AOT-cache-report.json')
        if (cached['status']!='original_native_entry_completed_with_recorded_AOT_cache_protocol_DIAGNOSTIC'
                or cached['factory_restored'] is not True or cached['original_configuration_restored'] is not True
                or cached['adapter_sha256']!=sha(adapter) or cached['wrapper_sha256']!=sha(wrapper)
                or cached['native_source_sha256']!=manifest['native_source_sha256']):
            raise ValueError('Actual cache protocol source and restoration checks required')
        record.update(status='original_native16_with_actual_recorded_AOT_protocol_DIAGNOSTIC',
            native_coordinator=entry(args.output/'result.json'),AOT_cache_report=entry(args.output/'AOT-cache-report.json'))
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        subprocess.Popen=original_popen;sys.argv=old_argv;record['completed_unix']=time.time()
        if args.output.exists():write(args.output/'AOT-protocol-controller.json',record)


if __name__=='__main__':main()
