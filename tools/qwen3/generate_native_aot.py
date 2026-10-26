"""Run ordinary inputs through the accepted dynamic model and native AOT cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from build import ROOT, entry, sha, write
from run_regression import require_configured_python
from terminal_reports import read_terminal_report


def generation_command(original, *, prompts, requests, max_new_tokens, chunk_size,
                       chat_template, output, workdir, mailbox):
    """Preserve the accepted runtime arguments and replace only request options."""
    if (prompts is None) == (requests is None):
        raise ValueError('Exactly one original text or token-ID input required')
    if chat_template and prompts is None:
        raise ValueError('Original chat template requires text input')
    remove_values={'--prompts-json','--requests-json','--max-new-tokens','--chunk-size',
                   '--output','--workdir','--compiler-mailbox'}
    words=list(original[:5]);args=list(original[5:]);i=0
    while i<len(args):
        flag=args[i]
        if flag in remove_values:
            if i+1>=len(args) or args[i+1].startswith('--'):
                raise ValueError('Complete original native option required')
            i+=2
        elif flag=='--chat-template' or (flag=='--startup-timing' and mailbox is None):i+=1
        else:words.append(flag);i+=1
    words += ['--prompts-json' if prompts is not None else '--requests-json',
        prompts if prompts is not None else requests,'--max-new-tokens',str(max_new_tokens),
        '--chunk-size',str(chunk_size),'--output',str(output),'--workdir',str(workdir)]
    if mailbox is not None:words+=['--compiler-mailbox',str(mailbox)]
    if chat_template:words.append('--chat-template')
    return words


def admit(index, baseline, state_suite, abba):
    """Read immutable independent closure evidence, never an input-specific oracle."""
    from native_model_acceptance import require_model_acceptance
    from revisit_release_index import indexed_files
    from revisit_native_compile_session_abba import require_normal_owner
    read=lambda p:json.loads(Path(p).read_text())
    model,closure=require_model_acceptance(index)
    refs=[];indexed_files(model,refs)
    suite=read(state_suite/'result.json');paired=read(abba)
    if (suite['status']!='all_original_native_AOT_state304_cases_individually_independently_revisited_DIAGNOSTIC'
            or type(suite['DSP_calls']) is not int or suite['DSP_calls']!=304
            or suite['all_original_state_boundaries_and_counts_retained'] is not True
            or suite['new_model_PRIMARY_compilations']!=0
            or paired['status']!='native_AOT_ABBA64_same_session_worker_all_packets_guards_and_normal_owners_independently_revisited_DIAGNOSTIC'
            or type(paired['DSP_calls']) is not int or paired['DSP_calls']!=64
            or paired['new_DSP_calls']!=0 or paired['startup_samples_per_mode']!=2):
        raise ValueError('Complete independent original cache state304 and paired native64 required')
    indexed_files(suite,refs);indexed_files(paired,refs)
    manifest=paired['backend_namespace']
    if (manifest['adapter_sha256']!=sha(Path(__file__).with_name('native_aot_output.py'))
            or manifest['numerical_revision']!=model['source_revision']
            or manifest['recipe_sha256']!=index['machine_recipe']['sha256']
            or manifest['runtime_catalog_sha256']!=index['runtime_catalog']['sha256']):
        raise ValueError('The cache namespace must use the actual current numerical model and runtime')
    expected=[('tails',6),('eos',9),('old-long32',33),('prefill1024',33),('decode993',95),('code64',128)]
    if [(r['case'],r['DSP_calls']) for r in suite['rows']]!=expected:
        raise ValueError('Original six cache state transitions and counts required')
    for row in suite['rows']:
        owner=require_normal_owner(row['controller']['path'],suite['deadline_unix'])
        require_normal_owner(row['reviewer_controller']['path'],suite['deadline_unix'])
        proof=read(row['independent_review']['path']);protocol=read(proof['AOT_protocol']['path'])
        if (proof['status']!='native_AOT_original_online_state_packets_full_guards_raw_FNV_and_owners_independently_revisited_DIAGNOSTIC'
                or proof['controller']!=row['controller'] or proof['DSP_calls']!=row['DSP_calls']
                or protocol['manifest']!=manifest
                or owner['source_root']!=str(ROOT)):
            raise ValueError('Actual independent state receipts and complete cache namespace required')
    for arm in paired['arms']:
        producer=read(Path(arm['native']['path']).parent/'result.json')
        if producer['baseline'] != entry(baseline/'result.json'):
            raise ValueError('Paired native arms must bind this exact original deployment')
    base=read(baseline/'result.json');prior=read(baseline/'native-result.json')
    if (base['status']!='native_generation_timing_verified_against_preaccepted_full_packet_oracle'
            or base['native_sha256']!=sha(baseline/'native-result.json')
            or base['numerical_source_revision']!=model['source_revision']
            or base['worker'][base['worker'].index('--machine-recipe')+1]!=index['machine_recipe']['path']
            or base['worker'][base['worker'].index('--machine-recipe-sha256')+1]!=manifest['recipe_sha256']
            or prior['source_sha256']!=manifest['native_source_sha256']
            or prior['input_packer']!='joined'
            or base['oracle_acceptance_sha256']!=paired['original_oracle']['sha256']
            or {str(tuple(c['contract']['variables']['query'])):c['DSO_sha256'] for c in prior['captures']}!=index['candidate_libraries']):
        raise ValueError('Tested original joined deployment must bind current accepted DSOs and oracle')
    return model,manifest,base,prior


REMOTE_STOP = r"""
import json,os,signal,sys,time
from pathlib import Path
wrapper,report,output=sys.argv[1:]
found=[]
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():continue
    try:
        words=[v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
        def option(flag,value):return words.count(flag)==1 and words.index(flag)+1<len(words) and words[words.index(flag)+1]==value
        if wrapper in words and option('--cache-report',report) and option('--output',output):
            found.append((int(folder.name),(folder/'stat').read_text().split()[21]))
    except (OSError,UnicodeError,ValueError):continue
if len(found)>1:raise RuntimeError('Ambiguous actual original native owner; retained without signalling')
for pid,ticks in found:
    folder=Path('/proc',str(pid))
    try:
        if (folder/'stat').read_text().split()[21]!=ticks:raise RuntimeError('Native owner PID identity changed')
        os.kill(pid,signal.SIGTERM)
    except FileNotFoundError:continue
    stop=time.monotonic()+25
    while folder.exists() and time.monotonic()<stop:time.sleep(.2)
    if folder.exists():raise RuntimeError('Original native owner did not finish its cleanup')
print(json.dumps(dict(matched_owners=found,actual_native_owners_gone=True)))
"""


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','index','baseline-native','cache-state304','cache-abba','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('name','cache-name'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    p.add_argument('--artifact-catalog',type=Path,help=argparse.SUPPRESS)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompts-json');inputs.add_argument('--requests-json')
    p.add_argument('--chat-template',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=32);p.add_argument('--chunk-size',type=int,default=32)
    a=p.parse_args();read=lambda q:json.loads(Path(q).read_text())
    git=lambda root,*args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if (not a.output.is_absolute() or a.output.exists() or git(ROOT,'status','--porcelain')
            or a.deadline_unix-time.time()<300
            or any(not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name)
                   for name in (a.name,a.cache_name))):
        p.error('Fresh scoped run, committed source and an admitted generation deadline required')
    # Native code owns tokenization, vocabulary, context budget and chunk admission.
    json.loads(a.prompts_json if a.prompts_json is not None else a.requests_json)
    config=read(a.config);require_configured_python(config);board=config['board'];index=read(a.index)
    model,manifest,base,prior=admit(index,a.baseline_native,a.cache_state304,a.cache_abba)
    numerical=Path(model['source_root'])
    if git(numerical,'status','--porcelain') or git(numerical,'rev-parse','HEAD')!=model['source_revision']:
        raise ValueError('Current accepted numerical source must remain frozen')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target']]
    if subprocess.run(ssh+['pgrep -x buffer_stream'],capture_output=True).returncode!=1:
        raise ValueError('Exclusive native DSP session required')
    native=base['board'][4]
    if subprocess.check_output(ssh+[shlex.join(['sha256sum',native])],text=True).split()[0]!=manifest['native_source_sha256']:
        raise ValueError('Actual already accepted native deployment changed')
    platform_info=json.loads(subprocess.check_output(ssh+[shlex.join([board['python'],'-c',
        'import json,platform;print(json.dumps(dict(python_version=platform.python_version(),architecture=platform.machine())))'])],text=True))
    if any(manifest[key]!=value for key,value in platform_info.items()):raise ValueError('Actual ARM cache platform changed')
    artifact=None
    if a.artifact_catalog is not None:
        from run_native_aot_generate_arm import require_artifact_catalog
        artifact=read(a.artifact_catalog/'result.json');catalog=read(a.artifact_catalog/'catalog.json')
        if (artifact['source']!=entry(Path(__file__).with_name('build_native_artifact_catalog.py'))
                or artifact['accepted_model']!=model
                or artifact['catalog']!=entry(a.artifact_catalog/'catalog.json')
                or catalog['base_runtime_catalog']!=index['runtime_catalog']):
            raise ValueError('Exact current reply-derived artifact preparation required')
        manifest=dict(manifest,runtime_catalog_sha256=artifact['catalog']['sha256'])
        require_artifact_catalog(artifact,catalog,artifact['catalog']['sha256'],manifest)
    a.output.mkdir();deploy=Path(board['observations'])/(a.name+'-aot-wrapper-sources')
    subprocess.run(ssh+[shlex.join(['mkdir',str(deploy)])],check=True)
    adapter=Path(__file__).with_name('native_aot_output.py');wrapper=Path(__file__).with_name('run_native_aot_generate_arm.py')
    deployed=[]
    for source in (adapter,wrapper):
        target=deploy/source.name
        subprocess.run(ssh+[shlex.join([board['python'],'-c',
            'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(target)])],input=source.read_bytes(),check=True)
        if subprocess.check_output(ssh+[shlex.join(['sha256sum',str(target)])],text=True).split()[0]!=sha(source):
            raise ValueError('Actual deployed cache integration source changed')
        deployed.append(dict(source=entry(source),board_path=str(target)))
    board_manifest=deploy/'namespace.json'
    subprocess.run(ssh+[shlex.join([board['python'],'-c',
        'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(board_manifest)])],input=json.dumps(manifest,sort_keys=True).encode(),check=True)
    old_worker=base['worker'];host_mailbox=Path(old_worker[old_worker.index('--mailbox')+1]).parent/('live-'+a.name+'-mailbox')
    board_mailbox=Path(base['board'][base['board'].index('--compiler-mailbox')+1]).parent/host_mailbox.name
    if artifact is None:host_mailbox.mkdir(exist_ok=False)
    worker_args=list(old_worker[2:]);worker_args[worker_args.index('--mailbox')+1]=str(host_mailbox)
    worker=[old_worker[0],str(Path(__file__).with_name('native_compile_session_worker.py')),
        '--numerical-source',str(numerical),'--numerical-revision',model['source_revision'],
        '--deadline-unix',str(a.deadline_unix),*worker_args]
    artifact_args=[]
    if artifact is not None:
        board_catalog=deploy/'catalog.json';board_proof=deploy/'catalog-preparation.json'
        for source,target in ((a.artifact_catalog/'catalog.json',board_catalog),(a.artifact_catalog/'result.json',board_proof)):
            subprocess.run(ssh+[shlex.join([board['python'],'-c',
                'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(target)])],input=source.read_bytes(),check=True)
            if subprocess.check_output(ssh+[shlex.join(['sha256sum',str(target)])],text=True).split()[0]!=sha(source):
                raise ValueError('Actual catalog preparation deployment differs')
        for row in catalog['entries']:
            source=Path(row['library']['path']);target=Path(row['library']['remote_path']);expected=row['library']['sha256']
            if sha(source)!=expected:raise ValueError('Actual accepted model library changed')
            subprocess.run(ssh+[shlex.join(['mkdir','-p',str(target.parent)])],check=True)
            exists=subprocess.run(ssh+[shlex.join(['test','-f',str(target)])]).returncode
            if exists==1:
                with source.open('rb') as data:
                    subprocess.run(ssh+[shlex.join([board['python'],'-c',
                        'import pathlib,sys;pathlib.Path(sys.argv[1]).open("xb").write(sys.stdin.buffer.read())',str(target)])],stdin=data,check=True)
            elif exists!=0:raise ValueError('Actual artifact deployment lookup failed')
            if subprocess.check_output(ssh+[shlex.join(['sha256sum',str(target)])],text=True).split()[0]!=expected:
                raise ValueError('Actual accepted artifact DSO deployment differs')
        artifact_args=['--artifact-catalog-proof',str(board_proof),'--artifact-catalog-proof-sha256',sha(a.artifact_catalog/'result.json')]
        worker=None
    native_output=Path(board['observations'])/a.name;cache_report=Path(board['observations'])/(a.name+'-AOT-cache-report.json')
    original=generation_command(base['board'],prompts=a.prompts_json,requests=a.requests_json,
        max_new_tokens=a.max_new_tokens,chunk_size=a.chunk_size,chat_template=a.chat_template,
        output=native_output,workdir='/dev/shm/qwen3-arm-dynamic-'+a.name,mailbox=board_mailbox if artifact is None else None)
    if artifact is not None:
        original[original.index('--catalog')+1]=str(board_catalog)
        original[original.index('--catalog-sha256')+1]=manifest['runtime_catalog_sha256']
    command=[*original[:4],str(deploy/wrapper.name),'--native-source',native,
        '--namespace-manifest',str(board_manifest),'--cache-root',str(Path(board['observations'])/(a.cache_name+'-native-AOT-cache')),
        '--cache-report',str(cache_report),*artifact_args,*original[5:]]
    record=dict(status='running',source_revision=git(ROOT,'rev-parse','HEAD'),source=entry(__file__),
        accepted_model_version=index['current_default_accepted_version'],accepted_model=model,
        state304=entry(a.cache_state304/'result.json'),ABBA64=entry(a.cache_abba),manifest=manifest,
        deployed_sources=deployed,worker=worker,board=command,host_mailbox=str(host_mailbox) if artifact is None else None,
        compiler_mode='original_live_handoff' if artifact is None else 'original_warm_catalog',
        artifact_catalog_preparation=None if artifact is None else entry(a.artifact_catalog/'result.json'),
        CPU_reference_executed=False,new_prompt_reference_equality_claimed=False,
        production_configuration_changed=False,performance_acceptance=False,start_unix=time.time())
    save=lambda:write(a.output/'result.json',record)
    save();wp=None;bp=None
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(signum,frame):raise InterruptedError('Native generation interrupted: '+str(signum))
    for sig in previous_handlers:signal.signal(sig,interrupted)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(numerical/'qwen3/runtime/arm')+':'+str(numerical),LD_LIBRARY_PATH=':'.join(config['library_path']))
    try:
        with (a.output/'worker.log').open('xb') as wl,(a.output/'board.log').open('xb') as bl:
            if worker is not None:wp=subprocess.Popen(worker,env=env,stdout=wl,stderr=subprocess.STDOUT,start_new_session=True)
            bp=subprocess.Popen(ssh+[shlex.join(command)],stdout=bl,stderr=subprocess.STDOUT,start_new_session=True)
            record['active_processes']=dict(coordinator=os.getpid(),board_ssh=bp.pid)
            if wp is not None:record['active_processes']['worker']=wp.pid
            save()
            brc=bp.wait(timeout=max(1,a.deadline_unix-time.time()-60))
            if brc and wp is not None:os.killpg(wp.pid,signal.SIGTERM)
            wrc=wp.wait(timeout=35) if wp is not None else None
        write(a.output/'run-exit.json',dict(board_returncode=brc,worker_returncode=wrc))
        if brc or wrc:raise ValueError('Actual original native or cache-only host worker failed')
        run,run_sha=read_terminal_report(ssh,native_output/'result.json',a.output/'native-result.json')
        cached,_=read_terminal_report(ssh,cache_report,a.output/'AOT-cache-report.json')
        if (cached['status']!='original_native_AOT_cached_generation_completed'
                or cached['factory_restored'] is not True or cached['original_configuration_restored'] is not True
                or cached['wrapper_sha256']!=sha(wrapper) or cached['adapter_sha256']!=sha(adapter)):
            raise ValueError('Complete actual native cache protocol and restored full guards required')
        # The immutable adapter validates the actual ARM interpreter before
        # execution. The host checks that receipt, not its own platform.
        recorded_digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if cached['namespace']!=recorded_digest:raise ValueError('Executed cache namespace differs')
        sys.path[:0]=[str(numerical/'qwen3/runtime/arm'),str(numerical)]
        from audit_arm_dynamic_warm_run import audit_compile_handoff
        from bounded_loader_policy import verify_native_layout
        if artifact is None:
            chain=audit_compile_handoff(run,host_mailbox,rebuilt=True)
        else:
            rows={r['graph']['sha256']:r for r in catalog['entries']}
            for capture in run['captures']:
                row=rows.get(capture['graph']['sha256'])
                if row is None or any(capture[key]!=row[key] for key in ('graph','retained_input_indices','contract','frontend_key','machine_key')):
                    raise ValueError('Actual original full graph and binding selector chose an unproven artifact')
            chain=dict(mode='original_full_graph_and_immutable_binding_warm_catalog',
                artifact_preparation=entry(a.artifact_catalog/'result.json'),original_compile_chain=artifact['original_compile_chain'],
                new_host_compiler_requests=0,new_model_PRIMARY_compilations=0)
        verify_native_layout(run)
        captures={str(tuple(c['contract']['variables']['query'])):c['DSO_sha256'] for c in run['captures']}
        if not captures or any(index['candidate_libraries'].get(key)!=value for key,value in captures.items()):
            raise ValueError('Every executed dynamic query domain must use the accepted model DSO')
        raw=subprocess.check_output(ssh+[shlex.join(['cat',str(native_output/'stream.stdout.txt')])])
        (a.output/'stream.stdout.txt').write_bytes(raw);events=[json.loads(s) for s in raw.decode().splitlines()]
        if (not events or events[-1]['stage']!='close'
                or [e for e in events if e.get('stage')=='invoke']!=[f['invoke'] for f in run['forwards']]
                or any(e.get(key,0) for e in events for key in ('rpc_result','DSP_result','output_result','loader_result'))
                or any(len({e[key] for key in ('AP_input_fnv1a32','RPC_input_fnv1a32','DSP_input_fnv1a32')})!=1
                       for e in events if e.get('stage')=='invoke')):
            raise ValueError('Actual accepted DSP calls, triple input FNV and normal closure required')
        record.update(status='accepted_dynamic_model_native_AOT_generation_completed',DSP_calls=run['DSP_calls'],
            native_sha256=run_sha,compile_chain=chain,cache_protocol=cached['protocol'],
            requests=[dict(prompt_ids=r['prompt_ids'],generated_ids=r['generated_ids'],
                generated_text=r.get('generated_text'),terminal_observation=r['terminal_observation']) for r in run['requests']],completed_unix=time.time())
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        if record['status']=='failed' and bp is not None:
            try:
                record['remote_cleanup']=json.loads(subprocess.check_output(ssh+[shlex.join([
                    board['python'],'-c',REMOTE_STOP,str(deploy/wrapper.name),str(cache_report),str(native_output)])],text=True,timeout=35))
            except BaseException as cleanup_error:
                record['remote_cleanup_error']=repr(cleanup_error)
        for process in (bp,wp):
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        save()
    print(json.dumps(record['requests'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
