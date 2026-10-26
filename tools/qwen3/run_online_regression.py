"""Run independent native ARM traces and observe production DSP packets online."""
import argparse,json,os,shlex,signal,subprocess,sys
from pathlib import Path
from run_regression import accepted_base,replace,sha,write,require_configured_python
from build import ROOT,verify_dependencies
from verify_board import inspect
from terminal_reports import read_terminal_report
from prepare_performance_runtime import validate_transport_override


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','accepted-run','output','observations','board-observations'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--name',required=True)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompts-json');inputs.add_argument('--requests-json')
    p.add_argument('--chat-template',action='store_true')
    p.add_argument('--max-new-tokens',type=int,required=True);p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--require-last-position',action='store_true',help='Require request zero to consume real cache position 1023')
    p.add_argument('--require-eos',action='store_true')
    p.add_argument('--runtime-catalog',type=Path,help='Opt-in transport catalog; all model entries must remain identical')
    p.add_argument('--board-runtime-catalog',type=Path)
    p.add_argument('--generation-timing',action='store_true')
    p.add_argument('--parameter-check',choices=('sha256','exact-bytes'),default='sha256')
    p.add_argument('--startup-timing',action='store_true')
    p.add_argument('--specialize-floats',action='store_true')
    p.add_argument('--input-packer',choices=('established','joined'),default='established')
    p.add_argument('--reference-reuse',type=Path,
                   help='Reuse a completed same-model ARM reference with the exact same plan and pinned sources')
    a=p.parse_args()
    if (not a.name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for x in a.name)
            or not a.output.is_absolute() or a.output.exists() or a.output.is_symlink()
            or not a.observations.is_absolute() or not a.observations.is_dir()
            or not a.board_observations.is_absolute() or not 1<=a.max_new_tokens<=1024
            or not 1<=a.chunk_size<=32 or (a.chat_template and not a.prompts_json)):
        p.error('Fresh named output, existing evidence paths and bounded generation required')
    base=accepted_base(a.accepted_run)
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean diagnostic checkout required')
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    changed=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',base['revision'],revision],text=True).splitlines()
    allowed={'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py',
             'qwen3/runtime/arm/audit_arm_dynamic_warm_run.py',
             'qwen3/runtime/arm/hexagon_qwen_generation_budget.py'}
    if any(not (f in allowed or f.startswith('tools/qwen3/') or f.startswith('qwen3/runtime/arm/test_')) for f in changed):
        raise ValueError('Online diagnostics must preserve accepted numerical compiler/model helpers')
    c=json.loads(a.config.read_text());b=c['board'];ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',b['ssh_target']]
    require_configured_python(c)
    runtime_override=None
    if bool(a.runtime_catalog)!=bool(a.board_runtime_catalog):
        raise ValueError('Paired host/board transport catalog paths required')
    if a.runtime_catalog:
        if not a.runtime_catalog.is_absolute() or not a.board_runtime_catalog.is_absolute():
            raise ValueError('Absolute transport catalog paths required')
        original=json.loads(Path(c['catalog']).read_text())
        candidate=json.loads(a.runtime_catalog.read_text())
        validate_transport_override(original,candidate)
        remote=subprocess.check_output(ssh+[shlex.join(['sha256sum',str(a.board_runtime_catalog)])],text=True).split()[0]
        if remote!=sha(a.runtime_catalog):raise ValueError('Board transport catalog differs from host')
        runtime_override=dict(path=str(a.runtime_catalog),board_path=str(a.board_runtime_catalog),sha256=remote)
    a.output.mkdir(parents=True)
    checked=verify_dependencies(c,json.loads((ROOT/'tools/qwen3/dependencies.lock.json').read_text()))
    write(a.output/'SDK-verification.json',dict(passed=True,files=len(checked),
        lock_sha256=sha(ROOT/'tools/qwen3/dependencies.lock.json')))
    actual=inspect(c);expected=json.loads((ROOT/'tools/qwen3/board-framework.lock.json').read_text())
    write(a.output/'board-framework-verification.json',dict(passed=actual==expected,actual=actual))
    if actual!=expected:raise ValueError('Pinned ARM reference/capture framework changed')
    O,BO=a.observations,a.board_observations
    deployment=BO/(a.name+'-sources')
    subprocess.run(ssh+[shlex.join(['mkdir',str(deployment)])],check=True)
    archive=subprocess.Popen(['git','-C',str(ROOT),'archive',revision],stdout=subprocess.PIPE)
    with archive.stdout:
        extract=subprocess.run(ssh+[shlex.join(['tar','-C',str(deployment),'-xf','-'])],stdin=archive.stdout)
    if archive.wait() or extract.returncode:raise RuntimeError('Committed source deployment failed')
    accepted_record=Path(json.loads((a.accepted_run/'acceptance.json').read_text())['audit']['path'])
    # Derive ABI contracts from the real accepted cold captures, not a prompt
    # whitelist. They describe tensors only; live DSP graphs still go through
    # actual capture, compiler reply validation and final artifact rehashing.
    cold_path=Path(c['observations'])/Path(base['board'][base['board'].index('--output')+1]).name/'result.json'
    cold,cold_sha=read_terminal_report(ssh,
        Path(base['board'][base['board'].index('--output')+1])/'result.json',
        a.output/'accepted-cold-result.json')
    plan=dict(schema=1,max_new_tokens=a.max_new_tokens,chunk_size=a.chunk_size,
        contracts=[x['contract'] for x in cold['captures']],
        accepted_cold_run=dict(path=str(cold_path),sha256=cold_sha),
        accepted_audit=dict(path=str(accepted_record),sha256=sha(accepted_record)))
    if a.prompts_json:
        prompts=json.loads(a.prompts_json)
        if not isinstance(prompts,list) or not 1<=len(prompts)<=8 or any(type(t) is not str or not t or len(t.encode())>65536 for t in prompts):
            raise ValueError('One to eight bounded nonempty text prompts required')
        plan.update(prompts=prompts,chat_template=a.chat_template)
    else:plan['requests']=json.loads(a.requests_json)
    plan_host=O/(a.name+'-plan.json');write(plan_host,plan);plan_board=BO/plan_host.name
    reference_board=BO/(a.name+'-reference');reference_host=O/reference_board.name
    reference=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+b['pythonpath'],b['python'],
        str(deployment/'tools/qwen3/reference_packet_trace_arm.py'),'--sources',str(deployment),
        '--context',b['context'],'--hf',b['hf'],'--plan',str(plan_board),
        '--plan-sha256',sha(plan_host),'--output',str(reference_board)]
    if a.reference_reuse is not None:
        from compare_performance import accepted
        from reference_reuse import require_reference_reuse
        old_acceptance,_,_=accepted(a.reference_reuse)
        old_commands=json.loads((a.reference_reuse/'commands.json').read_text())
        old_trace=json.loads((a.reference_reuse/'reference-result.json').read_text())
        reused=require_reference_reuse(old_trace,plan_sha=sha(plan_host),numerical_revision=base['revision'],
            acceptance=old_acceptance,commands=old_commands,recipe=base['recipe'],
            reference_source_sha=sha(ROOT/'tools/qwen3/reference_packet_trace_arm.py'),
            codec_sha=sha(ROOT/'tools/qwen3/packet_trace.py'))
        if sha(a.reference_reuse/'reference-result.json')!=reused['sha256']:
            raise ValueError('Copied original reference differs from its accepted source')
        reference_board=Path(reused['board_path']).parent
        reference_host=Path(reused['path']).parent
        write(a.output/'reference-reuse.json',dict(status='same_plan_independent_ARM_reference_reused',
            original_acceptance=dict(path=str(a.reference_reuse/'acceptance.json'),sha256=sha(a.reference_reuse/'acceptance.json')),
            trace=reused,plan_sha256=sha(plan_host),CPU_reference_trace_executed_this_run=False))
        write(a.output/'progress.json',dict(stage='reusing_pinned_independent_ARM_reference'))
    else:
        write(a.output/'reference-command.json',reference);write(a.output/'progress.json',dict(stage='independent_ARM_native_reference'))
        with (a.output/'reference.log').open('xb') as log:
            subprocess.run(ssh+[shlex.join(reference)],stdout=log,stderr=subprocess.STDOUT,check=True)
    trace_path=reference_host/'result.json'
    trace,trace_sha=read_terminal_report(ssh,reference_board/'result.json',a.output/'reference-result.json')
    if trace['status']!='independent_native_packet_trace_completed':raise ValueError('Independent reference incomplete')
    if a.require_last_position:
        first=[f for f in trace['forwards'] if f['request_index']==0]
        if not first or first[-1]['positions'][-1]!=1023 or not trace['requests'][0]['terminal_observation']['reached_requested_token_limit']:
            raise ValueError('CPU native request did not reach physical cache position 1023 and its token budget')
    mailbox=O/('live-'+a.name+'-mailbox');mailbox.mkdir(exist_ok=False)
    worker=list(base['worker']);replace(worker,'--mailbox',mailbox)
    native=list(base['board'][4:])
    if runtime_override:
        replace(native,'--catalog',a.board_runtime_catalog)
        replace(native,'--catalog-sha256',runtime_override['sha256'])
    if a.generation_timing:native.append('--generation-timing')
    if a.parameter_check!='sha256':native+=['--parameter-check',a.parameter_check]
    if a.startup_timing:native.append('--startup-timing')
    if a.specialize_floats:native.append('--specialize-floats')
    if '--input-packer' in native:replace(native,'--input-packer',a.input_packer)
    elif a.input_packer=='joined':native+=['--input-packer','joined']
    for flag,value in [('--output',BO/a.name),('--workdir','/dev/shm/qwen3-arm-dynamic-'+a.name),
        ('--compiler-mailbox',BO/mailbox.name),('--packet-recording','hashes'),
        ('--max-new-tokens',a.max_new_tokens),('--chunk-size',a.chunk_size)]:replace(native,flag,value)
    if a.prompts_json:replace(native,'--prompts-json',a.prompts_json)
    else:
        native[native.index('--prompts-json')]='--requests-json';replace(native,'--requests-json',a.requests_json)
    if a.chat_template:native.append('--chat-template')
    observation_board=BO/(a.name+'-online-audit');observation_host=O/observation_board.name
    board=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+b['pythonpath'],b['python'],
        str(deployment/'tools/qwen3/run_online_packet_trace_arm.py'),'--sources',str(deployment),
        '--trace',str(reference_board/'result.json'),'--trace-sha256',trace_sha,
        '--plan',str(plan_board),'--plan-sha256',sha(plan_host),'--audit-output',str(observation_board),'--']+native
    write(a.output/'commands.json',dict(revision=revision,source_root=str(ROOT),deployment=str(deployment),
        numerical_revision=base['revision'],numerical_worker_source=base['source_root'],
        recipe=base['recipe'],worker=worker,board=board,reference=reference,plan_sha256=sha(plan_host),
        graph_whitelist=False,accepted_run=str(a.accepted_run),runtime_override=runtime_override))
    env=dict(os.environ,PYTHONPATH=base['source_root']+'/qwen3/runtime/arm:'+base['source_root'],
        LD_LIBRARY_PATH=':'.join(c['library_path']),PYTHONDONTWRITEBYTECODE='1')
    write(a.output/'progress.json',dict(stage='native_DSP_with_online_observer'))
    with (a.output/'worker.log').open('xb') as wl,(a.output/'board.log').open('xb') as bl:
        wp=subprocess.Popen(worker,stdout=wl,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            bp=subprocess.Popen(ssh+[shlex.join(board)],stdout=bl,stderr=subprocess.STDOUT,start_new_session=True)
            write(a.output/'active-processes.json',dict(coordinator=os.getpid(),worker=wp.pid,board_ssh=bp.pid))
            brc=bp.wait()
            if brc:os.killpg(wp.pid,signal.SIGTERM)
            wrc=wp.wait(timeout=35)
        finally:
            for process in [locals().get('bp'),wp]:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=35)
    write(a.output/'run-exit.json',dict(board_returncode=brc,worker_returncode=wrc))
    if brc or wrc:raise RuntimeError('Native online session failed; evidence retained')
    run_path=O/a.name/'result.json'
    run,run_sha=read_terminal_report(ssh,BO/a.name/'result.json',a.output/'native-result.json')
    observed,observed_sha=read_terminal_report(ssh,observation_board/'result.json',a.output/'online-audit.json')
    if observed['numerical_acceptance'] is not True or observed['run']['sha256']!=run_sha:
        raise ValueError('Observed actual native packets not accepted')
    sys.path[:0]=[str(Path(base['source_root'])/'qwen3/runtime/arm'),base['source_root']]
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    write(a.output/'progress.json',dict(stage='independent_actual_compile_chain_rehash'))
    chain=audit_compile_handoff(run,mailbox,rebuilt=True)
    if run['frontend_invocations'] or run['machine_code_compilations']:
        raise ValueError('Actual compilation observed; not accepted as warm reuse')
    if a.require_last_position:
        groups=[f for f in run['forwards'] if f['request_index']==0]
        if groups[-1]['cache_positions'][-1]!=1023:raise ValueError('DSP did not reach physical last cache position')
    eos=[r['index'] for r in run['requests'] if r['terminal_observation']['last_token_is_native_eos'] and not r['terminal_observation']['reached_requested_token_limit']]
    if a.require_eos and not eos:raise ValueError('Actual early EOS was not generated')
    # Copy compact evidence into the caller's run directory; raw tensors on a
    # difference remain explicitly referenced, never silently discarded.
    (a.output/'plan.json').write_bytes(plan_host.read_bytes())
    write(a.output/'acceptance.json',dict(status='native_online_numerical_and_compile_chain_acceptance_passed',
        revision=revision,numerical_revision=base['revision'],DSP_calls=run['DSP_calls'],
        numerical_acceptance=True,complete_raw_packets_retained=False,
        online_audit=dict(path=str(a.output/'online-audit.json'),sha256=observed_sha,
            board_path=str(observation_board/'result.json')),
        trace=dict(path=str(a.output/'reference-result.json'),sha256=trace_sha,
            board_path=str(reference_board/'result.json')),compile_handoff=chain,
        last_physical_position_required=a.require_last_position,early_native_EOS_requests=eos))
    write(a.output/'progress.json',dict(stage='accepted'));print('Native online regression independently accepted',flush=True)


if __name__=='__main__':main()
