"""Opt-in observer around the unchanged production native DSP packet exchange."""
import argparse,hashlib,json,os,platform,subprocess,sys,traceback
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','trace','plan','audit-output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--trace-sha256',required=True);p.add_argument('--plan-sha256',required=True)
    a,native=p.parse_known_args()
    if native[:1]==['--']:native=native[1:]
    if platform.machine()!='aarch64' or sha(a.trace)!=a.trace_sha256 or sha(a.plan)!=a.plan_sha256:
        raise ValueError('Actual ARM and pinned independent evidence required')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    from online_packet_audit import OnlinePacketObserver
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for
    import run_qwen_arm_dynamic_warm_generate as runner
    from hexagon_live_compile_mailbox import atomic_json
    def option(flag):
        if native.count(flag)!=1:raise ValueError('Exactly one native option required: '+flag)
        return native[native.index(flag)+1]
    if option('--packet-recording')!='hashes' or option('--local-packet-transport')!='shared-memory':
        raise ValueError('Explicit hash storage and shared-memory native exchange required')
    output=Path(option('--output'));context=Path(option('--context'));hf=Path(option('--hf'))
    trace=json.loads(a.trace.read_text());plan=json.loads(a.plan.read_text())
    if (trace['plan_sha256']!=a.plan_sha256 or trace['context_sha256']!=sha(context) or
            trace['generation_config_sha256']!=sha(hf/'generation_config.json') or
            trace['codec_sha256']!=sha(Path(__file__).with_name('packet_trace.py')) or
            trace['source_sha256']!=sha(Path(__file__).with_name('reference_packet_trace_arm.py')) or
            int(option('--max-new-tokens'))!=plan['max_new_tokens'] or int(option('--chunk-size'))!=plan['chunk_size']):
        raise ValueError('Production generation options differ from independent CPU plan')
    # Prompt equality is verified again from actual native token trajectories
    # and the completed tokenized request records; the reference never supplies
    # model data, KV or selected tokens to the production invocation.
    a.audit_output.mkdir(exist_ok=False)
    report=dict(schema=1,status='online_packet_observation_running',numerical_acceptance=False,
        trace=dict(path=str(a.trace),sha256=a.trace_sha256),plan_sha256=a.plan_sha256,
        source_sha256=sha(__file__),observer_sha256=sha(Path(__file__).with_name('online_packet_audit.py')),
        codec_sha256=sha(Path(__file__).with_name('packet_trace.py')),
        replay_source_sha256=sha(Path(__file__).with_name('packet_replay_arm.py')),
        tolerance=dict(full_logits_max_abs_less_than=1e-4,valid_KV_max_abs_less_than=1e-4),
        complete_raw_packets_retained=False,CPU_model_fallback=False,
        CPU_reference_execution=True,DSP_values_replaced=False,forwards=[])
    def save():atomic_json(a.audit_output/'result.json',report)
    from packet_replay_arm import make_replay,snapshot
    replay=make_replay(sources=a.sources,plan=a.plan,plan_sha256=a.plan_sha256,
        context=context,hf=hf,audit_output=a.audit_output)
    observer=OnlinePacketObserver(trace,layout_for,replay);original=runner.retain_packet
    report['forwards']=observer.rows;save()
    def retain(path,payload,mode):
        record=json.loads((output/'result.json').read_text())
        if Path(path).name=='input.bin':
            step=len(observer.rows)
            try:
                # Captures are saved by the real compiler callback before this
                # packet; bind against the actual newest capture's ABI.
                expected=trace['forwards'][step]
                candidates=[c for c in record['captures'] if c['contract']==expected['contract']]
                if len(candidates)!=1:raise ValueError('Unique actual captured ABI required')
                observer.input(payload,candidates[0]['contract'],len(record['requests']))
            except BaseException:
                failed=a.audit_output/f'failed-input-{step:04d}.bin';report['failed_input']=snapshot(failed,payload);save();raise
        else:
            try:
                observer.output(payload);save();print('online verified forward',len(observer.rows)-1,flush=True)
            except BaseException:
                if observer.pending is not None:
                    pending=observer.pending;expected=pending['expected']
                    report['first_failure']=dict(step=expected['step'],request_index=expected['request_index'],
                        query=expected['query'],positions=expected['positions'],
                        differing_input_tensor_indices=[i for i,(x,y) in enumerate(zip(pending['input']['fields'],expected['input']['fields'],strict=True)) if x!=y],
                        differing_output_tensor_indices=[i for i,(x,y) in enumerate(zip(pending.get('output',{}).get('fields',[]),expected['output']['fields'])) if x!=y])
                    folder=a.audit_output/f'difference-{len(observer.rows):04d}'
                    if folder.exists():
                        report['failed_forward']=dict(input=dict(path=str(folder/'input.bin'),sha256=sha(folder/'input.bin')),
                            output=dict(path=str(folder/'output.bin'),sha256=sha(folder/'output.bin')))
                    else:
                        folder=a.audit_output/f'failed-forward-{len(observer.rows):04d}';folder.mkdir()
                        report['failed_forward']=dict(input=snapshot(folder/'input.bin',observer.pending['payload']),output=snapshot(folder/'output.bin',payload))
                    save()
                raise
        return original(path,payload,mode)
    runner.retain_packet=retain
    try:
        sys.argv=[str(a.sources/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py')]+native
        runner.main();run=json.loads((output/'result.json').read_text())
        report['library_lifecycle']=observer.finish(run)
        if (run['frontend_invocations'] or run['machine_code_compilations']):
            raise ValueError('Actual compiler cache miss; retained but not a warm acceptance')
        report.update(status='native_DSP_online_numerical_observation_passed',numerical_acceptance=True,
            DSP_calls=run['DSP_calls'],run=dict(path=str(output/'result.json'),sha256=sha(output/'result.json')),
            frontend_invocations=run['frontend_invocations'],machine_code_compilations=run['machine_code_compilations'])
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        runner.retain_packet=original;save()
    print(report['status'],flush=True)


if __name__=='__main__':main()
