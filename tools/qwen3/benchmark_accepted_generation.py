"""Time native generation against a preaccepted full-packet oracle.

No inline CPU observer runs during the timed session. Actual capture and compiler
identity checks still run; all packet SHA, tokens and FNV are checked afterward.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from analyze_performance import summarize
from build import verify_dependencies
from compare_performance import accepted, compare, sha
from run_regression import accepted_base, replace, require_configured_python
from terminal_reports import read_terminal_report
from bounded_loader_policy import configured_policy, apply_arguments, verify_native_layout


DIAGNOSTIC_SOURCE_COMPARE = '''
import hashlib,json,sys
from pathlib import Path
old,new=map(Path,sys.argv[1:3]);expected=json.loads(sys.argv[3])
def inventory(root):
 result={}
 for p in root.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
   if p.is_symlink():raise ValueError('Ordinary deployment source files required')
   result[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return result
left,right=inventory(old),inventory(new)
changed={p for p in set(left)|set(right) if left.get(p)!=right.get(p)}
if changed!=set(expected) or any(right.get(p)!=v for p,v in expected.items()):
 raise ValueError('Loader diagnostic changed unrelated numerical/runtime source files')
print(json.dumps(dict(all_other_source_files_identical=True,original_files=len(left),
 diagnostic_files=len(right),changed_files={p:right[p] for p in sorted(changed)},
 diagnostic_manifest_sha256=hashlib.sha256(json.dumps(right,sort_keys=True).encode()).hexdigest())))
'''


def native_arguments(command, *, startup_timing=False):
    if (len(command)<6 or command.count('--')!=1 or '--audit-output' not in command
            or Path(command[4]).name!='run_online_packet_trace_arm.py'):
        raise ValueError('Original accepted inline-observer command required')
    arguments=command[command.index('--')+1:]
    if (arguments.count('--packet-recording')!=1
            or arguments[arguments.index('--packet-recording')+1]!='hashes'
            or arguments.count('--generation-timing')>1
            or arguments.count('--startup-timing')>1):
        raise ValueError('Accepted hash-only source and unambiguous timing mode required')
    result = list(arguments)
    if startup_timing and '--startup-timing' not in result:
        result.append('--startup-timing')
    return result


def filetrace_command(executable, directory, command):
    if not executable.is_absolute() or not directory.is_absolute():
        raise ValueError('Absolute board tracer and fresh trace directory required')
    return [str(executable), '-ff', '-tt', '-T', '-yy',
        '-e', 'trace=openat,read,pread64,readv,lseek,close,mmap,munmap,ioctl',
        '-e', 'raw=read,pread64,readv', '-s', '160',
        '-o', str(directory/'file-operations'), *command]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','oracle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--loader-filetrace-strace',type=Path)
    parser.add_argument('--loader-filetrace-output',type=Path)
    parser.add_argument('--loader-diagnostic-native-source',type=Path)
    parser.add_argument('--constant-load-split-proofs',type=Path)
    parser.add_argument('--loader-kernel-trace-start',type=Path)
    parser.add_argument('--input-packer',choices=('established','joined'),default='established')
    parser.add_argument('--startup-timing',action='store_true',
        help='Record startup phases required by the native AOT delivery validation')
    args=parser.parse_args()
    if bool(args.loader_filetrace_strace)!=bool(args.loader_filetrace_output):
        parser.error('Loader diagnostic requires both board tracer and fresh trace output')
    if bool(args.loader_diagnostic_native_source)!=bool(args.constant_load_split_proofs):
        parser.error('Bounded loader diagnostic requires explicit native source and constant proofs')
    if (not args.output.is_absolute() or args.output.exists() or args.output.is_symlink()
            or not args.name or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for x in args.name)):
        parser.error('Fresh absolute benchmark output and scoped name required')
    oracle_acceptance,oracle,_=accepted(args.oracle)
    commands=json.loads((args.oracle/'commands.json').read_text())
    base=accepted_base(Path(commands['accepted_run']))
    if base['revision']!=oracle_acceptance['numerical_revision']:
        raise ValueError('Verified numerical source differs from oracle')
    config=json.loads(args.config.read_text());require_configured_python(config)
    loader_policy=configured_policy(config)
    if loader_policy and args.loader_diagnostic_native_source:
        raise ValueError('Explicit policy and diagnostic source override cannot be combined')
    verify_dependencies(config,json.loads(Path(__file__).with_name('dependencies.lock.json').read_text()))
    board=config['board'];ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',board['ssh_target']]
    kernel_trace=None
    if args.loader_kernel_trace_start:
        if not args.loader_kernel_trace_start.is_absolute():
            raise ValueError('Absolute actual kernel diagnostic start receipt required')
        code='import json,sys;from pathlib import Path;print(Path(sys.argv[1]).read_text())'
        start=json.loads(subprocess.check_output(ssh+[shlex.join([board['python'],'-c',code,
            str(args.loader_kernel_trace_start)])],text=True))
        enabled=subprocess.check_output(ssh+[shlex.join(['sudo','-n','cat',
            '/sys/kernel/tracing/instances/qwen3_loader_20261008/events/qwen3_loader_20261008/map_entry/enable'])],text=True).strip()
        if (start['status']!='private_FastRPC_mapping_trace_active' or enabled!='1'
                or start['source_sha256']!=sha(Path(__file__).with_name('trace_fastrpc_mapping_arm.py'))):
            raise ValueError('Actual maintained private kernel mapping trace must be active')
        kernel_trace=dict(start_receipt=str(args.loader_kernel_trace_start),start=start,
            original_native_command_and_loader_unchanged=True)
    source=Path(commands['deployment']);native_source=source/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py'
    actual=subprocess.check_output(ssh+[shlex.join(['sha256sum',str(native_source)])],text=True).split()[0]
    if actual!=oracle['source_sha256']:
        raise ValueError('Accepted native runner deployment changed')
    loader_layout=None
    if loader_policy and oracle.get('constant_loader_layout',{}).get('kind')==loader_policy['kind']:
        original_cold=json.loads((Path(commands['accepted_run'])/'native-result.json').read_text())
        fingerprint=verify_native_layout(oracle)
        if fingerprint!=verify_native_layout(original_cold):
            raise ValueError('The same-model oracle must preserve all three original bounded payloads')
        apply_arguments(native_arguments(commands['board']),loader_policy)
        loader_layout=dict(native_source=str(native_source),native_source_sha256=actual,
            original_native_source=str(native_source),original_native_source_sha256=oracle['source_sha256'],
            proofs=loader_policy['proofs'],numerical_compiler_and_model_unchanged=True,
            common_payload_preserving_policy=loader_policy,
            source_comparison=dict(already_accepted_native_source_reused=True,
                                   original_bounded_payload_fingerprint=fingerprint))
    elif args.loader_diagnostic_native_source or loader_policy:
        diagnostic=(Path(loader_policy['baseline_native_source']) if loader_policy
                    else args.loader_diagnostic_native_source)
        proofs=Path(loader_policy['proofs']) if loader_policy else args.constant_load_split_proofs
        if not diagnostic.is_absolute() or not proofs.is_absolute():
            raise ValueError('Absolute explicit diagnostic source and proofs required')
        expected=sha(Path(base['source_root'])/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py')
        actual=subprocess.check_output(ssh+[shlex.join(['sha256sum',str(diagnostic)])],text=True).split()[0]
        if actual!=expected:raise ValueError('Maintained diagnostic native runner source changed')
        split_source=Path(__file__).with_name('split_constant_load_segments.py')
        expected_files={'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py':expected,
            'tools/qwen3/split_constant_load_segments.py':sha(split_source)}
        # A candidate oracle already deployed this committed runner. An older
        # baseline may differ only in this runner and the byte-proof verifier.
        if loader_policy and expected==oracle['source_sha256']:
            diagnostic=native_source
            comparison=json.dumps(dict(oracle_already_uses_committed_loader_source=True))
        else:
            comparison=subprocess.check_output(ssh+[shlex.join([board['python'],'-c',
                DIAGNOSTIC_SOURCE_COMPARE,str(source),str(diagnostic.parents[3]),
                json.dumps(expected_files)])],text=True)
        loader_layout=dict(native_source=str(diagnostic),native_source_sha256=actual,
            original_native_source=str(native_source),original_native_source_sha256=oracle['source_sha256'],
            proofs=str(proofs),numerical_compiler_and_model_unchanged=True,
            common_payload_preserving_policy=loader_policy,
            source_comparison=json.loads(comparison))
        native_source=diagnostic
    parent=Path(commands['board'][commands['board'].index('--output')+1]).parent
    input_packing=None
    if args.input_packer=='joined':
        if any((args.loader_filetrace_strace,args.loader_diagnostic_native_source,args.loader_kernel_trace_start)):
            raise ValueError('Host packing timing cannot combine with diagnostic tracing')
        if oracle.get('input_packer')=='joined':
            # This oracle already executed the joined serializer. Reuse its
            # exact deployed bytes; do not redeploy unrelated tooling against
            # the numerical revision or borrow another serializer's receipt.
            packer=source/'tools/qwen3/joined_input_packet.py'
            expected=oracle['input_packing_source_sha256']
            actual=subprocess.check_output(ssh+[shlex.join(['sha256sum',str(packer)])],text=True).split()[0]
            if actual!=expected or sha(Path(__file__).with_name('joined_input_packet.py'))!=expected:
                raise ValueError('The accepted joined serializer source changed')
            input_packing=dict(status='already_accepted_joined_deployment_reused',
                original_source=str(source),deployment=str(source),packer_sha256=expected,
                numerical_compiler_and_model_unchanged=True)
        elif sha(Path(__file__).resolve().parents[2]/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py') == oracle['source_sha256']:
            from deploy_input_packing import reuse_committed_packer
            input_packing=reuse_committed_packer(config,source,oracle['source_sha256'])
        else:
            from deploy_input_packing import deploy
            input_packing=deploy(config,base['revision'],source,parent/(args.name+'-input-packer-sources'))
            native_source=Path(input_packing['deployment'])/'qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py'
    # The host mailbox path is derived from the same existing paired observation
    # roots as the accepted run; it never imports alternate caches or a recipe.
    previous_mailbox=Path(commands['worker'][commands['worker'].index('--mailbox')+1])
    host_mailbox=previous_mailbox.parent/('live-'+args.name+'-mailbox')
    board_mailbox=Path(commands['board'][commands['board'].index('--compiler-mailbox')+1]).parent/host_mailbox.name
    worker=list(base['worker']);replace(worker,'--mailbox',host_mailbox)
    native=native_arguments(commands['board'],startup_timing=args.startup_timing)
    if '--input-packer' in native:replace(native,'--input-packer',args.input_packer)
    elif args.input_packer=='joined':native+=['--input-packer','joined']
    if loader_policy:native=apply_arguments(native,loader_policy)
    elif loader_layout:native+=['--constant-load-split-proofs',str(args.constant_load_split_proofs)]
    for flag,value in (('--output',parent/args.name),('--workdir','/dev/shm/qwen3-arm-dynamic-'+args.name),
            ('--compiler-mailbox',board_mailbox)):
        replace(native,flag,value)
    command=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+board['pythonpath'],board['python'],str(native_source),*native]
    trace=None
    if args.loader_filetrace_strace:
        command=filetrace_command(args.loader_filetrace_strace,args.loader_filetrace_output,command)
        tracer_sha=subprocess.check_output(ssh+[shlex.join(['sha256sum',str(args.loader_filetrace_strace)])],text=True).split()[0]
        subprocess.run(ssh+[shlex.join(['mkdir',str(args.loader_filetrace_output)])],check=True)
        trace=dict(executable=str(args.loader_filetrace_strace),sha256=tracer_sha,
            directory=str(args.loader_filetrace_output),read_arguments_raw=True,
            includes_original_Dynamo_AOT_parameter_snapshot_and_packet_pack=True)
    args.output.mkdir(parents=True);host_mailbox.mkdir(exist_ok=False)
    def write(name,value):(args.output/name).write_text(json.dumps(value,indent=2)+'\n')
    record=dict(status='native_generation_timing_without_inline_CPU_observer_running',
        coordinator_source_sha256=sha(__file__),oracle_acceptance_sha256=sha(args.oracle/'acceptance.json'),
        oracle_native_sha256=sha(args.oracle/'native-result.json'),numerical_source_revision=base['revision'],
        worker=worker,board=command,CPU_model_fallback=False,inline_CPU_observer=False,
        native_token_intervals_measured='--generation-timing' in native,
        includes_actual_capture_handoff_AOT_writeback_and_packet_hashing=True)
    record.update(loader_filetrace=trace,loader_layout=loader_layout,loader_kernel_trace=kernel_trace,
        input_packing=input_packing,
        timings_eligible_for_performance_acceptance=trace is None and (loader_layout is None or loader_policy is not None) and kernel_trace is None)
    write('result.json',record)
    env=dict(os.environ,PYTHONPATH=base['source_root']+'/qwen3/runtime/arm:'+base['source_root'],
        LD_LIBRARY_PATH=':'.join(config['library_path']),PYTHONDONTWRITEBYTECODE='1')
    try:
        with (args.output/'worker.log').open('xb') as wl,(args.output/'board.log').open('xb') as bl:
            wp=subprocess.Popen(worker,stdout=wl,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:
                native_start=time.perf_counter_ns()
                bp=subprocess.Popen(ssh+[shlex.join(command)],stdout=bl,stderr=subprocess.STDOUT,start_new_session=True)
                write('active-processes.json',dict(worker=wp.pid,board_ssh=bp.pid,coordinator=os.getpid()))
                brc=bp.wait()
                native_wall=(time.perf_counter_ns()-native_start)/1e9
                if brc:os.killpg(wp.pid,signal.SIGTERM)
                wrc=wp.wait(timeout=35)
            finally:
                for process in (locals().get('bp'),wp):
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=35)
        write('run-exit.json',dict(board_returncode=brc,worker_returncode=wrc))
        if brc or wrc:raise ValueError('Actual benchmark session failed')
        run,run_sha=read_terminal_report(ssh,parent/args.name/'result.json',args.output/'native-result.json')
        if (run['status']!='ARM_native_dynamic_compiler_handoff_generate_completed'
                or run['process_exit_code']!=0 or run['CPU_model_fallback']
                or run['frontend_invocations'] or run['machine_code_compilations']):
            raise ValueError('Actual warm native DSP session required')
        compare(oracle,run)
        if run.get('input_packer','established')!=args.input_packer:
            raise ValueError('Actual host input serializer differs from the selected mode')
        if [r['terminal_observation'] for r in oracle['requests']]!=[r['terminal_observation'] for r in run['requests']]:
            raise ValueError('Native stopping behavior changed')
        sys.path[:0]=[base['source_root']+'/qwen3/runtime/arm',base['source_root']]
        from audit_arm_dynamic_warm_run import audit_compile_handoff
        chain=audit_compile_handoff(run,host_mailbox,rebuilt=True)
        measurement=summarize(run)
        record.update(status='native_generation_timing_verified_against_preaccepted_full_packet_oracle',
            DSP_calls=run['DSP_calls'],native_sha256=run_sha,full_input_output_model_token_FNV_equality=True,
            native_process_wall_seconds=native_wall,
            native_process_wall_includes_import_model_load_and_shutdown=True,
            complete_compile_chain_rehashed=chain,measurement=measurement,
            scope='native_generate_without_inline_CPU_observer_including_capture_and_packet_hashing',
            new_CPU_reference_execution=False,production_TUI_startup_claimed=False)
        if trace is not None:
            record.update(status='native_loader_filetrace_diagnostic_verified_against_preaccepted_full_packet_oracle',
                scope='original_native_model_load_and_execution_filetrace_diagnostic',
                performance_acceptance=False)
        if loader_layout is not None:
            if loader_policy:
                verify_native_layout(run)
                record['common_payload_preserving_loader_policy_verified']=True
            elif run.get('constant_loader_layout',{}).get('kind')!='bounded_readonly_LOAD_diagnostic':
                raise ValueError('Actual native loader layout evidence missing')
            if not loader_policy:
                record.update(status='native_bounded_loader_diagnostic_verified_against_preaccepted_full_packet_oracle',
                    scope='bounded_loader_native_generation_diagnostic',performance_acceptance=False)
        if kernel_trace is not None:
            record.update(status='native_kernel_mapping_diagnostic_verified_against_preaccepted_full_packet_oracle',
                scope='original_native_kernel_mapping_diagnostic',performance_acceptance=False)
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        write('result.json',record)
    print(record['status'])


if __name__=='__main__':main()
