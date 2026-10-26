"""Independently review all global byte-copy original393 receipts and actual RPCs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from build import ROOT, entry, sha, write
from revisit_release_index import indexed_files
from revisit_native_compile_session_abba import require_normal_owner
from run_dynamic_pv_native393 import require_admission
from caller_copy_native_gain import positive_native_gain
import global_copy_release as reviewer_copy_release

# Capture the reviewer's maintenance helpers before the historical producer's
# paths are inserted below. That producer intentionally stays frozen and does
# not contain the later cache-owner replay fix.
if Path(reviewer_copy_release.__file__).resolve() != Path(__file__).with_name('global_copy_release.py').resolve():
    raise ValueError('Zero-tail review helpers must belong to the committed reviewer')


def require_complete(result):
    if (result.get('candidate_kind') != 'model-local-exact-global-byte-copy'
            or result.get('global_copy_candidate') is not True
            or result.get('fresh_cold9_independently_reviewed') is not True
            or any(type(result.get(k)) is not int for k in ('DSP_calls','candidate_DSP_calls',
                    'paired_baseline_DSP_calls','new_primary_model_codegen_invocations'))):
        raise ValueError('Only the separately identified byte-copy closure with exact integer counters is allowed')
    prefix = 'global_copy' if result.get('candidate_kind') == 'model-local-exact-global-byte-copy' else 'caller_copy'
    full = result['status'] == prefix+'_original393_complete_NOT_independently_accepted'
    if (result['status'] not in (prefix+'_original393_complete_NOT_independently_accepted',
            prefix+'_deferred_after_nonpositive_original_native89')
            or result['DSP_calls'] != (393 if full else 89)
            or result['candidate_DSP_calls'] != (361 if full else 57)
            or result['paired_baseline_DSP_calls'] != 32
            or result['original_state304_executed'] is not full
            or result['new_primary_model_codegen_invocations']
            or result['model_inference_acceptance'] or result['performance_acceptance']):
        raise ValueError('Honest complete original393 or deferred native89 required')
    if prefix == 'global_copy' and (
            type(result.get('newly_executed_DSP_calls')) is not int
            or result['newly_executed_DSP_calls'] != (384 if full else 80)
            or type(result.get('reused_independently_verified_DSP_calls')) is not int
            or result['reused_independently_verified_DSP_calls'] != 9):
        raise ValueError('Actual new calls and retained cold9 must be counted separately')
    return full


def require_session_exit(exit_record):
    if any(type(exit_record.get(k)) is not int or exit_record[k] != 0
           for k in ('board_returncode', 'worker_returncode')):
        raise ValueError('Actual board and worker return codes must both be zero')


def require_same_native_callback(original, candidate):
    """Only the inactive joined option and its metadata may precede this model."""
    if original == candidate:
        return
    expected = original.replace('    decode_outputs = unpack_torch_outputs\n',
        "    decode_outputs = unpack_torch_outputs\n    pack_inputs = pack_torch_request\n"
        "    if args.input_packer == 'joined':\n"
        "        from tools.qwen3.joined_input_packet import pack_torch_request_joined\n"
        "        pack_inputs = pack_torch_request_joined\n")
    expected = expected.replace('packet, layout, dimensions = pack_torch_request(contract, bound)',
                                'packet, layout, dimensions = pack_inputs(contract, bound)')
    expected = expected.replace("    parser.add_argument('--parameter-check'",
        "    parser.add_argument('--input-packer', choices=('established', 'joined'), default='established',\n"
        "                        help='Opt-in one-allocation host serialization of the unchanged request ABI')\n"
        "    parser.add_argument('--parameter-check'")
    expected = expected.replace('        arm_forward_timing=dict(',
        "        input_packer=args.input_packer,\n"
        "        input_packing_source_sha256=(sha(Path(__file__).resolve().parents[3]/\n"
        "            'tools/qwen3/joined_input_packet.py') if args.input_packer == 'joined' else\n"
        "            sha(Path(__file__).resolve().parents[3]/\n"
        "                'qcom_hexagon_backend/backend/hexagon_dynamic_buffer_io.py')),\n"
        "        arm_forward_timing=dict(")
    if candidate != expected:
        raise ValueError('Unrelated native callback or runner changes would invalidate caller-only attribution')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'closure', 'controller', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); read = lambda p:json.loads(Path(p).read_text())
    git = lambda root,*words:subprocess.check_output(['git','-C',str(root),*words],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git(ROOT,'status','--porcelain'):
        raise ValueError('Fresh independent review and committed reviewer required')
    result = read(a.closure/'result.json'); full = require_complete(result)
    global_copy = result.get('candidate_kind') == 'model-local-exact-global-byte-copy'
    prefix = 'global_copy' if global_copy else 'caller_copy'
    source = Path(result['source_root']); revision = result['source_revision']
    owner = read(a.controller); require_normal_owner(a.controller,owner['authorized_deadline_unix'])
    if (git(source,'rev-parse','HEAD') != revision or git(source,'status','--porcelain')
            or owner['source_root'] != str(source) or owner['source_revision'] != revision
            or owner['command'][1] != str(source/'tools/qwen3/run_global_copy_native393.py')
            or owner['command'][owner['command'].index('--output')+1] != str(a.closure)
            or owner['authorized_deadline_unix'] != result['deadline_unix']):
        raise ValueError('Exact source and actual normal whole-queue owner required')
    require_admission(owner['authorized_deadline_unix'],owner['start_unix'])
    producer = subprocess.check_output(['git','-C',str(source),'show',revision+':tools/qwen3/run_global_copy_native393.py'])
    if hashlib.sha256(producer).hexdigest() != result['source']['sha256']:
        raise ValueError('Actual historical producer changed')
    refs=[]; indexed_files(result,refs)
    prior = read(result['baseline_independent_revisit']['path']); indexed_files(prior,refs)
    cold_verified=result.get('fresh_cold9_independently_reviewed',False)
    if type(cold_verified) is not bool or (cold_verified and not global_copy):raise ValueError('Copy qualification must retain the original reused-cold gate')
    prior_status=('q2_dispatch_original393_independently_revisited' if cold_verified else
        'caller_copy_original393_independently_revisited' if global_copy else 'dynamic_PV_original393_independently_revisited')
    if (prior['status'] != prior_status or prior['DSP_calls'] != 393
            or prior['model_inference_acceptance'] is not True or prior['closure'] != result['baseline_closure']):
        raise ValueError('Exact already accepted dynamic PV baseline required')
    baseline = read(prior['closure']['path']); cold0 = Path(baseline['cold_run']); warm0 = Path(baseline['warm_run'])
    if str(cold0) != result['baseline_cold'] or str(warm0) != result['baseline_warm']:
        raise ValueError('Baseline session identity changed')
    sys.path[:0]=[str(source/'tools/qwen3'),str(source/'qwen3/runtime/arm'),str(source)]
    from run_integer_regressions import candidate_libraries,compare_integer,validate_parameter_mode,validate_float_policy
    from compare_performance import accepted
    from build_caller_copy_native_recipe import validate_transition
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    from bounded_loader_policy import verify_native_layout
    from audit_prepared_stage_reuse import cold_cache
    from revisit_performance_release import native_abba,states
    cold,warm=Path(result['cold_run']),Path(result['warm_run'])
    base,libraries=candidate_libraries(cold);old_base,old_libraries=candidate_libraries(cold0)
    relative='qwen3/runtime/arm/run_qwen_arm_dynamic_warm_generate.py'
    original_runner=subprocess.check_output(['git','-C',str(source),'show',old_base['revision']+':'+relative]).decode()
    candidate_runner=subprocess.check_output(['git','-C',str(source),'show',revision+':'+relative]).decode()
    require_same_native_callback(original_runner,candidate_runner)
    if subprocess.check_output(['git','-C',str(source),'diff','--name-only',old_base['revision'],revision,
                                '--','qcom_hexagon_backend/backend']):
        raise ValueError('Both timing arms must preserve the entire established backend')
    build=read(result['recipe_build']['path'])
    from build_global_copy_recipe import validate_transition as zero_transition
    zero_transition(read(old_base['recipe']['path']),read(base['recipe']['path']))
    model, expected = reviewer_copy_release.tested_model(build)
    review=read(result['reused_cold_review']['path'])
    reviewer_copy_release.require_cold_review(review, cold=cold, recipe=Path(base['recipe']['path']), libraries=expected, baseline=cold0)
    reviewer_copy_release.require_reviewer_owner(review, result['reused_cold_reviewer_controller']['path'])
    if (result.get('newly_executed_DSP_calls') != (384 if full else 80)
            or result.get('reused_independently_verified_DSP_calls') != 9):
        raise ValueError('Reused original cold9 must be counted separately from newly executed calls')
    indexed_files(model,refs)
    if (libraries != expected or result['candidate_libraries'] != {str(k):v for k,v in expected.items()}
            or base['recipe'] != build['recipe']):
        raise ValueError('Cold candidate must use both exact tested caller-copy libraries')
    # The admitted build independently binds both original model primaries,
    # constants, exact original fixed512 instructions and the model-local
    # dispatch tested by complete model packets. No caller rewrite is made here.
    if (model['all_original_model_primaries_and_constants_unchanged'] is not True
            or model['original_fixed512_text_unchanged'] is not True):
        raise ValueError('Only the reviewed exact byte-copy relinking may qualify')
    raw=read(cold/'native-result.json');compare_integer(read(cold0/'native-result.json'),raw,libraries)
    validate_float_policy(raw,True)
    if raw['DSP_calls'] != 9 or raw['complete_raw_packets_retained'] is not True:
        raise ValueError('Original full cold9 packets required')
    compiler=reviewer_copy_release.revisit_numerical_cold_cache(cold,base)
    _,old,_=accepted(warm0);_,raw,_=accepted(warm)
    compare_integer(old,raw,libraries);validate_float_policy(raw,True);validate_parameter_mode(raw,'exact-bytes')
    if raw['DSP_calls'] != 16:
        raise ValueError('Original independently observed warm16 required')
    timing=native_abba(a.closure);gain=positive_native_gain(timing['raw_native_measurements'])
    if gain is not full:
        raise ValueError('Actual native prefill gain and stable unchanged decode must govern state admission')
    state=states(a.closure,Path(result['original_state_suite']),libraries) if full else None
    folders=[cold,warm]+[a.closure/('native-abba-'+str(i)+'-'+kind) for i,kind in enumerate(('baseline','candidate','candidate','baseline'))]
    counts=[9,16,16,16,16,16]
    if full:
        folders += [a.closure/'state304'/case for case in ('tails','eos','old-long32','prefill1024','decode993','code64')]
        counts += [6,9,33,33,95,128]
    c=read(a.config);runtime=result['common_native_runtime_override'];fingerprint=None;traces=[];chains=[];pids=[]
    for folder,count in zip(folders,counts,strict=True):
        native=read(folder/'native-result.json');commands=read(folder/'commands.json') if (folder/'commands.json').exists() else read(folder/'result.json')
        require_session_exit(read(folder/'run-exit.json'))
        if (native['DSP_calls'] != count or native['process_exit_code'] or native['CPU_model_fallback']
                or native.get('input_packer','established') != 'established'
                or any(Path('/proc',str(pid)).exists() for pid in read(folder/'active-processes.json').values())):
            raise ValueError('Every actual native session and all host owners must close normally')
        actual=verify_native_layout(native)
        if fingerprint is None:fingerprint=actual
        if actual != fingerprint:raise ValueError('Common constant payload changed')
        command=commands['board'];worker=commands['worker']
        if folder != cold and (command[command.index('--catalog')+1] != runtime['board_path']
                or command[command.index('--catalog-sha256')+1] != runtime['sha256']):
            raise ValueError('Every native timing/state session must share the same exact runtime')
        chains.append(audit_compile_handoff(native,Path(worker[worker.index('--mailbox')+1]),rebuilt=True))
        output=Path(command[command.index('--output')+1]);trace=Path(c['observations'])/output.name/'stream.stdout.txt'
        events=[json.loads(line) for line in trace.read_text().splitlines()]
        if (not events or events[-1]['stage'] != 'close'
                or [e for e in events if e.get('stage')=='invoke'] != [r['invoke'] for r in native['forwards']]
                or any(e.get(k,0) for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
            raise ValueError('All raw invokes and normal close required')
        ready=[e for e in events if e.get('stage')=='ready']
        if len(ready) != 1 or type(ready[0].get('ARM_pid')) is not int:
            raise ValueError('Actual AP owner required')
        pids.append(ready[0]['ARM_pid']);traces.append(entry(trace))
    b=c['board'];subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',b['ssh_target'],shlex.join([
        b['python'],'-c','from pathlib import Path;import sys;assert all(not Path("/proc",pid).exists() for pid in sys.argv[1:])',*map(str,pids)])],check=True)
    write(a.output,dict(status=prefix+'_original393_independently_revisited' if full else prefix+'_original_native89_independently_revisited_AND_deferred',
        closure=entry(a.closure/'result.json'),controller=entry(a.controller),source_root=str(source),source_revision=revision,
        numerical_source_root=base['source_root'],numerical_source_revision=base['revision'],
        DSP_calls=393 if full else 89,new_DSP_calls=0,model_inference_acceptance=full,positive_native_callback_gain=gain,positive_prefill_native_gain=gain,
        candidate_DSP_calls=361 if full else 57,paired_baseline_DSP_calls=32,
        candidate_libraries={str(k):v for k,v in libraries.items()},compiler_cache=compiler,native=timing,states=state,
        raw_invocation_traces=traces,compile_chains=chains,actual_AP_owners_gone=pids,
        model_math_and_owned_lifetimes_unchanged=True,
        ordered_FMA_precision_and_owned_lifetimes_preserved=True,
        newly_executed_DSP_calls=result.get('newly_executed_DSP_calls', 89 if not full else 393),
        reused_independently_verified_DSP_calls=result.get('reused_independently_verified_DSP_calls', 0),production_TUI_speedup_claimed=False,
        fresh_cold9_independently_reviewed=cold_verified,global_copy_candidate=True,
        reviewer_source=entry(__file__),reviewer_revision=git(ROOT,'rev-parse','HEAD'),
        reviewer_maintenance_helper=entry(reviewer_copy_release.__file__),completed_unix=time.time()))


if __name__=='__main__':
    main()
