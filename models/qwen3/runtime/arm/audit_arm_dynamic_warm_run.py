"""Independently rehash ARM-local native run versus the pinned host-driven run.

Bitexact full requests/responses preserve that control's observed numerics;
they do not make its unresolved whole-model numerical acceptance pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

from hexagon_runtime.hexagon_compiled_artifact_cache import digest_file, canonical, describe_artifacts
from hexagon_runtime.hexagon_library_lifecycle import audit_library_events
from hexagon_runtime.hexagon_host_timing import audit_arm_forward_timings
from hexagon_qwen_generation_budget import cache_demand


def audit_local_packet_transport(run):
    """Check opt-in staging against the actual AP ready event, not just CLI."""
    mode = run.get('local_packet_transport', 'files')
    ready = [event for event in run['stream_events'] if event.get('stage') == 'ready']
    if len(ready) != 1 or mode not in ('files', 'shared-memory'):
        raise ValueError('One actual local packet transport required')
    shared = ready[0].get('shared_memory_packets', False)
    if type(shared) is not bool or shared != (mode == 'shared-memory'):
        raise ValueError('Actual AP transport disagrees with driver mode')
    return dict(mode=mode, actual_AP_ready_checked=True,
                DSP_resident_KV=False, zero_copy_FastRPC_claim=False)


def audit_text_entry(current, hf):
    """Independently retokenize/decode with the original local HF assets."""
    from transformers import AutoTokenizer
    source = current['input_source']
    expected_names = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
    if (source.get('kind') != 'native_HuggingFace_tokenizer'
            or type(source.get('chat_template')) is not bool
            or [row['name'] for row in source['files']] != expected_names
            or len(source['prompts']) != len(current['requests'])
            or source.get('enable_thinking') != (False if source['chat_template'] else None)):
        raise ValueError('Actual native text provenance required')
    for row in source['files']:
        if digest_file(Path(hf)/row['name']) != row['sha256']:
            raise ValueError('Native tokenizer asset changed')
    tokenizer = AutoTokenizer.from_pretrained(str(hf), local_files_only=True, trust_remote_code=False)
    if type(tokenizer).__name__ != source['tokenizer_class']:
        raise ValueError('Native tokenizer class changed')
    for prompt, request in zip(source['prompts'], current['requests'], strict=True):
        ids = (tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False) if source['chat_template']
            else tokenizer.encode(prompt, add_special_tokens=False))
        if (ids != request['prompt_ids']
                or tokenizer.decode(request['generated_ids'], skip_special_tokens=True) != request['generated_text']):
            raise ValueError('Native text does not reproduce actual model input/output tokens')
    terminal = []
    if 'generation_options' in current:
        options = current['generation_options']
        config_path = Path(hf)/'generation_config.json'
        if digest_file(config_path) != options['generation_config_sha256']:
            raise ValueError('Native generation configuration changed')
        eos = json.loads(config_path.read_text()).get('eos_token_id')
        eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
        if (eos_ids != options['native_eos_token_ids'] or options['do_sample'] is not False
                or type(options['max_new_tokens']) is not int or not 1 <= options['max_new_tokens'] <= 1024
                or any(cache_demand(len(row['prompt_ids']),options['max_new_tokens']) > 1024
                       for row in current['requests'])
                or ('native_cache_capacity' in options and options['native_cache_capacity'] != 1024)):
            raise ValueError('Recorded generation options differ')
        for request in current['requests']:
            tokens = request['generated_ids']
            observed = dict(last_token_is_native_eos=tokens[-1] in eos_ids,
                reached_requested_token_limit=len(tokens) == options['max_new_tokens'],
                generated_tokens=len(tokens))
            if len(tokens) > options['max_new_tokens'] or observed != request['terminal_observation']:
                raise ValueError('Actual output terminal observation differs')
            terminal.append(observed)
    return dict(native_text_retokenized=True, native_output_decoded=True,
                terminal_observations_checked=bool(terminal), requests=terminal)


def admit_reload_control(control):
    """Explicitly admit the earlier native control, not an arbitrary status.

It loaded each capture in a separate DSP session. Reuse its existing lifetime,
native request and capture audits; packet equality below remains mandatory.
"""
    from audit_qwen_native_request_sequence import (
        validate_completed_request_status, request_groups, audit_sessions, audit_capture_execution)
    if (control.get('status') != 'native_Qwen_multiple_requests_with_explicit_DSP_entry_reload'
            or control.get('native_generate') is not True
            or control.get('process_exit_code') != 0
            or control.get('captured_graph_policy') != 'compile_each_actual_capture'
            or control.get('q1_dynamic_library_bindings') != []):
        raise ValueError('Strict actual-capture native reload control required')
    validate_completed_request_status(control)
    request_groups(control)
    sessions = audit_sessions(control)
    def read_linked(row):
        if digest_file(row['path']) != row['sha256']:
            raise ValueError('Historical control linker record changed')
        result = json.loads(Path(row['path']).read_text())
        library = result['library']
        if digest_file(library['path']) != library['sha256']:
            raise ValueError('Historical control calculation library changed')
        return result
    capture = audit_capture_execution(control, read_linked)
    return dict(sessions=len(sessions), capture_compilation=capture)


def audit_rebuilt_artifacts(compiled, producer, capture):
    """Cold rebuilds need their OWN verified artifact chain, not the old key."""
    from hexagon_module_scalar_binding import binding_content_signature
    entry = compiled['compiled_entry']
    frontend, machine = compiled['frontend_cache'], compiled['machine_cache']
    parameters = hashlib.sha256(canonical(producer['immutable_inputs'])).hexdigest()
    front, code = frontend['request'], machine['request']
    if producer.get('guarded_module_constants') is not None:
        from models.qwen3.runtime.support.guarded_module_constants import validate_capture_constants
        validate_capture_constants(producer)
        if front.get('guarded_module_constants_sha256')!=producer['guarded_module_constants']['sha256']:
            raise ValueError('Native module constants differ from actual compiler cache identity')
    if (entry['parameters_sha256'] != parameters or front['parameters_sha256'] != parameters
            or code['parameters_sha256'] != parameters
            or entry['binding_content_sha256'] != binding_content_signature(producer['immutable_inputs'])
            or front['graph'] != producer['graph_identity'] or capture['graph'] != front['graph']
            or code['graph_sha256'] != front['graph']['sha256']
            or code['options']['retained_input_indices'] != producer['retained_input_indices']
            or front['retained_input_indices'] != producer['retained_input_indices']
            or code['options']['function_name'] != front['function_name']
            or frontend['key'] != capture['frontend_key'] or machine['key'] != capture['machine_key']):
        raise ValueError('Rebuilt graph, weights, retained ABI or entry provenance differs')
    for cache in (frontend, machine):
        artifacts = cache['artifacts']
        if not artifacts or describe_artifacts(cache['directory'], [row['path'] for row in artifacts]) != artifacts:
            raise ValueError('Actual rebuilt compiler artifacts changed')
    linked = machine['linked_result']
    if digest_file(linked['path']) != linked['sha256']:
        raise ValueError('Rebuilt linker record changed')
    product = json.loads(Path(linked['path']).read_text())
    if product['library']['sha256'] != capture['DSO_sha256']:
        raise ValueError('Actual executed library differs from rebuilt linker product')


def session_closed(mailbox, completed):
    marker = Path(mailbox)/'session-closed.json'
    if not marker.exists():
        return False
    if marker.is_symlink() or marker.stat().st_size > 16384:
        raise ValueError('Bounded session completion marker required')
    record = json.loads(marker.read_text())
    if (record.get('schema') != 1 or type(record.get('requests_submitted')) is not int
            or record['requests_submitted'] != completed
            or record.get('status') != 'ARM_native_dynamic_compiler_handoff_generate_completed'
            or (Path(mailbox)/f'capture-{completed:02d}'/'ready.json').exists()):
        raise ValueError('ARM run failed or compiler request accounting differs at completion')
    return True


def audit_compile_handoff(current, mailbox, *, rebuilt=False):
    """Read independent request/response artifacts, not just driver summaries."""
    from hexagon_live_compile_mailbox import validate_response, LIVE_STATUS
    if mailbox is None or current.get('live_compiler_handoff') is not True:
        raise ValueError('Live handoff audit requires its actual private mailbox')
    transactions = current['compiler_transactions']
    if len(transactions) != len(current['captures']):
        raise ValueError('Every live capture requires a completed compiler transaction')
    for index, (transaction, capture) in enumerate(zip(transactions, current['captures'], strict=True)):
        folder = Path(mailbox)/f'capture-{index:02d}'
        producer_path, result_path = folder/'result.json', folder/'compiled'/'result.json'
        producer = json.loads(producer_path.read_text())
        response = json.loads((folder/'response.json').read_text())
        compiled = json.loads(result_path.read_text())
        if current.get('generation_options',{}).get('specialize_python_floats') is True:
            from models.qwen3.runtime.support.guarded_module_constants import validate_capture_constants
            validate_capture_constants(producer)
            declaration=producer.get('guarded_module_constants')
            validation=current.get('parameter_validation',{})
            if (declaration is None or validation.get('guarded_python_float_forwards_checked')!=current['DSP_calls']
                    or validation.get('guarded_python_float_constants',{}).get('ledger_sha256')!=declaration['sha256']
                    or compiled['frontend_cache']['request'].get('guarded_module_constants_sha256')!=declaration['sha256']):
                raise ValueError('Actual native/producer/compiler constant guards differ')
        request_sha = digest_file(producer_path)
        if (request_sha != transaction['producer_sha256']
                or response['result_sha256'] != digest_file(result_path)
                or producer['artifact']['path'] != 'graph.pkl'
                or digest_file(folder/'graph.pkl') != producer['artifact']['sha256']
                or producer['status'] != LIVE_STATUS or producer['graph_identity'] != capture['graph']):
            raise ValueError('Actual compiler request or result changed')
        entry = validate_response(response, request_sha, compiled)
        if (current.get('compiler_library_transport') == 'request_scoped_reply'
                or 'returned_library' in response):
            from hexagon_compile_library_reply import checked_reply_library
            checked_reply_library(folder, entry, response=response)
        if rebuilt:
            audit_rebuilt_artifacts(compiled, producer, capture)
        for key in ('graph', 'contract', 'retained_input_indices', 'frontend_key', 'machine_key'):
            if entry[key] != capture[key]:
                raise ValueError('Compiler response differs from executed capture: '+key)
        if (entry['library']['sha256'] != capture['DSO_sha256']
                or digest_file(entry['library']['path']) != capture['DSO_sha256']
                or transaction['frontend_invocations'] != compiled['frontend_invocations']
                or transaction['machine_code_compilations'] != compiled['machine_code_compilations']):
            raise ValueError('Executed library or compiler accounting differs')
    if current['machine_code_compilations'] != sum(t['machine_code_compilations'] for t in transactions):
        raise ValueError('Compiler transaction total differs')
    completion_checked = False
    if 'compile_requests_submitted' in current:
        if (current['compile_requests_submitted'] != len(transactions)
                or not session_closed(mailbox, len(transactions))):
            raise ValueError('Actual live compiler session did not close consistently')
        completion_checked = True
    return dict(actual_requests_and_results_rehashed=True, transactions=len(transactions),
                machine_code_compilations=current['machine_code_compilations'],
                frontend_invocations=sum(t['frontend_invocations'] for t in transactions),
                complete_rebuilt_artifacts_rehashed=rebuilt,
                session_completion_marker_checked=completion_checked)


def first_prefill_control(current, control):
    """Explicit single-forward cold test, not a claim about the whole control."""
    if (len(current['requests']) != 1 or len(current['forwards']) != 1
            or len(current['captures']) != 1 or current['DSP_calls'] != 1):
        raise ValueError('First-prefill selection requires exactly one actual forward')
    request, call = current['requests'][0], current['forwards'][0]
    if (len(request['generated_ids']) != 1 or request['first_forward'] != 0
            or request['end_forward'] != 1 or call['AOT_capture_index'] != 0
            or call['query'] != len(request['prompt_ids'])
            or control['forwards'][0]['AOT_capture_index'] != 0
            or request['prompt_ids'] != control['requests'][0]['prompt_ids']):
        raise ValueError('One complete unchunked prompt and its first generation decision required')
    selected = dict(control)
    for key in ('forwards', 'frontend_cache', 'machine_cache', 'captured_buffer_contracts'):
        selected[key] = control[key][:1]
    selected['requests'] = [dict(control['requests'][0],
        generated_ids=control['requests'][0]['generated_ids'][:1], end_forward=1)]
    return selected


def audit(run_path, run_sha, control_path, control_sha, *, compiler_mailbox=None,
          first_prefill_only=False, allow_rebuilt_artifacts=False, control_entry_reload=False):
    if digest_file(run_path) != run_sha or digest_file(control_path) != control_sha:
        raise ValueError('Pinned completed run/control records required')
    current = json.loads(Path(run_path).read_text())
    if current.get('packet_recording', 'full') != 'full':
        raise ValueError('Full retained packets required; hashes-only is not this audit')
    control = json.loads(Path(control_path).read_text())
    reload_evidence = None
    if control_entry_reload:
        if first_prefill_only:
            raise ValueError('Reload control mode requires the complete native request')
        reload_evidence = admit_reload_control(control)
    if first_prefill_only:
        control = first_prefill_control(current, control)
    handoff = current['status'] == 'ARM_native_dynamic_compiler_handoff_generate_completed'
    if allow_rebuilt_artifacts and not handoff:
        raise ValueError('Rebuilt comparison requires a full actual compiler handoff chain')
    compiler_evidence = audit_compile_handoff(current, compiler_mailbox,
        rebuilt=allow_rebuilt_artifacts) if handoff else None
    if (current['status'] not in ('ARM_native_dynamic_warm_generate_completed',
                                 'ARM_native_dynamic_compiler_handoff_generate_completed')
            or current['architecture'] != 'aarch64' or current['native_generate'] is not True
            or current['CPU_model_fallback'] is not False or current['per_forward_SSH'] is not False
            or current['process_exit_code'] != 0
            or control['status'] != ('native_Qwen_multiple_requests_with_explicit_DSP_entry_reload'
                                    if control_entry_reload else 'native_Qwen_requests_on_one_DSP_process')
            or current['DSP_calls'] != len(current['forwards'])
            or len(current['forwards']) != len(control['forwards'])
            or len(current['requests']) != len(control['requests'])):
        raise ValueError('Matching completed native runs required')
    for actual, expected in zip(current['requests'], control['requests'], strict=True):
        for key in ('prompt_ids', 'generated_ids', 'first_forward', 'end_forward'):
            if actual[key] != expected[key]:
                raise ValueError('Native request/generation sequence changed: '+key)
    if len(current['captures']) != len(control['frontend_cache']):
        raise ValueError('Actual capture count differs')
    rebuild_differences = []
    for capture, frontend, machine, contract in zip(current['captures'], control['frontend_cache'],
            control['machine_cache'], control['captured_buffer_contracts'], strict=True):
        if (capture['graph'] != frontend['request']['graph']
                or capture['retained_input_indices'] != frontend['request']['retained_input_indices']
                or capture['frontend_key'] != frontend['key']
                or (not allow_rebuilt_artifacts and capture['machine_key'] != machine['key'])
                or capture['contract'] != contract['contract']):
            raise ValueError('Actual graph or compiled product binding changed')
        if capture['machine_key'] != machine['key']:
            rebuild_differences.append(dict(graph=capture['graph'],
                control_machine_key=machine['key'], actual_machine_key=capture['machine_key']))
    rows = []
    for step, (actual, expected) in enumerate(zip(current['forwards'], control['forwards'], strict=True)):
        if type(actual['query']) is not int or not 1 <= actual['query'] <= 32 or actual['step'] != step:
            raise ValueError('Invalid actual forward ordinal/query')
        for key in ('query', 'AOT_capture_index', 'DSO_sha256', 'cache_positions', 'input_token_ids', 'logits_top_token'):
            if key == 'DSO_sha256' and allow_rebuilt_artifacts:
                if actual[key] != current['captures'][actual['AOT_capture_index']]['DSO_sha256']:
                    raise ValueError('Invocation differs from independently verified rebuilt library')
                continue
            if actual[key] != expected[key]:
                raise ValueError('Actual forward/control differs: '+key)
        pair = dict(step=step, query=actual['query'])
        for side in ('input', 'output'):
            name = f'forward-{step:02d}-q{actual["query"]}/{side}.bin'
            left, right = Path(run_path).parent/name, Path(control_path).parent/name
            if (left.is_symlink() or right.is_symlink() or left.stat().st_size != right.stat().st_size
                    or left.stat().st_size != actual['invoke'][side+'_bytes']):
                raise ValueError('Complete actual packet files required')
            a, b = digest_file(left), digest_file(right)
            if a != actual[side+'_sha256'] or b != expected[side+'_sha256'] or a != b:
                raise ValueError(f'Actual complete {side} packet changed at step {step}')
            pair[side] = dict(bytes=left.stat().st_size, sha256=a, control_sha256=b)
        rows.append(pair)
    lifecycle = audit_library_events(current['stream_events'],
        [row['DSP_library_slot'] for row in current['forwards']])
    transport = audit_local_packet_transport(current)
    if [event for event in current['stream_events'] if event.get('stage') == 'invoke'] != [
            row['invoke'] for row in current['forwards']]:
        raise ValueError('Recorded forwards do not match actual invocation events')
    return dict(status=('ARM_native_compiler_handoff_complete_packets_bitexact_to_control' if handoff
                        else 'ARM_native_warm_run_complete_packets_bitexact_to_control'),
        compiler_handoff=compiler_evidence,
        control_reload_evidence=reload_evidence,
        rebuilt_artifact_comparison=allow_rebuilt_artifacts, rebuild_differences=rebuild_differences,
        control_selection='first_prefill_only' if first_prefill_only else 'complete_run',
        actual_packet_bytes_rehashed=True, forwards=rows, library_lifecycle=lifecycle,
        local_packet_transport=transport,
        arm_forward_timing=audit_arm_forward_timings(current),
        requests=len(current['requests']), actual_AOT_captures=len(current['captures']),
        numerical_acceptance=False, independent_CPU_quality_acceptance=False,
        performance_acceptance=False, DSP_resident_KV=False,
        automatic_cold_compilation=bool(compiler_evidence and compiler_evidence['machine_code_compilations'] > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--run-sha256', required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--control-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler-mailbox', type=Path)
    parser.add_argument('--hf', type=Path, help='Additionally rehash native tokenizer and verify text/stop metadata')
    parser.add_argument('--control-entry-reload', action='store_true',
                        help='Explicit earlier native control; recheck all session/capture ownership')
    parser.add_argument('--first-prefill-only', action='store_true',
                        help='Explicitly compare one complete prompt/one decision to the control first forward')
    parser.add_argument('--allow-rebuilt-artifacts', action='store_true',
                        help='Verify the entire new compiler artifact chain; allow new machine key/library hash')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    result = dict(status='checking', source_sha256=digest_file(__file__),
        run=dict(path=str(args.run), sha256=args.run_sha256),
        control=dict(path=str(args.control), sha256=args.control_sha256))
    try:
        result.update(audit(args.run, args.run_sha256, args.control, args.control_sha256,
                            compiler_mailbox=args.compiler_mailbox,
                            first_prefill_only=args.first_prefill_only,
                            allow_rebuilt_artifacts=args.allow_rebuilt_artifacts,
                            control_entry_reload=args.control_entry_reload))
        if args.hf:
            result['native_text'] = audit_text_entry(json.loads(args.run.read_text()), args.hf)
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
