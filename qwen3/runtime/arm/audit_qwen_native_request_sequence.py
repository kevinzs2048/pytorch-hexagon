"""Audit native sequential requests, cache continuity and DSP entry lifetimes.

CPU controls are independent generate() runs, not a shared handwritten token
loop. This checks packets and lifecycle; it cannot accept model numerics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hexagon_compiled_artifact_cache import canonical, digest_file
from hexagon_host_timing import audit_host_forward_timings
from hexagon_buffer_transport import buffer_capacities_for_contract
from prepare_qwen_dynamic_debug_caller import debug_contract, profiles_and_types
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import MAGIC, layout_for
from qwen_native_request_sequence import verify_request
from capture_qwen_symbolic_graph_carrier import CONTEXT_SHA


def validate_completed_request_status(record):
    if record['status'] not in ('native_Qwen_multiple_requests_on_one_DSP_DSO',
            'native_Qwen_multiple_requests_with_explicit_DSP_entry_reload',
            'native_Qwen_requests_on_one_DSP_process',
            'native_Qwen_generic_prompt_generate_on_one_DSP_DSO'):
        raise ValueError('Completed native request status required')
    if (record['status']=='native_Qwen_requests_on_one_DSP_process') != bool(record.get('library_switching')):
        raise ValueError('Native request status and library lifetime mode differ')
    if record['status'] == 'native_Qwen_generic_prompt_generate_on_one_DSP_DSO':
        if (len(record['requests']) != 1
                or any(record[key] != record['requests'][0][key]
                       for key in ('prompt_ids', 'generated_ids'))):
            raise ValueError('Single native request summary differs from actual request')


def request_groups(record):
    rows = record['requests']
    forwards = record['forwards']
    if not rows or record['DSP_calls'] != len(forwards):
        raise ValueError('Completed native requests and all DSP calls required')
    start = 0
    groups = []
    for index, row in enumerate(rows):
        end = row['end_forward']
        if (row['index'] != index or row['first_forward'] != start
                or type(end) is not int or not start < end <= len(forwards)
                or not row['native_cache_object_reused']):
            raise ValueError('Sequential request intervals or native cache identity differ')
        group = forwards[start:end]
        if any(part['request_index'] != index or part['step'] != start+i
               for i, part in enumerate(group)):
            raise ValueError('Forward request ownership differs')
        verify_request(group, row['prompt_ids'], row['generated_ids'])
        groups.append(group)
        start = end
    if start != len(forwards):
        raise ValueError('Unowned DSP forwards remain')
    return groups


def compare_native_controls(row, group, cpu):
    """Keep numerical token divergence separate from broken request transport."""
    control = [dict(query=part['query'], cache_positions=part['positions'],
                    input_token_ids=part['token_ids']) for part in cpu['forwards']]
    verify_request(control, cpu['prompt_ids'], cpu['generated_ids'])
    dsp_ids, cpu_ids = row['generated_ids'], cpu['generated_ids']
    common = min(len(dsp_ids), len(cpu_ids))
    difference = next((index for index in range(common) if dsp_ids[index] != cpu_ids[index]), None)
    if difference is None and len(dsp_ids) != len(cpu_ids):
        difference = common
    return dict(same_generated_ids=dsp_ids == cpu_ids,
        first_generated_token_difference=difference,
        same_native_shape_position_trace=[(part['query'], part['cache_positions']) for part in group]
            == [(part['query'], part['cache_positions']) for part in control],
        same_native_input_token_trace=[part['input_token_ids'] for part in group]
            == [part['input_token_ids'] for part in control])


def audit_sessions(record):
    """Every load owns a contiguous interval of completed forward calls."""
    if record.get('library_switching'):
        from hexagon_library_lifecycle import audit_library_events
        sessions=record['DSP_sessions'];slots=record['DSP_library_slots'];forwards=record['forwards']
        if (len(sessions)!=1 or not forwards or not 1<=len(slots)<=4
                or sessions[0]['index']!=0 or sessions[0]['first_forward']!=0
                or sessions[0]['end_forward']!=len(forwards) or record['DSP_calls']!=len(forwards)
                or sessions[0]['process_exit_code']!=0
                or sessions[0]['events']!=record['stream_events']
                or [row['index'] for row in slots]!=list(range(len(slots)))
                or len({row['sha256'] for row in slots})!=len(slots)):
            raise ValueError('One-process library ownership/coverage differs')
        trace=audit_library_events(record['stream_events'],[row['DSP_library_slot'] for row in forwards])
        if trace['loaded_libraries']!=len(slots):
            raise ValueError('Loaded library table differs from actual events')
        for row in forwards:
            index=row['DSP_library_slot']
            if (type(index) is not int or not 0<=index<len(slots)
                    or row['DSP_session']!=0 or row['DSO_sha256']!=slots[index]['sha256']):
                raise ValueError('Forward executed a different owned library')
        last=slots[forwards[-1]['DSP_library_slot']]
        if (record['selected_DSO_sha256']!=last['sha256']
                or record['loaded_library_source']!=last['library']
                or sessions[0]['sha256']!=slots[0]['sha256']
                or sessions[0]['library']!=slots[0]['library']):
            raise ValueError('Initial/final library identity differs')
        return sessions
    sessions = record.get('DSP_sessions')
    if sessions is None:
        sessions = [dict(index=0, library=record['loaded_library_source'],
            sha256=record['selected_DSO_sha256'], first_forward=0,
            end_forward=record['DSP_calls'], events=record['stream_events'],
            process_exit_code=record['process_exit_code'])]
    if not sessions:
        raise ValueError('At least one completed DSP session required')
    expected_start = 0
    flat = []
    for index, session in enumerate(sessions):
        end = session['end_forward']
        if (session['index'] != index or session['first_forward'] != expected_start
                or type(end) is not int
                or not expected_start < end <= record['DSP_calls']
                or session['process_exit_code'] != 0
                or [row['stage'] for row in session['events']] !=
                   ['open', 'load', 'ready', *(['invoke']*(end-expected_start)), 'unload', 'close']):
            raise ValueError('DSP load lifetime or completed-forward boundary differs')
        if 'DSP_sessions' in record:
            for forward in record['forwards'][expected_start:end]:
                if forward['DSP_session'] != index or forward['DSO_sha256'] != session['sha256']:
                    raise ValueError('Forward owned by different DSP entry')
        expected_start = end
        flat.extend(session['events'])
    if (expected_start != record['DSP_calls'] or flat != record['stream_events']
            or sessions[-1]['sha256'] != record['selected_DSO_sha256']
            or sessions[-1]['library'] != record['loaded_library_source']):
        raise ValueError('DSP session coverage differs')
    return sessions


def audit_session_capacities(record, sessions, contract):
    """The contract and actual ready events must agree, not only the run JSON."""
    if 'buffer_capacities' not in record:
        return None  # Historical records predate contract-derived capacities.
    expected = buffer_capacities_for_contract(contract)
    if record['buffer_capacities'] != expected:
        raise ValueError('Recorded capacities differ from selected tensor contract')
    for session in sessions:
        ready = [row for row in session['events'] if row['stage'] == 'ready']
        if len(ready) != 1 or any(type(ready[0].get(key)) is not int
                                 or ready[0][key] != value for key, value in expected.items()):
            raise ValueError('Actual runtime capacities differ from tensor contract')
    return expected


def audit_capture_execution(record, read_linked):
    """Independently bind each actual forward to its captured graph's library."""
    if record.get('captured_graph_policy') != 'compile_each_actual_capture':
        return None
    captures = record['AOT_compiles']
    if (not captures or record['q1_dynamic_library_bindings']
            or any(len(record[key]) != len(captures) for key in
                   ('frontend_cache', 'machine_cache', 'captured_buffer_contracts'))):
        raise ValueError('Incomplete all-capture compilation provenance')
    libraries = []
    for capture, frontend, machine in zip(captures, record['frontend_cache'], record['machine_cache']):
        if (frontend['request']['graph']['sha256'] != machine['request']['graph_sha256']
                or frontend['request']['parameters_sha256'] != machine['request']['parameters_sha256']
                or frontend['request']['function_name'] != machine['request']['options']['function_name']
                or frontend['request']['retained_input_indices']
                    != machine['request']['options']['retained_input_indices']
                or len(frontend['request']['retained_input_indices']) != capture['retained_inputs']):
            raise ValueError('Captured/frontend/machine identity mismatch (graph, weights or interface)')
        linked = read_linked(machine['linked_result'])
        libraries.append(linked['library']['sha256'])
    seen = set()
    for forward in record['forwards']:
        index = forward.get('AOT_capture_index')
        if type(index) is not int or not 0 <= index < len(captures):
            raise ValueError('Forward lacks its actual AOT capture identity')
        lo, hi = record['captured_buffer_contracts'][index]['contract']['variables']['query']
        if (forward['DSO_sha256'] != libraries[index] or not lo <= forward['query'] <= hi):
            raise ValueError('Forward executes a different library or exceeds its captured shape range')
        seen.add(index)
    if seen != set(range(len(captures))):
        raise ValueError('Some compiled captures have no actual DSP execution')
    return dict(captures=len(captures), every_forward_uses_its_captured_artifact=True,
        captured_shape_ranges_respected=True, legacy_decode_rebindings=0)


def audit_unchanged_kv(incoming, outgoing, layout, positions):
    """A functional cache update may change only this request's active slots."""
    if not positions or positions != list(range(positions[0], positions[0]+len(positions))):
        raise ValueError('Contiguous cache update positions required')
    inputs = layout['inputs'][3:5]+layout['inputs'][6:]
    outputs = layout['outputs'][:56]
    if len(inputs) != 56 or len(outputs) != 56:
        raise ValueError('Selected Qwen KV packet ABI differs')
    start, end = positions[0], positions[-1]+1
    checked_bytes = 0
    for source, target in zip(inputs, outputs):
        if source['shape'] != target['shape'] or source['shape'] != [1, 8, 1024, 128]:
            raise ValueError('Selected Qwen KV shape differs')
        if not 0 <= start < end <= source['shape'][2]:
            raise ValueError('Cache update outside capacity')
        before = np.frombuffer(incoming[16+source['offset']:16+source['offset']+source['bytes']],
                               dtype='<u4').reshape(source['shape'])
        after = np.frombuffer(outgoing[target['offset']:target['offset']+target['bytes']],
                              dtype='<u4').reshape(target['shape'])
        if not np.isfinite(after.view('<f4')[:, :, start:end, :]).all():
            raise ValueError('DSP returned nonfinite updated KV')
        for region in (slice(0, start), slice(end, source['shape'][2])):
            old, new = before[:, :, region, :], after[:, :, region, :]
            if not np.array_equal(old, new):
                raise ValueError('DSP modified KV outside the active update positions')
            checked_bytes += old.nbytes
    return dict(unchanged_history_and_future_bitexact=True, updated_KV_finite=True, checked_bytes=checked_bytes,
                update_positions=positions)


def audit_logits(outgoing, layout, recorded_top):
    row = layout['outputs'][-1]
    logits = np.frombuffer(outgoing[row['offset']:row['offset']+row['bytes']], dtype='<f4')
    if logits.size != np.prod(row['shape']) or not logits.size or not np.isfinite(logits).all():
        raise ValueError('DSP logits are nonfinite or have the wrong size')
    if int(logits.argmax()) != recorded_top:
        raise ValueError('Recorded DSP top token differs from returned logits')
    return dict(all_logits_finite=True, recorded_top_matches_raw_logits=True)


def kv_packet_digests(packet, layout, *, side):
    """Hash every cache tensor, excluding framing, mask, positions and logits."""
    if side == 'input':
        rows, prefix = layout['inputs'][3:5]+layout['inputs'][6:], 16
    elif side == 'output':
        rows, prefix = layout['outputs'][:56], 0
    else:
        raise ValueError('Input or output packet side required')
    if len(rows) != 56:
        raise ValueError('Selected Qwen cache tensor cardinality differs')
    result = []
    for row in rows:
        start, size = prefix+row['offset'], row['bytes']
        if start < 0 or size <= 0 or start+size > len(packet):
            raise ValueError('Cache tensor extends outside actual packet')
        result.append(dict(shape=row['shape'], bytes=size,
            sha256=hashlib.sha256(packet[start:start+size]).hexdigest()))
    return result


def audit_cache_handoff(previous_output, incoming, layout):
    if kv_packet_digests(incoming, layout, side='input') != previous_output:
        raise ValueError('Next native forward did not receive the previous DSP KV bytes')
    return dict(previous_DSP_KV_reused_bitexact=True,
                checked_bytes=sum(row['bytes'] for row in previous_output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsp', type=Path, required=True)
    parser.add_argument('--dsp-sha256', required=True)
    parser.add_argument('--cpu-results', type=Path, nargs='+', required=True)
    parser.add_argument('--cpu-sha256', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--caller', type=Path, required=True,
                        help='Explicit compiled caller/ABI receipt; no historical directory lookup')
    parser.add_argument('--caller-sha256', required=True)
    args = parser.parse_args()
    if (args.output.exists() or len(args.cpu_results) != len(args.cpu_sha256)
            or digest_file(args.dsp) != args.dsp_sha256
            or digest_file(args.caller) != args.caller_sha256):
        raise ValueError('Fresh result and pinned run/ABI records required')
    dsp = json.loads(args.dsp.read_text())
    validate_completed_request_status(dsp)
    if (not dsp['native_generate'] or dsp['process_exit_code'] != 0
            or digest_file(dsp['loaded_library_source']) != dsp['selected_DSO_sha256']
            or len(dsp['requests']) != len(args.cpu_results)):
        raise ValueError('Completed native model/DSP library lifetimes required')
    sessions = audit_sessions(dsp)
    executed_libraries=dsp['DSP_library_slots'] if dsp.get('library_switching') else sessions
    for session in executed_libraries:
        if digest_file(session['library']) != session['sha256']:
            raise ValueError('Executed DSP entry changed')
    def read_linked(row):
        if digest_file(row['path']) != row['sha256']:
            raise ValueError('Captured graph linked-product record changed')
        linked = json.loads(Path(row['path']).read_text())
        if digest_file(linked['library']['path']) != linked['library']['sha256']:
            raise ValueError('Captured graph compiled library changed')
        return linked
    capture_execution = audit_capture_execution(dsp, read_linked)
    host_timing = audit_host_forward_timings(dsp)
    groups = request_groups(dsp)
    _, types = profiles_and_types(json.loads(args.caller.read_text()))
    contract = debug_contract(types, [])
    buffer_capacities = audit_session_capacities(dsp, sessions, contract)
    results, previous = [], {}
    for row, group, cpu_path, expected in zip(dsp['requests'], groups, args.cpu_results, args.cpu_sha256):
        if digest_file(cpu_path) != expected:
            raise ValueError('Independent CPU control record changed')
        cpu = json.loads(cpu_path.read_text())
        if (cpu['status'] != 'native_generic_CPU_generate_completed'
                or cpu['context_sha256'] != CONTEXT_SHA
                or cpu['prompt_ids'] != row['prompt_ids']
                or cpu['prefill_chunk_size'] != dsp['prefill_chunk_size']
                or cpu['max_new_tokens'] != dsp['max_new_tokens']):
            raise ValueError('Independent CPU request control differs')
        comparison = compare_native_controls(row, group, cpu)
        cache_transitions = []
        previous_output = None
        for position, forward in enumerate(group):
            query = forward['query']
            folder = args.dsp.parent/f"forward-{forward['step']:02d}-q{query}"
            incoming, outgoing = folder/'input.bin', folder/'output.bin'
            layout = layout_for(contract, {'query': query})
            if (digest_file(incoming) != forward['input_sha256']
                    or digest_file(outgoing) != forward['output_sha256']
                    or incoming.stat().st_size != 16+layout['input_bytes']
                    or outgoing.stat().st_size != layout['output_bytes']):
                raise ValueError('Actual sequential DSP packet differs')
            # Reading sparse tmpfs holes through mmap may instantiate pages in
            # the FILE and exhaust its filesystem. read() keeps audit buffers
            # anonymous; it does not change the packet's allocation or bytes.
            raw = np.frombuffer(incoming.read_bytes(), dtype=np.uint8)
            output_raw = np.frombuffer(outgoing.read_bytes(), dtype=np.uint8)
            transition = audit_unchanged_kv(raw, output_raw, layout, forward['cache_positions'])
            transition['logits'] = audit_logits(output_raw, layout, forward['logits_top_token'])
            transition['handoff'] = (None if previous_output is None else
                audit_cache_handoff(previous_output, raw, layout))
            previous_output = kv_packet_digests(output_raw, layout, side='output')
            cache_transitions.append(transition)
            del output_raw
            if raw[:8].tobytes() != MAGIC or int.from_bytes(raw[8:16], 'little', signed=True) != query:
                raise ValueError('Sequential DSP packet framing differs')
            values = []
            for item in layout['inputs'][:3]:
                offset = 16+item['offset']
                values.append(np.frombuffer(raw[offset:offset+item['bytes']], dtype='<i8').tolist())
            if values != [forward['input_token_ids'], forward['cache_positions'], forward['cache_positions']]:
                raise ValueError('Actual request token/position bytes differ')
            if position == 0:
                for item in layout['inputs'][3:5]+layout['inputs'][6:]:
                    offset = 16+item['offset']
                    if np.count_nonzero(np.frombuffer(raw[offset:offset+item['bytes']], dtype='<f4')):
                        raise ValueError('Native cache reset left nonzero KV in a new request')
            del raw
        key = tuple(row['prompt_ids'])
        packet_hashes = [(f['input_sha256'], f['output_sha256']) for f in group]
        repeat = None if key not in previous else packet_hashes == previous[key]
        previous[key] = packet_hashes
        results.append(dict(index=row['index'], prompt_ids=row['prompt_ids'],
            CPU_result_sha256=expected, CPU_generated_ids=cpu['generated_ids'],
            DSP_generated_ids=row['generated_ids'], **comparison,
            initial_KV_all_zero=True, repeated_request_packets_bitexact=repeat,
            cache_transitions=cache_transitions,
            forwards=len(group), AOT_compiles_after_request=row['AOT_compiles_after_request']))
    record = dict(status='native_sequential_request_packets_and_cache_reset_audited_NOT_numerics',
        source_sha256=digest_file(__file__), DSP_result_sha256=args.dsp_sha256,
        selected_DSO_sha256=dsp['selected_DSO_sha256'], requests=results,
        one_library_load=len(executed_libraries) == 1, library_loads=len(executed_libraries),
        AP_processes=len(sessions),
        machine_code_compilations=dsp['machine_code_compilations'],
        DSP_calls=dsp['DSP_calls'], numerical_acceptance=False)
    if capture_execution is not None:
        record['capture_execution'] = capture_execution
    if host_timing is not None:
        record['host_forward_timing'] = host_timing
    if buffer_capacities is not None:
        record['buffer_capacities'] = buffer_capacities
    args.output.mkdir()
    (args.output/'result.json').write_bytes(canonical(record)+b'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
