import copy
import json

import pytest

from audit_arm_dynamic_warm_run import (
    audit, audit_compile_handoff, first_prefill_control, audit_rebuilt_artifacts, admit_reload_control,
    audit_local_packet_transport)
from hexagon_compiled_artifact_cache import digest_file
from test_hexagon_library_lifecycle import events


@pytest.mark.parametrize('mode,flag,valid', [('files', False, True), ('shared-memory', True, True),
    ('files', True, False), ('shared-memory', False, False), ('shared-memory', 1, False), ('unknown', False, False)])
def test_packet_transport_requires_actual_ap_agreement(mode, flag, valid):
    run = dict(local_packet_transport=mode,
               stream_events=[dict(stage='ready', shared_memory_packets=flag)])
    if valid:
        assert audit_local_packet_transport(run)['mode'] == mode
    else:
        with pytest.raises(ValueError):
            audit_local_packet_transport(run)


def files(tmp_path):
    run_root, control_root = tmp_path/'run', tmp_path/'control'
    run_root.mkdir(); control_root.mkdir()
    calls = [x for x in events() if x['stage'] == 'invoke']
    rows = []
    for index, slot in enumerate((0, 1, 0)):
        query = 5 if slot == 0 else 1
        row = dict(step=index, query=query, AOT_capture_index=slot,
            DSO_sha256=str(slot), cache_positions=[index], input_token_ids=[10],
            logits_top_token=11, DSP_library_slot=slot, invoke=calls[index])
        for root in (run_root, control_root):
            folder = root/f'forward-{index:02d}-q{query}'
            folder.mkdir()
            for side, size in (('input', 64), ('output', 128)):
                path = folder/(side+'.bin')
                path.write_bytes(bytes([index])*size)
                row[side+'_sha256'] = digest_file(path)
        rows.append(row)
    captures = [dict(graph={'sha256': str(i)}, retained_input_indices=[1],
        frontend_key='front'+str(i), machine_key='machine'+str(i), contract={'query': i}) for i in range(2)]
    requests = [dict(prompt_ids=[10], generated_ids=[11, 12, 13], first_forward=0, end_forward=3)]
    current = dict(status='ARM_native_dynamic_warm_generate_completed', architecture='aarch64',
        native_generate=True, CPU_model_fallback=False, per_forward_SSH=False,
        process_exit_code=0, DSP_calls=3, forwards=rows, requests=requests, captures=captures,
        stream_events=events())
    control = dict(status='native_Qwen_requests_on_one_DSP_process', forwards=copy.deepcopy(rows),
        requests=requests, frontend_cache=[dict(key=c['frontend_key'], request=dict(graph=c['graph'],
            retained_input_indices=[1])) for c in captures],
        machine_cache=[{'key': c['machine_key']} for c in captures],
        captured_buffer_contracts=[{'contract': c['contract']} for c in captures])
    return run_root, control_root, current, control


def check(data):
    root, control_root, current, control = data
    run, baseline = root/'result.json', control_root/'result.json'
    run.write_text(json.dumps(current)); baseline.write_text(json.dumps(control))
    return audit(run, digest_file(run), baseline, digest_file(baseline))


def test_complete_raw_packets_and_lifecycle_are_rechecked(tmp_path):
    result = check(files(tmp_path))
    assert result['actual_packet_bytes_rehashed'] is True
    assert len(result['forwards']) == 3
    assert result['numerical_acceptance'] is False


@pytest.mark.parametrize('fault', ['raw_bytes', 'contract', 'capture', 'event', 'packet_length'])
def test_record_agreement_is_insufficient_for_pass(tmp_path, fault):
    data = files(tmp_path)
    root, _, record, _ = data
    if fault == 'raw_bytes':
        (root/'forward-00-q5/output.bin').write_bytes(b'X'*128)
    elif fault == 'contract':
        record['captures'][0]['contract'] = {'other': True}
    elif fault == 'capture':
        record['captures'][0]['graph'] = {'sha256': 'wrong'}
    elif fault == 'event':
        record['forwards'][0]['invoke']['kernel_us'] += 1
    else:
        record['forwards'][0]['invoke']['input_bytes'] = 63
    with pytest.raises(ValueError):
        check(data)


def handoff_files(tmp_path):
    from hexagon_live_compile_mailbox import LIVE_STATUS
    root, _, current, _ = files(tmp_path)
    mailbox = tmp_path/'mailbox'
    mailbox.mkdir()
    current.update(live_compiler_handoff=True, compiler_transactions=[], machine_code_compilations=0)
    for index, capture in enumerate(current['captures']):
        folder = mailbox/f'capture-{index:02d}'
        folder.mkdir(); (folder/'compiled').mkdir()
        graph = folder/'graph.pkl'
        graph.write_bytes(b'synthetic graph; never unpickled by auditor')
        library = folder/'code.so'
        library.write_bytes(b'code'+bytes([index]))
        capture['DSO_sha256'] = digest_file(library)
        producer = dict(status=LIVE_STATUS, graph_identity=capture['graph'],
                        artifact={'path': 'graph.pkl', 'sha256': digest_file(graph)})
        (folder/'result.json').write_text(json.dumps(producer))
        producer_sha = digest_file(folder/'result.json')
        compiled = dict(status='carried_ARM_Qwen_graph_bound_to_Hexagon_library_NOT_DSP',
            producer={'sha256': producer_sha}, DSP_calls=0, CPU_model_forward=False,
            compiled_entry=dict(capture, library={'path': str(library), 'sha256': digest_file(library)}),
            frontend_invocations=0, machine_code_compilations=0)
        path = folder/'compiled'/'result.json'
        path.write_text(json.dumps(compiled))
        (folder/'response.json').write_text(json.dumps(dict(schema=1, producer_sha256=producer_sha,
            result_sha256=digest_file(path), status='compiled')))
        current['compiler_transactions'].append(dict(producer_sha256=producer_sha,
            frontend_invocations=0, machine_code_compilations=0))
    return current, mailbox


def test_handoff_audits_actual_graphs_products_and_accounting(tmp_path):
    current, mailbox = handoff_files(tmp_path)
    result = audit_compile_handoff(current, mailbox)
    assert result['actual_requests_and_results_rehashed'] is True
    assert result['transactions'] == 2


@pytest.mark.parametrize('fault', [None, 'missing', 'failed', 'count'])
def test_handoff_checks_actual_session_completion(tmp_path, fault):
    current, mailbox = handoff_files(tmp_path)
    current['compile_requests_submitted'] = 2
    marker = dict(schema=1, requests_submitted=2,
                  status='ARM_native_dynamic_compiler_handoff_generate_completed')
    if fault == 'failed':
        marker['status'] = 'failed'
    elif fault == 'count':
        marker['requests_submitted'] = 1
    if fault != 'missing':
        (mailbox/'session-closed.json').write_text(json.dumps(marker))
    if fault is None:
        assert audit_compile_handoff(current, mailbox)['session_completion_marker_checked'] is True
    else:
        with pytest.raises(ValueError):
            audit_compile_handoff(current, mailbox)


def test_new_reply_transport_requires_actual_returned_library(tmp_path):
    current, mailbox = handoff_files(tmp_path)
    current['compiler_library_transport'] = 'request_scoped_reply'
    with pytest.raises(ValueError, match='Returned library'):
        audit_compile_handoff(current, mailbox)


@pytest.mark.parametrize('field,value', [('status', 'other'), ('native_generate', False),
    ('process_exit_code', 1), ('captured_graph_policy', 'legacy'),
    ('q1_dynamic_library_bindings', [{}])])
def test_reload_control_cannot_relax_graph_ownership(field, value):
    control = dict(status='native_Qwen_multiple_requests_with_explicit_DSP_entry_reload',
        native_generate=True, process_exit_code=0, captured_graph_policy='compile_each_actual_capture',
        q1_dynamic_library_bindings=[])
    control[field] = value
    with pytest.raises(ValueError, match='Strict actual-capture'):
        admit_reload_control(control)


def test_reload_control_reuses_all_existing_provenance_checks(monkeypatch):
    import audit_qwen_native_request_sequence as existing
    control = dict(status='native_Qwen_multiple_requests_with_explicit_DSP_entry_reload',
        native_generate=True, process_exit_code=0, captured_graph_policy='compile_each_actual_capture',
        q1_dynamic_library_bindings=[])
    calls = []
    def check(name, result):
        def inner(record, *args):
            assert record is control
            calls.append(name)
            return result
        return inner
    monkeypatch.setattr(existing, 'validate_completed_request_status', check('status', None))
    monkeypatch.setattr(existing, 'request_groups', check('requests', []))
    monkeypatch.setattr(existing, 'audit_sessions', check('sessions', [{}, {}]))
    monkeypatch.setattr(existing, 'audit_capture_execution', check('captures', {'checked': True}))
    assert admit_reload_control(control) == dict(sessions=2, capture_compilation={'checked': True})
    assert calls == ['status', 'requests', 'sessions', 'captures']


@pytest.mark.parametrize('fault', ['graph', 'library', 'transaction', 'count', 'absent'])
def test_handoff_record_agreement_does_not_replace_evidence(tmp_path, fault):
    current, mailbox = handoff_files(tmp_path)
    if fault == 'graph':
        (mailbox/'capture-00'/'graph.pkl').write_bytes(b'changed')
    elif fault == 'library':
        (mailbox/'capture-00'/'code.so').write_bytes(b'changed')
    elif fault == 'transaction':
        current['compiler_transactions'][0]['producer_sha256'] = 'bad'
    elif fault == 'count':
        current['machine_code_compilations'] = 1
    else:
        mailbox = None
    with pytest.raises(ValueError):
        audit_compile_handoff(current, mailbox)


def first_prefill_data(tmp_path):
    _, _, current, control = files(tmp_path)
    current['DSP_calls'] = 1
    current['forwards'] = current['forwards'][:1]
    current['captures'] = current['captures'][:1]
    current['requests'] = [dict(prompt_ids=[1, 2, 3, 4, 5], generated_ids=[11],
                               first_forward=0, end_forward=1)]
    control['requests'][0]['prompt_ids'] = [1, 2, 3, 4, 5]
    return current, control


def test_single_prefill_selection_is_explicit_and_does_not_mutate_control(tmp_path):
    current, control = first_prefill_data(tmp_path)
    before = copy.deepcopy(control)
    selected = first_prefill_control(current, control)
    assert len(selected['forwards']) == 1
    assert selected['requests'] == current['requests']
    assert control == before


@pytest.mark.parametrize('fault', ['extra_forward', 'chunked', 'generation', 'offset'])
def test_prefill_selection_cannot_hide_unchecked_calls(tmp_path, fault):
    current, control = first_prefill_data(tmp_path)
    if fault == 'extra_forward':
        current['forwards'] *= 2
    elif fault == 'chunked':
        current['forwards'][0]['query'] = 2
    elif fault == 'generation':
        current['requests'][0]['generated_ids'].append(12)
    else:
        current['requests'][0]['first_forward'] = 1
    with pytest.raises(ValueError):
        first_prefill_control(current, control)


def rebuilt_fixture(tmp_path):
    import hashlib
    from hexagon_compiled_artifact_cache import canonical, describe_artifacts
    from hexagon_module_scalar_binding import binding_content_signature
    front, machine = tmp_path/'front', tmp_path/'machine'
    front.mkdir(); machine.mkdir()
    (front/'graph.mlirbc').write_bytes(b'synthetic bytecode')
    (machine/'code.so').write_bytes(b'synthetic code')
    library = dict(path=str(machine/'code.so'), sha256=digest_file(machine/'code.so'))
    linked = machine/'linked.json'
    linked.write_text(json.dumps({'library': library}))
    producer = dict(immutable_inputs=[], retained_input_indices=[0], graph_identity={'sha256': 'g'})
    parameters = hashlib.sha256(canonical([])).hexdigest()
    capture = dict(graph=producer['graph_identity'], frontend_key='front', machine_key='new',
                   DSO_sha256=library['sha256'])
    request = dict(graph=producer['graph_identity'], parameters_sha256=parameters,
                   retained_input_indices=[0], function_name='forward')
    compiled = dict(compiled_entry=dict(parameters_sha256=parameters,
                        binding_content_sha256=binding_content_signature([])),
        frontend_cache=dict(directory=str(front), key='front', request=request,
                            artifacts=describe_artifacts(front, ['graph.mlirbc'])),
        machine_cache=dict(directory=str(machine), key='new',
            request=dict(graph_sha256='g', parameters_sha256=parameters,
                         options={'retained_input_indices': [0], 'function_name': 'forward'}),
            artifacts=describe_artifacts(machine, ['code.so', 'linked.json']),
            linked_result={'path': str(linked), 'sha256': digest_file(linked)}))
    return compiled, producer, capture


def test_rebuilt_artifact_chain_rehashes_new_provenance(tmp_path):
    audit_rebuilt_artifacts(*rebuilt_fixture(tmp_path))


@pytest.mark.parametrize('fault', ['front_bytes', 'code_bytes', 'weights', 'graph', 'retained', 'entry', 'key'])
def test_rebuild_option_does_not_relax_actual_compiler_integrity(tmp_path, fault):
    compiled, producer, capture = rebuilt_fixture(tmp_path)
    if fault == 'front_bytes':
        (tmp_path/'front'/'graph.mlirbc').write_bytes(b'changed')
    elif fault == 'code_bytes':
        (tmp_path/'machine'/'code.so').write_bytes(b'changed')
    elif fault == 'weights':
        compiled['machine_cache']['request']['parameters_sha256'] = 'bad'
    elif fault == 'graph':
        compiled['machine_cache']['request']['graph_sha256'] = 'bad'
    elif fault == 'retained':
        compiled['machine_cache']['request']['options']['retained_input_indices'] = [1]
    elif fault == 'entry':
        compiled['machine_cache']['request']['options']['function_name'] = 'bad'
    else:
        capture['machine_key'] = 'bad'
    with pytest.raises(ValueError):
        audit_rebuilt_artifacts(compiled, producer, capture)
