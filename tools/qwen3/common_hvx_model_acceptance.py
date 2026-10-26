"""Bind a changed numerical model to its actually measured common HVX runtime.

The earlier unchanged-model transport comparison stays historical. A full
native393 has nine scalar cold numerical calls and 384 common-HVX calls:
352 changed-model calls and 32 paired old-model calls. It is not a new
unchanged-model transport speed comparison or a scalar384 for the new model.
"""
import json
from pathlib import Path
from ordered_pv_release_acceptance import KIND as PV_KIND, COMPLETE as PV_COMPLETE, AUDITED as PV_AUDITED, require_pv393, require_pv_sources

MODE = 'model_native393_common_HVX_runtime'
KINDS = {
    PV_KIND: (PV_COMPLETE, PV_AUDITED),
    'dynamic-qk-invariant-vector-q2to32': ('query2to32_dynamic_QK_invariant_vector_native_timing_and_original_states_accepted',
        'query2to32_dynamic_QK_invariant_vector_native393_independently_revisited'),
    'empty-tensor-cse-q2to32': ('query2to32_empty_tensor_CSE_native_timing_and_original_states_accepted',
        'query2to32_empty_tensor_CSE_native393_independently_revisited'),
    'gqa-key-tile-q1': ('query1_GQA_key_tile_native_timing_and_original_states_accepted',
        'query1_GQA_key_tile_native393_independently_revisited'),
    'gqa-key-scalar-forward-q1': ('query1_GQA_key_scalar_forward_native_timing_and_original_states_accepted',
        'query1_GQA_key_scalar_forward_native393_independently_revisited'),
}


def identity(row):
    return {k: row[k] for k in ('path', 'sha256')}


def validate_binding(index, record, closure, proof, layout):
    current = index['current_default_accepted_version']
    numerical = index['accepted_versions'][current]
    kind = closure['candidate_kind']
    if kind not in KINDS:
        raise ValueError('Distinct accepted K model kind required')
    closure_status, audit_status = KINDS[kind]
    pv = kind == PV_KIND
    if pv:
        require_pv393(closure, proof)
    if (record['acceptance_mode'] != MODE or record['numerical_version'] != current
            or record['numerical_baseline_acceptance'] != numerical['acceptance']
            or record['acceptance'] != numerical['acceptance']
            or record['independent_audit'] != numerical['numerical_audit']
            or record['loader_policy_audit'] != numerical['loader_policy_audit']
            or record['runtime_catalog'] != index['runtime_catalog']
            or closure['status'] != closure_status
            or any(type(closure[k]) is not int or closure[k] != n for k, n in
                (('DSP_calls', 393), ('candidate_DSP_calls', 361), ('paired_baseline_DSP_calls', 32)))
            or closure['model_inference_acceptance'] is not (False if pv else True)
            or closure['source_revision'] != numerical['source_revision']
            or closure['candidate_libraries'] != index['candidate_libraries']
            or proof['status'] != audit_status
            or proof['closure'] != identity(record['acceptance'])
            or proof['controller'] != identity(record['controller_process'])
            or proof['source_revision' if pv else 'numerical_source_revision'] != numerical['source_revision']
            or proof['candidate_libraries'] != closure['candidate_libraries']
            or proof['model_inference_acceptance'] is not True
            or any(type(proof[k]) is not int or proof[k] != n for k, n in
                (('DSP_calls', 393), ('new_DSP_calls', 0), ('candidate_DSP_calls', 361), ('paired_baseline_DSP_calls', 32)))
            or len(proof['raw_invocation_traces']) != 12
            or layout['status'] != 'original393_common_payload_preserving_loader_policy_independently_revisited'
            or layout['closure'] != identity(record['acceptance'])
            or layout['DSP_calls'] != 393 or layout['new_DSP_calls'] != 0
            or layout['all_original_and_candidate_constant_bytes_reverified'] is not True
            or layout['loader_failure_fixed'] is not False or len(layout['sessions']) != 12):
        raise ValueError('Complete distinct model393, common-HVX identity and actual byte/layout audits required')
    runtime = closure['common_native_runtime_override']
    if (identity(runtime) != identity(record['runtime_catalog'])
            or runtime['board_path'] != record['board_runtime_catalog']):
        raise ValueError('Default runtime must be the actually measured common runtime')
    baseline = record['paired_baseline_numerical_version']
    legacy = record['historical_transport_acceptance']
    transport_base = legacy['numerical_version']
    rollback = index['runtime_rollback']
    if (baseline == current or baseline not in index['accepted_versions']
            or baseline != index['rollback']['previous_accepted_version']
            or transport_base == current or transport_base not in index['accepted_versions']
            or legacy['numerical_baseline_acceptance'] != index['accepted_versions'][transport_base]['acceptance']
            or legacy['runtime_catalog'] != record['runtime_catalog']
            or rollback['numerical_version'] != transport_base
            or rollback['numerical_baseline_acceptance'] != index['accepted_versions'][transport_base]['acceptance']
            or record['new_transport_speedup_claimed'] is not False
            or record['changed_model_scalar384_claimed'] is not False):
        raise ValueError('Historical transport comparison and tested old-model scalar rollback must remain distinct')
    if baseline != transport_base:
        paired = record.get('paired_baseline_model_HVX_acceptance', {})
        if (kind not in ('gqa-key-scalar-forward-q1','empty-tensor-cse-q2to32','dynamic-qk-invariant-vector-q2to32',PV_KIND) or paired.get('acceptance_mode') != MODE
                or paired.get('numerical_version') != baseline
                or paired.get('acceptance') != index['accepted_versions'][baseline]['acceptance']
                or paired.get('historical_transport_acceptance') != legacy
                or paired.get('runtime_catalog') != record['runtime_catalog']):
            raise ValueError('Exact accepted paired N32 model runtime lineage required')
    elif 'paired_baseline_model_HVX_acceptance' in record:
        raise ValueError('Unchanged historical transport baseline must not borrow another model identity')
    return current, baseline


def revisit(index, record, references, indexed_files, accepted_transport_parts, git, _seen=()):
    read = lambda p: json.loads(Path(p).read_text())
    closure = read(record['acceptance']['path'])
    proof = read(record['independent_audit']['path'])
    layout = read(record['loader_policy_audit']['path'])
    current, baseline = validate_binding(index, record, closure, proof, layout)
    # PV -> QK -> CSE -> N32 is the longest supported changed-model chain.
    if current in _seen or len(_seen) > 3:
        raise ValueError('Repeated or unsupported model runtime lineage')
    numerical = index['accepted_versions'][current]
    if (git(numerical['source_root'], 'rev-parse', 'HEAD') != numerical['source_revision']
            or git(numerical['source_root'], 'status', '--porcelain')):
        raise ValueError('Accepted numerical source must remain frozen')
    legacy = record['historical_transport_acceptance']; transport_base = legacy['numerical_version']
    paired_audit = None
    if baseline == transport_base:
        historical = dict(index, current_default_accepted_version=transport_base)
        prior = accepted_transport_parts(historical, legacy, references)
    else:
        paired = record['paired_baseline_model_HVX_acceptance']
        paired_closure = read(paired['acceptance']['path'])
        expected = ('dynamic-qk-invariant-vector-q2to32' if closure['candidate_kind']==PV_KIND else
                    'empty-tensor-cse-q2to32' if closure['candidate_kind']=='dynamic-qk-invariant-vector-q2to32'
                    else 'gqa-key-tile-q1')
        if paired_closure['candidate_kind'] != expected:
            raise ValueError('The directly paired model must retain its independently accepted original kind')
        historical = dict(index, current_default_accepted_version=baseline,
            candidate_libraries=paired_closure['candidate_libraries'],
            rollback=dict(index['rollback'], previous_accepted_version=paired['paired_baseline_numerical_version']))
        paired_audit = revisit(historical, paired, references, indexed_files, accepted_transport_parts, git, _seen+(current,))
        prior = paired_audit['historical_transport_audit']
    controller = read(record['controller_process']['path'])
    if (controller['status'] != 'completed' or controller['returncode'] != 0
            or controller['source_revision'] != numerical['source_revision']
            or any(Path('/proc', str(controller[k])).exists() for k in ('controller_pid', 'child_pid'))):
        raise ValueError('Actual completed new-model393 controller required')
    if closure['candidate_kind'] == PV_KIND:
        source = Path(numerical['source_root'])
        require_pv_sources(closure, proof, source, index['accepted_versions'][baseline]['acceptance'])
        if (controller['source_root'] != str(source)
                or controller['authorized_deadline_unix']-controller['start_unix'] < 14400
                or controller['completed_unix'] >= controller['authorized_deadline_unix']
                or closure['deadline_unix'] != controller['authorized_deadline_unix']):
            raise ValueError('Exact frozen PV reviewer, original four-hour controller and paired QK393 required')
    root = Path(record['acceptance']['path']).parent
    expected = {root/'cold', root/'warm16'}
    expected.update(root/('native-abba-'+str(i)+'-'+role)
        for i, role in enumerate(('baseline', 'candidate', 'candidate', 'baseline')))
    expected.update(root/'state304'/case for case in
        ('tails', 'eos', 'old-long32', 'prefill1024', 'decode993', 'code64'))
    if ({Path(row['folder']) for row in layout['sessions']} != expected
            or len(layout['sessions']) != len(expected)):
        raise ValueError('Every original393 session must appear exactly once')
    sessions = []
    for row in layout['sessions']:
        folder = Path(row['folder'])
        if folder.parent != root and folder.parent != root/'state304':
            raise ValueError('Only the actually closed model393 sessions may bind the runtime')
        actual = read(folder/'native-result.json')
        commands = read(folder/'commands.json') if (folder/'commands.json').exists() else read(folder/'result.json')
        if folder == root/'cold':
            if actual['DSP_calls'] != 9:
                raise ValueError('Nine original cold numerical calls required')
            continue
        command = commands['board']
        if (command[command.index('--catalog')+1] != record['board_runtime_catalog']
                or command[command.index('--catalog-sha256')+1] != record['runtime_catalog']['sha256']):
            raise ValueError('Every actual post-cold model arm must use the same HVX binary')
        old = folder.name in ('native-abba-0-baseline', 'native-abba-3-baseline')
        libraries = read(index['accepted_versions'][baseline]['acceptance']['path'])['candidate_libraries'] if old else closure['candidate_libraries']
        if any(f['DSO_sha256'] != libraries['(1, 1)' if f['query'] == 1 else '(2, 32)'] for f in actual['forwards']):
            raise ValueError('Actual paired model libraries differ from their accepted identities')
        sessions.append(dict(folder=str(folder), DSP_calls=actual['DSP_calls'], baseline=old))
    if (len(sessions) != 11 or sum(s['DSP_calls'] for s in sessions) != 384
            or sum(s['DSP_calls'] for s in sessions if s['baseline']) != 32):
        raise ValueError('Exactly eleven common-HVX sessions and 352 changed / 32 old model calls required')
    source = record['source_root']
    if (source != record['historical_transport_acceptance']['source_root']
            or record['source_revision'] != prior['source_revision']
            or record['source_backup'] != record['historical_transport_acceptance']['source_backup']):
        raise ValueError('Actual unchanged accepted HVX source and recovery required')
    for artifact in (proof, layout):
        indexed_files(artifact, references)
    return dict(acceptance_mode=MODE, source_revision=record['source_revision'], numerical_version=current,
        DSP_calls=384, changed_model_DSP_calls=352, paired_old_model_DSP_calls=32,
        sessions=sessions, independent_audit=record['independent_audit'],
        historical_transport_audit=prior, new_transport_speedup_claimed=False,
        paired_baseline_model_runtime_audit=paired_audit, changed_model_scalar384_claimed=False)
