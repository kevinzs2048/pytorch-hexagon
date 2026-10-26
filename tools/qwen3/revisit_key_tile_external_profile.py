"""Revisit raw external-helper counters for the accepted N32 model.

Counter times attribute the instrumented model. They do not measure a new
production speedup. The unchanged model object and delegated callees remain
bound to their actual build and staged board receipts.
"""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time

from build import ROOT, entry, sha
from probe_model_memory_arm import parse_profile


def fnv(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


def revisit_packets(board, build, forward, query=1):
    run = json.loads((board/'result.json').read_text())
    events = [json.loads(line) for line in (board/'stream.stdout.txt').read_text().splitlines()]
    if (run['status'] != 'diagnostic_model_memory_complete_packet_bitexact'
            or run['process_exit_code'] != 0 or run['query'] != query
            or run['model_calls'] != 4 or run['diagnostic_query_calls'] != 2 or run['DSP_calls'] != 6
            or len(run['rows']) != 4 or len(run['profiles']) != 2
            or run['stream_events'] != events or events[-1]['stage'] != 'close'
            or events[-1]['rpc_result'] != 0
            or run['input_sha256'] != forward['input_sha256']
            or run['expected_output_sha256'] != forward['output_sha256']
            or run['baseline_library_sha256'] != forward['DSO_sha256']
            or run['profile_library_sha256'] != build['library']['sha256']
            or run['model_object_unchanged'] is not True
            or run['model_graph_changed'] is not False or run['power_votes_changed'] is not False):
        raise ValueError('Exact original model ABBA4, two raw queries and normal close required')
    dynamic = query != 1
    symbol = '_mlir_ciface___qwen_i8_multi_matmul' if dynamic else '_mlir_ciface___qwen_i8_matmul'
    if (type(query) is not int or not 1 <= query <= 32
            or build['external_profile_symbols']['10'] != symbol
            or (dynamic and (build.get('query_bounds') != [2, 32]
                or build.get('input_bytes_are_maximum_capacity') is not True))
            or (not dynamic and 'query_bounds' in build)):
        raise ValueError('Original dynamic query category and exact corresponding INT8 callee required')
    invokes = [e for e in events if e['stage'] == 'invoke']
    if (len(invokes) != 6 or [e['ordinal'] for e in invokes] != list(range(1, 7))
            or any(e[k] != 0 for e in invokes for k in ('rpc_result', 'DSP_result', 'output_result'))):
        raise ValueError('Six unique consecutive normally completed RPC invocations required')
    used = []
    for i, row in enumerate(run['rows']):
        e = row['actual_invoke']
        if (row['index'] != i or row['library'] != ('baseline', 'profile', 'profile', 'baseline')[i]
                or row['all_logits_and_56_KV_byteexact'] is not True
                or row['output_sha256'] != forward['output_sha256'] or e not in invokes
                or e['library_slot'] != (0, 1, 1, 0)[i]
                or e['input_bytes'] != forward['invoke']['input_bytes'] or e['output_bytes'] != 235488768
                or (dynamic and not 16 < e['input_bytes'] <= 235012880)
                or (not dynamic and e['input_bytes'] != 234885520)
                or any(e[k] != forward['invoke']['AP_input_fnv1a32'] for k in
                    ('AP_input_fnv1a32', 'DSP_input_fnv1a32', 'RPC_input_fnv1a32'))):
            raise ValueError('Every complete original model packet, FNV and actual library slot must match')
        used.append(e)
    profiles, raws = [], []
    query_fnv = fnv(b'QWMEM01\0' + bytes(8))
    if build['external_profile_kinds'] != list(range(1, 17)) or build['required_profile_kinds'] != [1, 2, 3, 10]:
        raise ValueError('All original external helper and allocation categories required')
    for i, row in enumerate(run['profiles']):
        path = board/('profile-' + str(i+1) + '.bin')
        # Profile calls follow model indices one and two, respectively.
        actual = parse_profile(path.read_bytes(), build['external_profile_kinds'], build['required_profile_kinds'])
        e = row['actual_invoke']
        if (row['index'] != i+1 or actual != row['profile'] or actual['invocations'] != i+1
                or actual['totals']['10']['calls'] != 197 or e not in invokes
                or e['library_slot'] != 1 or e['input_bytes'] != 16 or e['output_bytes'] != 128+4096*48
                or any(e[k] != query_fnv for k in
                    ('AP_input_fnv1a32', 'DSP_input_fnv1a32', 'RPC_input_fnv1a32'))):
            raise ValueError('Raw external counters must bind all 197 original INT8 calls and their actual queries')
        profiles.append(actual); raws.append(entry(path)); used.append(e)
    if sorted(e['ordinal'] for e in used) != list(range(1, 7)):
        raise ValueError('No repeated or omitted model/query evidence allowed')
    groups = {}
    for kind, symbol in build['external_profile_symbols'].items():
        groups[symbol] = {key: statistics.mean(p['totals'][kind][key] for p in profiles)
            for key in ('calls', 'bytes', 'microseconds', 'cycles')}
    timing = {}
    for field in ('kernel_us', 'seconds'):
        baseline = statistics.mean(r['actual_invoke'][field] for r in run['rows'] if r['library'] == 'baseline')
        diagnostic = statistics.mean(r['actual_invoke'][field] for r in run['rows'] if r['library'] == 'profile')
        timing[field] = dict(baseline=baseline, diagnostic=diagnostic, diagnostic_overhead_ratio=diagnostic/baseline)
    return dict(groups=groups, raw_profiles=raws, raw_stdout=entry(board/'stream.stdout.txt'),
        result=entry(board/'result.json'), paired_instrumentation_timing=timing,
        complete_output_hash=forward['output_sha256'], complete_diagnostic_output_file_retained=False,
        model_DSP_calls=4, diagnostic_query_DSP_calls=2, new_DSP_calls=0,
        instrumented_times_are_not_production_gain=True, model_inference_acceptance=False)


def require_accepted_model(accepted,audit,closure):
    from revisit_key_tile_native_closure import require_complete_closure
    kinds={'gqa-key-tile-q1':'query1_GQA_key_tile_native393_independently_revisited',
        'empty-tensor-cse-q2to32':'query2to32_empty_tensor_CSE_native393_independently_revisited'}
    kind=accepted.get('candidate_kind')
    if kind not in kinds:
        raise ValueError('Explicit accepted N32 or original dynamic CSE model required')
    require_complete_closure(accepted,kind)
    if (audit.get('status')!=kinds[kind] or audit.get('closure')!=closure
            or audit.get('DSP_calls')!=393 or audit.get('new_DSP_calls')!=0
            or audit.get('model_inference_acceptance') is not True):
        raise ValueError('Actual independently accepted matching model393 required')
    return kind


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'diagnostic', 'accepted-closure', 'accepted-audit', 'source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--query', type=int, default=1)
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    git = lambda root, *args: subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    revision = git(ROOT, 'rev-parse', 'HEAD')
    if not a.output.is_absolute() or a.output.exists() or git(ROOT, 'status', '--porcelain'):
        raise ValueError('Fresh absolute report and frozen independent source required')
    accepted = read(a.accepted_closure/'result.json'); audit = read(a.accepted_audit)
    native_path = a.accepted_closure/'cold/native-result.json'; native = read(native_path)
    build = read(a.diagnostic/'model/result.json'); plan = read(a.diagnostic/'board-plan-v1.json')
    forward = next(r for r in native['forwards'] if r['query'] == a.query)
    model_kind=require_accepted_model(accepted,audit,entry(a.accepted_closure/'result.json'))
    if (accepted['candidate_libraries']['(1, 1)' if a.query == 1 else '(2, 32)'] != forward['DSO_sha256']
            or build['status'] != 'diagnostic_model_memory_library_built_NOT_device_acceptance'
            or any(build[k] is not True for k in ('model_object_unchanged', 'profile_state_and_counters_no_preemptible_relocations',
                'allocator_counters_no_preemptible_relocations', 'all_wrapped_model_callees_resolve'))
            or build['model_graph_changed'] is not False or build['power_votes_changed'] is not False
            or build['inspect_fmaf'] is not False or build['DSP_calls'] != 0):
        raise ValueError('Actual accepted N32393 and unchanged diagnostic model required')
    controls = []
    for name in ('model-link-process-v1.json', 'board-process-v1.json'):
        owner = read(a.diagnostic/name)
        if (owner['status'] != 'completed' or owner['returncode'] != 0
                or owner['source_root'] != str(a.source) or owner['source_revision'] != git(a.source, 'rev-parse', 'HEAD')
                or git(a.source, 'status', '--porcelain') or owner['completed_unix'] >= owner['authorized_deadline_unix']
                or any(Path('/proc', str(owner[k])).exists() for k in ('controller_pid', 'child_pid'))):
            raise ValueError('Actual normal link and board exits with unchanged source required')
        controls.append(entry(a.diagnostic/name))
    for row in plan['files']:
        if sha(row['source']) != row['sha256']:
            raise ValueError('Actual staged diagnostic bytes changed')
    if (next(r for r in plan['files'] if r['name'] == 'probe.py')['sha256'] != sha(a.source/'tools/qwen3/probe_model_memory_arm.py')
            or build['source_sha256'] != sha(a.source/'tools/qwen3/profile_model_memory.py')
            or build['source_revision'] != git(a.source, 'rev-parse', 'HEAD')):
        raise ValueError('Actual unchanged compiler and full-output checker must be staged')
    for key in ('model_object', 'original_link_receipt', 'original_wrapper', 'diagnostic_wrapper', 'profiler_source',
                'runtime_header', 'library', 'original_memory_profile_source', 'original_model_effect_guard'):
        if sha(build[key]['path']) != build[key]['sha256']:
            raise ValueError('Pinned model or diagnostic build input changed: '+key)
    for row in build['original_link_inputs'] + build['constants']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Original compiler, runtime or constant bytes changed')
    link = read(build['original_link_receipt']['path'])
    if sha(link['library']['path']) != forward['DSO_sha256'] or link['library']['sha256'] != forward['DSO_sha256']:
        raise ValueError('Diagnostic must wrap the exact currently accepted N32 model')
    delegate = build['preserved_existing_memcpy']
    if (delegate['original_text_instructions_identical'] is not True or delegate['original_object_unchanged'] is not True):
        raise ValueError('Existing memcpy instructions must remain unchanged')
    for key in ('original_object', 'diagnostic_delegate_object'):
        if sha(delegate[key]['path']) != delegate[key]['sha256']:
            raise ValueError('Original or delegated memcpy object changed')
    # Re-extract both instruction sections independently, without trusting metadata.
    objcopy = delegate['rename_command'][0]
    if sha(objcopy) != next(r['sha256'] for r in delegate['tools'] if r['path'] == objcopy):
        raise ValueError('Original object inspection tool changed')
    sections = []
    for key in ('original_object', 'diagnostic_delegate_object'):
        sections.append(subprocess.check_output([objcopy, '--dump-section', '.text=/dev/stdout', delegate[key]['path'], '/dev/null']))
    import hashlib
    if (sections[0] != sections[1] or len(sections[0]) != delegate['original_text_bytes']
            or hashlib.sha256(sections[0]).hexdigest() != delegate['original_text_sha256']):
        raise ValueError('Delegated memcpy instructions differ from the original model')
    command = plan['command']; runtime = accepted['common_native_runtime_override']
    if (command[command.index('--catalog')+1] != runtime['board_path']
            or command[command.index('--catalog-sha256')+1] != runtime['sha256']
            or command[command.index('--constant-loader-mode')+1] != 'verified-payload-preserving'):
        raise ValueError('Actual accepted HVX and bounded original constants required')
    board = a.diagnostic/'run-v1/board'; run = read(board/'result.json')
    if (run['accepted_run_sha256'] != sha(native_path) or run['receipt_sha256'] != sha(a.diagnostic/'model/result.json')
            or run['catalog_sha256'] != runtime['sha256']
            or run['constant_loader_layout']['kind'] != 'bounded_readonly_LOAD_payload_preserving_v1'
            or len(run['constant_loader_layout']['libraries']) != 3
            or run['constant_loader_layout']['libraries'] != native['constant_loader_layout']['libraries']
            or run['constant_loader_layout']['original_weights_code_and_symbol_values_preserved'] is not True
            or run['constant_loader_layout']['loader_failure_fixed'] is not False):
        raise ValueError('Original accepted packet, current library, runtime and all three bounded constants required')
    staged_audit = next(r for r in plan['files'] if r['name'] == 'accepted-audit.json')
    numerical = read(staged_audit['source'])
    if (numerical['status'] != 'framework_native_numerical_acceptance_passed' or numerical['numerical_acceptance'] is not True
            or numerical['run_sha256'] != sha(native_path) or run['accepted_audit_sha256'] != staged_audit['sha256']):
        raise ValueError('Original ARM numerical audit must bind the exact current cold packet')
    proof = revisit_packets(board, build, forward, a.query)
    if a.query != 1:
        kept = run['preserved_complete_profile_output']; full = board/Path(kept['path']).name
        if (Path(kept['path']).name != 'complete-profile-output.bin' or kept['bytes'] != 235488768
                or full.stat().st_size != 235488768 or sha(full) != kept['sha256']
                or kept['sha256'] != forward['output_sha256']):
            raise ValueError('Original dynamic graph must preserve every diagnostic logits/56KV byte')
        proof.update(complete_diagnostic_output_file_retained=True, complete_output=entry(full))
    pid = run['ready']['ARM_pid']; config = read(a.config)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', config['board']['ssh_target'],
        shlex.join([config['board']['python'], '-c',
            'from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()', str(pid)])], check=True)
    if git(ROOT, 'rev-parse', 'HEAD') != revision or git(ROOT, 'status', '--porcelain'):
        raise ValueError('Independent audit source changed')
    proof.update(status=('accepted_dynamic_CSE_external_helpers_complete_output_raw_and_owners_independently_revisited_DIAGNOSTIC' if model_kind=='empty-tensor-cse-q2to32' else
            'current_accepted_N32_external_helpers_raw_and_owners_independently_revisited_DIAGNOSTIC'
            if a.query == 1 else 'original_multi_token_external_helpers_complete_output_raw_and_owners_independently_revisited_DIAGNOSTIC'),
        query=a.query,accepted_model_kind=model_kind,
        accepted_closure=entry(a.accepted_closure/'result.json'), accepted_audit=entry(a.accepted_audit),
        controls=controls, build=entry(a.diagnostic/'model/result.json'), board_plan=entry(a.diagnostic/'board-plan-v1.json'),
        no_new_primary_model_codegen=True, actual_AP_owner_gone=pid,
        audit_source_revision=revision, audit_source_sha256=sha(__file__), completed_unix=time.time())
    a.output.write_text(json.dumps(proof, indent=2)+'\n'); print(json.dumps(proof['groups'], indent=2))


if __name__ == '__main__':
    main()
