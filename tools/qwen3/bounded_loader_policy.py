"""Explicit common loader policy; numerical/performance gates remain separate."""
import json
from pathlib import Path

from build import sha

KIND = 'bounded_readonly_LOAD_payload_preserving_v1'
MODE = 'verified-payload-preserving'
MAX_LOAD_BYTES = 256 * 1024**2


def configured_policy(config):
    policy = config.get('bounded_constant_loader')
    if policy is None:
        return None
    required = {'kind', 'proofs', 'baseline_native_source', 'payload_replay', 'native_replay'}
    if not isinstance(policy, dict) or set(policy) != required or policy['kind'] != KIND:
        raise ValueError('Explicit bounded constant loader policy required')
    for key in ('proofs', 'baseline_native_source'):
        if type(policy[key]) is not str or not Path(policy[key]).is_absolute():
            raise ValueError('Absolute board loader policy paths required')
    evidence = (
        ('payload_replay', 'bounded_loader_original_q8_q1_full_output_and_three_FNV_replay_passed', 2),
        ('native_replay', 'independently_reparsed_native16_bounded_loader_complete_numerics_passed', 16),
    )
    for key, status, count in evidence:
        pin = policy[key]
        if (not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}
                or not Path(pin['path']).is_absolute() or sha(pin['path']) != pin['sha256']):
            raise ValueError('Pinned completed loader numerical evidence changed')
        record = json.loads(Path(pin['path']).read_text())
        if (record['status'] != status or record['DSP_calls'] != count
                or record.get('performance_acceptance') is not False
                or record.get('loader_failure_fixed') is not False):
            raise ValueError('Loader diagnostics cannot invent acceptance or a stable failure fix')
    return dict(policy)


def apply_arguments(arguments, policy):
    """Inherit exactly one policy, rejecting conflicting accepted-run options."""
    result = list(arguments)
    for flag, value in (('--constant-load-split-proofs', policy['proofs']),
                        ('--constant-loader-mode', MODE)):
        if flag in result:
            index = result.index(flag)
            if result.count(flag) != 1 or index + 1 == len(result) or result[index + 1] != value:
                raise ValueError('Both native timing arms must use the same bounded loader policy')
        else:
            result += [flag, value]
    return result


def verify_native_layout(native):
    layout = native.get('constant_loader_layout', {})
    libraries = layout.get('libraries', [])
    expected = {row['sha256'] for row in json.loads(Path(__file__).with_name('model-constants.lock.json').read_text())['libraries']}
    if (layout.get('kind') != KIND or layout.get('original_weights_code_and_symbol_values_preserved') is not True
            or layout.get('loader_failure_fixed') is not False or len(libraries) != 3
            or any(row['max_load_bytes'] != MAX_LOAD_BYTES for row in libraries)
            or {row['original_sha256'] for row in libraries} != expected):
        raise ValueError('Actual three bounded constant payload proofs required')
    for row in libraries:
        checks = row['checks']
        required = ('all_other_file_bytes_identical', 'only_dynsym_section_location_and_DT_SYMTAB_pointer_changed',
                    'all_dynamic_symbol_values_and_table_bytes_identical', 'first_page_program_header_table',
                    'original_symbol_code_constant_addresses_and_bytes_unchanged')
        if (any(checks.get(key) is not True for key in required)
                or not 0 < checks['readonly_LOAD_filesz_max'] <= MAX_LOAD_BYTES
                or not 0 < checks['candidate_program_headers'] <= 10):
            raise ValueError('Actual firmware-compatible payload preservation checks required')
    return sorted((row['original_sha256'], row['candidate_sha256'], row['proof_sha256']) for row in libraries)
