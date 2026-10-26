"""Bind an optional host packer to its own393 and the identical current model.

This never changes the numerical default or production launcher. The complete
PI audit is a prerequisite; an ARM serializer benchmark or negative89 is not a
model acceptance. A later numerical default must establish its own binding.
"""
import json
import math
from pathlib import Path

MODE = 'host_input_packing_original393_identical_model'


def identity(value):
    return {k: value[k] for k in ('path', 'sha256')}


def validate(index, record, closure, audit):
    current = index['current_default_accepted_version']
    numerical = index['accepted_versions'][current]
    if (record['acceptance_mode'] != MODE or record['numerical_version'] != current
            or record['input_packer'] != 'joined'
            or record['numerical_baseline_acceptance'] != numerical['acceptance']
            or record['runtime_catalog'] != index['runtime_catalog']
            or record['production_launcher_switched'] is not False
            or record['TUI_speedup_claimed'] is not False):
        raise ValueError('Optional joined packer must bind the exact current numerical and runtime acceptance')
    if (closure['candidate_kind'] != 'host-input-packing-joined'
            or closure['status'] != 'host_input_packing_native393_complete_NOT_independently_accepted'
            or closure['model_inference_acceptance'] is not False
            or type(closure['DSP_calls']) is not int or closure['DSP_calls'] != 393
            or closure['candidate_DSP_calls'] != 361 or closure['paired_baseline_DSP_calls'] != 32
            or closure['original_state304_executed'] is not True
            or closure['original_numerical_dependencies_unchanged'] is not True
            or closure['original_full393_minimum_remaining_seconds'] != 14400
            or closure['new_primary_model_codegen_invocations'] != 0
            or closure['source_revision'] != record['source_revision']
            or closure['baseline_closure'] != identity(numerical['acceptance'])
            or closure['baseline_independent_revisit'] != identity(numerical['numerical_audit'])
            or closure['candidate_libraries'] != index['candidate_libraries']
            or closure['common_native_runtime_override']['sha256'] != index['runtime_catalog']['sha256']):
        raise ValueError('Own full original393 with identical current model DSOs and unchanged policy required')
    if (audit['status'] != 'host_input_packing_native393_independently_revisited'
            or audit['candidate_kind'] != 'host-input-packing-joined'
            or audit['closure'] != identity(record['acceptance'])
            or audit['controller'] != identity(record['controller_process'])
            or audit['source_root'] != record['source_root']
            or audit['source_revision'] != record['source_revision']
            or type(audit['DSP_calls']) is not int or audit['DSP_calls'] != 393
            or audit['new_DSP_calls'] != 0 or audit['model_inference_acceptance'] is not True
            or audit['candidate_DSP_calls'] != 361 or audit['paired_baseline_DSP_calls'] != 32
            or audit['same_original_model_graphs'] is not True or audit['new_PRIMARY'] != 0
            or audit['positive_packing_gain'] is not True
            or audit['candidate_libraries'] != index['candidate_libraries']
            or audit['numerical_source_comparison']['all_other_numerical_runtime_files_identical'] is not True
            or audit['native_packing_timing'] != closure['native_packing_timing']
            or audit['native']['raw_native_measurements'] != closure['native_model_timing']
            or audit['states'] is None or audit['states']['DSP_calls'] != 304
            or len(audit['raw_invocation_traces']) != 12
            or len(audit['actual_AP_owners_gone']) != 12
            or len(audit['compiler_cache']) != 2
            or any(row['new_PRIMARY'] != 0 for row in audit['compiler_cache'])):
        raise ValueError('Own independent393, raw timing, complete state and source comparison required')
    timing = closure['native_packing_timing']
    if set(timing) != {'1', '13', '14'}:
        raise ValueError('Original q1/q13/q14 paired native timings required')
    if {q: {k: v for k, v in group.items() if k != 'packing_seconds'}
            for q, group in timing.items()} != closure['native_model_timing']:
        raise ValueError('Packing report must retain the same actual native kernel and callback measurements')
    for group in timing.values():
        if set(group) != {'kernel_seconds', 'callback_seconds', 'packing_seconds'}:
            raise ValueError('Complete pack, callback and kernel measurements required')
        for row in group.values():
            if (set(row) != {'baseline', 'candidate', 'ratio'}
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in row.values())
                    or row['ratio'] != row['candidate'] / row['baseline']):
                raise ValueError('Finite raw derived timing ratios required')
    if (timing['1']['callback_seconds']['ratio'] >= .99
            or any(group['packing_seconds']['ratio'] >= .7 for group in timing.values())
            or any(group[key]['ratio'] > 1.02 for group in timing.values()
                for key in ('kernel_seconds', 'callback_seconds'))):
        raise ValueError('Actual callback improvement and unchanged-model nonregression required')


def revisit(index, record, references, indexed_files, git):
    read = lambda p: json.loads(Path(p).read_text())
    indexed_files(record, references)
    closure = read(record['acceptance']['path']); audit = read(record['independent_audit']['path'])
    validate(index, record, closure, audit)
    root = Path(record['source_root']); backup = read(record['source_backup']['path'])
    if (git(root, 'rev-parse', 'HEAD') != record['source_revision'] or git(root, 'status', '--porcelain')
            or backup['status'] != 'source_full_bundle_actual_restore_HEAD_tree_clean_fsck_passed'
            or backup['source_root'] != str(root) or backup['source_revision'] != record['source_revision']
            or backup['tree'] != git(root, 'rev-parse', 'HEAD^{tree}')
            or backup['temporary_restore_removed'] is not True):
        raise ValueError('Frozen original packing source and actual source recovery required')
    from build import sha
    if (audit['reviewer_source']['sha256'] != sha(root / 'tools/qwen3/revisit_input_packing_closure.py')
            or closure['source_sha256'] != sha(root / 'tools/qwen3/run_input_packing_iteration.py')):
        raise ValueError('Actually used driver and independent reviewer bytes changed')
    controller = read(record['controller_process']['path'])
    if (controller['status'] != 'completed' or controller['returncode'] != 0
            or controller['source_root'] != str(root) or controller['source_revision'] != record['source_revision']
            or controller['completed_unix'] >= controller['authorized_deadline_unix']
            or controller['authorized_deadline_unix'] - controller['start_unix'] < 14400):
        raise ValueError('Actual normal original-budget controller required')
    indexed_files(backup, references); indexed_files(closure, references); indexed_files(audit, references)
    return dict(acceptance_mode=MODE, numerical_version=record['numerical_version'],
        source_revision=record['source_revision'], DSP_calls=393,
        independent_audit=record['independent_audit'], native_packing_timing=closure['native_packing_timing'],
        numerical_default_unchanged=True, production_launcher_switched=False, new_DSP_calls=0)
