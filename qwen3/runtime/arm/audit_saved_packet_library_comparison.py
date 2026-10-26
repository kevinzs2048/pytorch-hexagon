"""Rehash same-packet DSP library comparisons; report differences, never accept them.

The matching completed ARM telemetry supplies its actual q1 ABI, not a guessed
layout from a different graph. No CPU reference or new model execution occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from audit_qwen_native_request_sequence import audit_unchanged_kv
from hexagon_compiled_artifact_cache import canonical, digest_file
from hexagon_library_lifecycle import audit_library_events
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import MAGIC, layout_for


def tensor_difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype or not left.size:
        raise ValueError('Equal nonempty tensor layouts required')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonfinite saved model output')
    diff = np.abs(left.astype(np.float64)-right.astype(np.float64))
    return dict(bitexact=left.tobytes()==right.tobytes(),
        numerically_different_elements=int(np.count_nonzero(diff)),
        mean_abs_error=float(diff.mean()), max_abs_error=float(diff.max()))


def compare_outputs(actual, reference, layout, positions):
    if len(actual) != layout['output_bytes'] or len(reference) != len(actual):
        raise ValueError('Complete output packets required')
    rows = []
    for index, row in enumerate(layout['outputs']):
        if row['dtype'] != 'float32':
            raise ValueError('Expected Qwen float output ABI')
        values = [np.frombuffer(raw, dtype='<f4', count=row['bytes']//4,
                                offset=row['offset']).reshape(row['shape'])
                  for raw in (actual, reference)]
        item = dict(output_index=index, **tensor_difference(*values))
        if index < 56:
            item['updated_KV'] = tensor_difference(*(v[:, :, positions, :] for v in values))
        else:
            item['top_tokens'] = [int(v.argmax()) for v in values]
        rows.append(item)
    if len(rows) != 57:
        raise ValueError('Complete 56 KV tensors and logits required')
    return rows


def read_cpu_reference(path, digest, architecture, forward):
    if digest_file(path) != digest:
        raise ValueError('Pinned CPU reference record changed')
    record = json.loads(path.read_text())
    if (record['status'] != f'dynamic_exact_packet_{architecture}_eager_logits_KV_recorded'
            or record.get('reference_precision', 'int8') != 'int8'
            or record['inputs_exact'] is not True or record['CPU_reference_execution'] is not True
            or record['DSP_calls'] != 0 or record['query'] != forward['query']
            or record['start_position'] != forward['cache_positions'][0]
            or record['request_sha256'] != forward['input_sha256']
            or record['updated_KV']['entries'] != 56 or record['updated_KV']['active_slots'] != 1):
        raise ValueError('Same-input native INT8 CPU reference required')
    arrays = []
    for field, name, shape in [('last_logits', 'last-logits.bin', (1, 1, 151936)),
                               ('updated_KV', 'updated-kv.bin', (56, 1, 8, 1, 128))]:
        # ARM and x86 evidence have different mount prefixes; resolve only the
        # fixed sibling filenames and verify contents, never follow record paths.
        payload = Path(path).parent/name
        if payload.is_symlink() or digest_file(payload) != record[field]['sha256']:
            raise ValueError('CPU tensor bytes changed')
        values = np.fromfile(payload, dtype='<f4')
        if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
            raise ValueError('CPU reference tensor layout/finiteness differs')
        arrays.append(values.reshape(shape))
    return record, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'baseline', 'telemetry'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('arm_reference', 'x86_reference'):
        parser.add_argument('--'+name.replace('_', '-'), type=Path)
        parser.add_argument('--'+name.replace('_', '-')+'-sha256')
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('Fresh independent audit output required')
    records = {}
    for name in ('replay', 'baseline', 'telemetry'):
        path = getattr(args, name)
        if digest_file(path) != getattr(args, name+'_sha256'):
            raise ValueError('Pinned record changed: '+name)
        records[name] = json.loads(path.read_text())
    run, baseline, telemetry = (records[name] for name in ('replay', 'baseline', 'telemetry'))
    if (run['status'] != 'same_packet_library_comparison_recorded_NOT_numerical_acceptance'
            or run['comparison_only'] is not True or run['process_exit_code'] != 0
            or run['native_generate_execution'] is not False
            or run['baseline_sha256'] != args.baseline_sha256 or len(run['forward_indices']) != 1
            or telemetry['status'] != 'ARM_native_dynamic_compiler_handoff_generate_completed'
            or telemetry['process_exit_code'] != 0 or telemetry['CPU_model_fallback'] is not False):
        raise ValueError('Completed same-packet comparison and actual ARM ABI required')
    index = run['forward_indices'][0]
    old, new = baseline['forwards'][index], telemetry['forwards'][index]
    if (old['query'] != 1 or new['query'] != 1
            or any(old[key] != new[key] for key in ('input_sha256', 'cache_positions', 'input_token_ids'))):
        raise ValueError('Only same-input singleton decode comparison admitted')
    capture = telemetry['captures'][new['AOT_capture_index']]
    if capture['DSO_sha256'] != new['DSO_sha256']:
        raise ValueError('Actual capture/forward library differs')
    layout = layout_for(capture['contract'], {'query': 1})
    cpu = {}
    for name, architecture in [('arm_reference', 'ARM'), ('x86_reference', 'x86')]:
        path, digest = getattr(args, name), getattr(args, name+'_sha256')
        if bool(path) != bool(digest):
            raise ValueError('CPU reference requires both path and digest')
        if path:
            cpu[architecture] = read_cpu_reference(path, digest, architecture, old)
    if len(cpu) == 2 and any(cpu['ARM'][0][key] != cpu['x86'][0][key]
                            for key in ('context_sha256', 'checkpoint_sha256')):
        raise ValueError('CPU model/context differ between architectures')
    folder = args.baseline.parent/f'forward-{index:02d}-q1'
    incoming, expected = [(folder/(side+'.bin')).read_bytes() for side in ('input', 'output')]
    if (len(incoming) != 16+layout['input_bytes'] or len(expected) != layout['output_bytes']
            or incoming[:8] != MAGIC or int.from_bytes(incoming[8:16], 'little', signed=True) != 1
            or any(hashlib.sha256(raw).hexdigest() != old[side+'_sha256']
                   for raw, side in ((incoming, 'input'), (expected, 'output')))):
        raise ValueError('Historical complete request/response changed')
    libraries = run['libraries']
    if not 2 <= len(libraries) <= 4 or run['DSP_calls'] != 2*len(libraries):
        raise ValueError('Both orders of bounded library comparison required')
    for library in libraries:
        if digest_file(library['path']) != library['sha256']:
            raise ValueError('Replayed actual library changed')
    order = list(range(len(libraries)))
    order += list(reversed(order))
    if [row['library_slot'] for row in run['replays']] != order:
        raise ValueError('Comparison did not cover both library orders')
    lifecycle = audit_library_events(run['events'], order)
    if [e for e in run['events'] if e['stage']=='invoke'] != [r['invoke'] for r in run['replays']]:
        raise ValueError('Actual invocation coverage differs')
    results, digests = [], {}
    for ordinal, row in enumerate(run['replays']):
        path = args.replay.parent/f'replay-{ordinal:02d}-from-{index:02d}'/'output.bin'
        raw = path.read_bytes()  # Do not mmap sparse evidence.
        digest = hashlib.sha256(raw).hexdigest()
        if (row['baseline_forward'] != index or row['input_sha256'] != old['input_sha256']
                or digest != row['output_sha256'] or len(raw) != layout['output_bytes']
                or row['invoke']['input_bytes'] != len(incoming)
                or row['invoke']['output_bytes'] != len(raw)
                or row['full_original_output_bitexact'] != (raw == expected)):
            raise ValueError('Independent comparison packet check failed')
        slot = row['library_slot']
        digests.setdefault(slot, set()).add(digest)
        item = dict(ordinal=ordinal, library_slot=slot, output_sha256=digest,
            full_original_output_bitexact=raw==expected,
            matches_new_ARM_reported_digest=digest==new['output_sha256'],
            KV_update_scope=audit_unchanged_kv(incoming, raw, layout, old['cache_positions']),
            outputs_vs_historical=compare_outputs(raw, expected, layout, old['cache_positions']))
        item['same_input_CPU'] = {}
        for architecture, (_, (cpu_logits, cpu_kv)) in cpu.items():
            logits_row = layout['outputs'][-1]
            logits = np.frombuffer(raw, dtype='<f4', count=logits_row['bytes']//4,
                                   offset=logits_row['offset']).reshape(logits_row['shape'])
            layers = []
            for n, row_layout in enumerate(layout['outputs'][:56]):
                values = np.frombuffer(raw, dtype='<f4', count=row_layout['bytes']//4,
                    offset=row_layout['offset']).reshape(row_layout['shape'])
                layers.append(tensor_difference(values[:, :, old['cache_positions'], :], cpu_kv[n]))
            item['same_input_CPU'][architecture] = dict(logits=tensor_difference(logits, cpu_logits),
                updated_KV=layers, top_tokens=[int(logits.argmax()), int(cpu_logits.argmax())])
        results.append(item)
    result = dict(status='same_input_libraries_independently_compared_NOT_numerical_acceptance',
        source_sha256=digest_file(__file__), replay_sha256=args.replay_sha256,
        baseline_sha256=args.baseline_sha256, telemetry_sha256=args.telemetry_sha256,
        libraries=libraries, lifecycle=lifecycle, comparisons=results,
        CPU_references={name: dict(context_sha256=record['context_sha256'],
                                  checkpoint_sha256=record['checkpoint_sha256'],
                                  record_sha256=(args.arm_reference_sha256 if name=='ARM'
                                                 else args.x86_reference_sha256),
                                  logits_sha256=record['last_logits']['sha256'],
                                  updated_KV_sha256=record['updated_KV']['sha256'])
                        for name, (record, _) in cpu.items()},
        CPU_ARM_vs_x86_logits=(tensor_difference(cpu['ARM'][1][0], cpu['x86'][1][0])
                               if len(cpu)==2 else None),
        repeated_library_outputs_bitexact=all(len(hashes)==1 for hashes in digests.values()),
        new_ARM_packets_independently_available=False, native_generate_execution=False,
        numerical_acceptance=False, performance_acceptance=False)
    args.output.mkdir()
    (args.output/'result.json').write_bytes(canonical(result)+b'\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
