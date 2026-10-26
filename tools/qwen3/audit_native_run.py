"""Independently audit all requests, complete packets and ARM W8A8 controls."""
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import argparse
from terminal_reports import read_terminal_report
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--sources', type=Path, required=True)
p.add_argument('--observations', type=Path, required=True)
p.add_argument('--board-observations', type=Path, required=True)
p.add_argument('--hf', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--name', required=True)
p.add_argument('--ssh-target', required=True)
a = p.parse_args()
F, R, O, BO = a.output, a.sources, a.observations, a.board_observations
F.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(R), str(R/'qwen3/runtime/arm')]
from audit_arm_dynamic_warm_run import audit_compile_handoff, audit_text_entry, audit_local_packet_transport
from audit_qwen_native_request_sequence import (
    audit_unchanged_kv, audit_cache_handoff, kv_packet_digests, audit_logits)
from qwen_native_request_sequence import verify_request
from audit_saved_packet_library_comparison import tensor_difference
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for, MAGIC
from hexagon_library_lifecycle import audit_library_events


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


name = a.name
source, control = O/name, O/(name+'-ARM-controls')
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',a.ssh_target]
run,run_sha=read_terminal_report(ssh,BO/name/'result.json',F/(name+'-native-result.json'))
cpu,cpu_sha=read_terminal_report(ssh,BO/(name+'-ARM-controls')/'result.json',
    F/(name+'-CPU-result.json'))
assert run['status'] == 'ARM_native_dynamic_compiler_handoff_generate_completed'
assert run['process_exit_code'] == 0 and not run['CPU_model_fallback']
assert run['complete_raw_packets_retained'] and run['DSP_calls'] == len(run['forwards'])
assert cpu['status'] == 'ARM_native_same_input_and_generate_numerics_recorded_NOT_acceptance'
assert run_sha == cpu['run_sha256']
assert len(run['requests']) == len(cpu['native_requests']) > 0
assert len(run['forwards']) == len(cpu['same_input']) > 0
# The current runner checks native cache identity before admitting completion;
# its request schema has no historical native_cache_object_reused field.
# Independently check ownership and all actual cache bytes below.
groups = []
start = 0
for index, request in enumerate(run['requests']):
    end = request['end_forward']
    assert request['index'] == index and request['first_forward'] == start
    assert type(end) is int and start < end <= len(run['forwards'])
    group = run['forwards'][start:end]
    assert all(row['request_index'] == index and row['step'] == start+i for i, row in enumerate(group))
    verify_request(group, request['prompt_ids'], request['generated_ids'])
    groups.append(group)
    start = end
assert start == len(run['forwards'])
handoff = audit_compile_handoff(run, O/('live-'+name+'-mailbox'), rebuilt=True)
text = audit_text_entry(run, str(a.hf))
transport = audit_local_packet_transport(run)
lifecycle = audit_library_events(run['stream_events'], [r['DSP_library_slot'] for r in run['forwards']])

dest = F/name
dest.mkdir(exist_ok=True)
members = ['result.json']
for row in run['forwards']:
    folder = f"forward-{row['step']:02d}-q{row['query']}"
    members.extend(folder+'/'+field+'.bin' for field in ('input', 'output'))
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', a.ssh_target,
           shlex.join(['tar', '--sparse', '-C', str(BO/name), '-cf', '-', *members])]
with (dest/'sparse-transfer.stderr.log').open('ab') as log:
    archive = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
    with archive.stdout:
        extract = subprocess.run(['tar', '--skip-old-files', '-C', str(dest), '-xf', '-'],
                                 stdin=archive.stdout, stderr=log)
    assert archive.wait() == extract.returncode == 0
shutil.copytree(control, dest/'ARM-controls', dirs_exist_ok=True)
shutil.copyfile(F/(name+'-CPU-result.json'),dest/'ARM-controls/result.json')
assert sha(dest/'result.json') == run_sha
assert sha(dest/'ARM-controls/result.json') == cpu_sha


def reference(record):
    path = dest/'ARM-controls'/Path(record['path']).relative_to(BO/control.name)
    assert sha(path) == record['sha256']
    return np.fromfile(path, dtype='<f4').reshape(record['shape'])


same, native, requests = [], [], []
all_pass = True
for request, group, independent_request in zip(run['requests'], groups, cpu['native_requests'], strict=True):
    assert independent_request['index'] == request['index']
    assert independent_request['same_forward_count'] and len(group) == len(independent_request['forwards'])
    request_pass = independent_request['same_generated_ids'] and independent_request['same_terminal']
    assert independent_request['generated_ids'] == request['generated_ids']
    assert independent_request['terminal_observation'] == request['terminal_observation']
    previous = None
    for row, independent in zip(group, independent_request['forwards'], strict=True):
        ref = cpu['same_input'][row['step']]
        assert ref['step'] == row['step'] and ref['inputs_exact']
        assert independent['same_positions'] and independent['same_token_ids']
        assert independent['positions'] == row['cache_positions']
        assert independent['token_ids'] == row['input_token_ids']
        folder = dest/f"forward-{row['step']:02d}-q{row['query']}"
        for field in ('input', 'output'):
            assert sha(folder/(field+'.bin')) == row[field+'_sha256'] == ref[field+'_sha256']
        incoming, raw = [(folder/(field+'.bin')).read_bytes() for field in ('input', 'output')]
        layout = layout_for(run['captures'][row['AOT_capture_index']]['contract'], {'query': row['query']})
        assert len(incoming) == 16+layout['input_bytes'] and len(raw) == layout['output_bytes']
        assert incoming[:8] == MAGIC and int.from_bytes(incoming[8:16], 'little', signed=True) == row['query']
        actual_inputs = [np.frombuffer(incoming, dtype='<i8', offset=16+s['offset'], count=s['bytes']//8).reshape(-1).tolist() for s in layout['inputs'][:3]]
        assert actual_inputs == [row['input_token_ids'], row['cache_positions'], row['cache_positions']]
        scope = audit_unchanged_kv(incoming, raw, layout, row['cache_positions'])
        audit_logits(raw, layout, row['logits_top_token'])
        if previous is None:
            assert row['cache_positions'][0] == 0
            fields = [s for s in layout['inputs'] if s.get('dtype') == 'float32' and s.get('shape') == [1, 8, 1024, 128]]
            assert len(fields) == 56
            for field in fields:
                values = np.frombuffer(incoming, dtype='<u4', offset=16+field['offset'], count=field['bytes']//4)
                assert not np.any(values)
            del values
        else:
            audit_cache_handoff(previous, incoming, layout)
        previous = kv_packet_digests(raw, layout, side='output')
        values = [np.frombuffer(raw, dtype='<f4', offset=s['offset'], count=s['bytes']//4).reshape(s['shape']) for s in layout['outputs']]
        assert len(values) == 57 and all(np.isfinite(v).all() for v in values)
        for refs, field, positions, rows, metric_field in (
                (ref, 'reference_updated_KV', row['cache_positions'], same, 'updated_KV'),
                (independent, 'reference_valid_KV', list(range(max(row['cache_positions'])+1)), native, 'valid_KV')):
            logits = tensor_difference(values[-1], reference(refs['reference_logits']))
            saved = reference(refs[field])
            kv = [tensor_difference(v[:, :, positions, :], saved[i]) for i, v in enumerate(values[:56])]
            assert logits == refs['logits'] and kv == refs[metric_field]
            error = max(x['max_abs_error'] for x in kv)
            passed = logits['max_abs_error'] < 1e-4 and error < 1e-4
            request_pass = request_pass and passed
            rows.append(dict(request=request['index'], step=row['step'], query=row['query'], logits=logits, KV_max_abs=error, passed=passed))
        del incoming, raw, values, saved
        print('audited', row['step'], 'query', row['query'], 'same', same[-1]['logits']['max_abs_error'], same[-1]['KV_max_abs'], 'native', native[-1]['logits']['max_abs_error'], native[-1]['KV_max_abs'], flush=True)
    requests.append(dict(index=request['index'], generated_ids=request['generated_ids'], terminal=request['terminal_observation'], passed=bool(request_pass)))
    all_pass = all_pass and request_pass

reuse = []
for index, capture in enumerate(run['captures']):
    uses = [row for row in run['forwards'] if row['AOT_capture_index'] == index]
    assert uses and all(row['DSO_sha256'] == capture['DSO_sha256'] for row in uses)
    reuse.append(dict(capture=index, query_bounds=capture['contract']['variables'], queries=sorted({r['query'] for r in uses}), requests=sorted({r['request_index'] for r in uses}), calls=len(uses), DSO_sha256=capture['DSO_sha256']))
result = dict(status='framework_native_numerical_acceptance_passed' if all_pass else 'framework_native_numerical_acceptance_failed', source_sha256=sha(__file__), run_sha256=run_sha, CPU_record_sha256=cpu_sha, packet_transfer='GNU sparse tar; complete logical bytes independently hashed; no mmap', handoff=handoff, text=text, transport=transport, library_lifecycle=lifecycle, cache_continuity_bitexact=True, request_reset_all_zero=True, same_input=same, independent_native_trajectory=native, requests=requests, capture_reuse=reuse, numerical_acceptance=bool(all_pass), tolerance=1e-4)
(F/(name+'-independent-audit.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('same_input', 'independent_native_trajectory')}, indent=2))
if not all_pass:
    sys.exit(1)
