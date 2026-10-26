"""Compare the same complete model packets at both runtime dispatch branches.

The three original q8/q7/q6 packets retain their fresh ARM W8A8 oracle.
Only q2/q32 are derived from the proved empty q8 packet; their oracle is
the unchanged accepted DSP library. Every input/output is retained sparsely.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

from probe_dynamic_pv_model_arm import resize_cold_packet
from probe_zero_tail_q1_model_arm import retain_exact_sparse, sha


def first_difference(contract, query, expected, actual):
    import numpy as np
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for
    left = np.frombuffer(expected, dtype='<u4')
    right = np.frombuffer(actual, dtype='<u4')
    indices = np.flatnonzero(left != right)[:64]
    outputs = layout_for(contract, {'query': query})['outputs']
    rows = []
    for word in indices:
        offset = int(word) * 4
        tensor = next((i for i, r in enumerate(outputs)
                       if r['offset'] <= offset < r['offset'] + r['bytes']), None)
        row = dict(byte_offset=offset, expected_bits=int(left[word]), actual_bits=int(right[word]))
        if tensor is not None:
            r = outputs[tensor]
            row.update(output_tensor=tensor,
                       kind='logits' if tensor == 56 else ('K' if tensor % 2 == 0 else 'V'),
                       layer=None if tensor == 56 else tensor // 2,
                       index=list(map(int, np.unravel_index((offset-r['offset'])//4, r['shape']))))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'manifest', 'catalog', 'constant-load-split-proofs', 'workdir', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--deadline-unix', type=int, required=True)
    a = p.parse_args()
    if (platform.machine() != 'aarch64' or sys.byteorder != 'little' or a.output.exists()
            or not a.output.is_absolute() or sha(a.manifest) != a.manifest_sha256
            or a.deadline_unix-time.time() < 1800):
        raise ValueError('Actual ARM, pinned manifest and admitted model comparison required')
    m = json.loads(a.manifest.read_text())
    if (m['status'] != 'Q2_model_comparison_manifest_prepared_NOT_executed'
            or sha(a.catalog) != m['catalog_sha256'] or m['queries'] != [8,7,6,2,32]
            or m['DSP_calls'] or m['model_inference_acceptance']):
        raise ValueError('Exact prepared full model diagnostic contract required')
    sys.path[:0] = [str(a.sources/'qwen3/runtime/arm'), str(a.sources)]
    from hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import stage
    from hexagon_live_compile_mailbox import atomic_json
    from probe_model_loop_regions_arm import stage_constants
    packet_inputs = {}
    for row in m['packets']:
        path = a.manifest.parent/('input-q'+str(row['query'])+'.bin')
        if sha(path) != row['input']['sha256']:
            raise ValueError('Original accepted ARM input packet changed')
        reference = a.manifest.parent/('reference-q'+str(row['query'])+'.bin')
        if sha(reference) != row['output']['sha256']:
            raise ValueError('Original CPU-accepted complete output reference changed')
        packet_inputs[row['query']] = path
    libraries = {}
    layout = stage_constants(stage, json.loads(a.catalog.read_text()), a.workdir,
                             m['constants'], a.constant_load_split_proofs)
    for role in ('baseline','candidate'):
        source = a.manifest.parent/(role+'.so')
        want = m['libraries'][role]['sha256']
        if sha(source) != want:
            raise ValueError('Pinned baseline/candidate library changed')
        destination = a.workdir/(want+'.so')
        shutil.copy2(source,destination)
        if sha(destination) != want:
            raise ValueError('Staged full model library changed')
        libraries[role] = str(destination)
    a.output.mkdir()
    record = dict(status='Q2_model_ABBA20_running_NOT_accepted', source_sha256=sha(__file__),
                  manifest_sha256=a.manifest_sha256, deadline_unix=a.deadline_unix,
                  constant_layout=layout, ready=None, rows=[], cases=[], DSP_calls=0,
                  original_ARM_W8A8_queries=[8,7,6], derived_DSP_reference_queries=[2,32],
                  model_inference_acceptance=False, performance_acceptance=False,
                  diagnostic_instrumentation=False, power_votes_changed=False)
    save = lambda: atomic_json(a.output/'result.json', record)
    save()
    previous = Path.cwd()
    env = dict(os.environ, LD_LIBRARY_PATH=str(a.workdir),
               ADSP_LIBRARY_PATH=str(a.workdir)+';'+str(a.workdir/'rfsa')+
               ';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    command = ['flock','-n','/tmp/qwen3-qcs9075-probe.lock',str(a.workdir/'buffer_stream'),
               libraries['baseline'],str(m['input_bytes']),str(m['output_bytes'])]
    try:
        os.chdir(a.workdir)
        (a.output/'stream').mkdir()
        with HexagonSharedPacketStream(command, env, a.output/'stream',
                input_bytes=m['input_bytes'], output_bytes=m['output_bytes'],
                initial_library=libraries['baseline'], timeout=240) as stream:
            record['ready'] = stream.start(); save()
            original = packet_inputs[8].read_bytes()
            for query in m['queries']:
                payload = (packet_inputs[query].read_bytes() if query in packet_inputs else
                           resize_cold_packet(m['contract'], original, query))
                original_row = next((r for r in m['packets'] if r['query']==query), None)
                expected_sha = original_row['output']['sha256'] if original_row else None
                input_path = a.output/('input-q'+str(query)+'.bin')
                retain_exact_sparse(input_path,payload)
                expected_output = ((a.manifest.parent/('reference-q'+str(query)+'.bin')).read_bytes()
                                   if original_row else None)
                for index, role in enumerate(('baseline','candidate','candidate','baseline')):
                    if a.deadline_unix-time.time() < 300:
                        raise TimeoutError('No new invoke after model comparison admission expires')
                    record['pending'] = dict(query=query,index=index,role=role); save()
                    stream.select_library(libraries[role])
                    output,event = stream.invoke_packet(payload,m['output_bytes'])
                    record['DSP_calls'] += 1
                    record['pending']['actual_invoke'] = event; save()
                    path = a.output/f'output-q{query}-{index}-{role}.bin'
                    retain_exact_sparse(path,output)
                    actual_sha = sha(path)
                    if expected_sha is None:
                        expected_sha = actual_sha
                    fnvs = [event[k] for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')]
                    row = dict(query=query,index=index,role=role,input_file=input_path.name,
                               input_sha256=sha(input_path),output_file=path.name,
                               output_sha256=actual_sha,actual_invoke=event,
                               all_logits_and_56_KV_byteexact=actual_sha==expected_sha)
                    record['rows'].append(row); save()
                    if (any(event[k] for k in ('rpc_result','DSP_result','output_result'))
                            or len(set(fnvs)) != 1
                            or (original_row and fnvs[0] != original_row['raw_FNV_recomputed'])
                            or actual_sha != expected_sha):
                        if expected_output is not None:
                            record['first_differences'] = first_difference(m['contract'],query,expected_output,output)
                        raise ValueError('Complete logits/KV or actual three input FNV values differ; raw packets retained')
                    if index == 0:
                        expected_output = output
                    record.pop('pending'); save()
                record['cases'].append(query); save()
        if stream.process_exit_code or record['DSP_calls'] != 20:
            raise ValueError('Actual complete ABBA20 and normal stream close required')
        record.update(status='actual_Q2_model_ABBA20_full_logits_and_56KV_bitexact_NOT_generation_acceptance',
                      process_exit_code=stream.process_exit_code,stream_events=stream.events,
                      completed_unix=time.time())
        os.chdir(previous); shutil.rmtree(a.workdir)
        record['workdir_removed'] = True; save()
    except BaseException:
        record.update(status='failed',error=traceback.format_exc()); save(); raise
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    main()
