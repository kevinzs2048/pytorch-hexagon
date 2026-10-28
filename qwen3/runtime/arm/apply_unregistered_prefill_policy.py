"""Apply an explicitly checked arithmetic candidate to actual weighted graphs.

Streams the existing large text IR into the optimizer with one function
annotation; no second raw weight/IR file. New machine code/DSP remain unaccepted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from hexagon_compiled_artifact_cache import digest_file as sha


def annotate_prefix(prefix):
    marker = b'hexagon.fp32_prefill_capacity = 1024 : i64'
    if prefix.count(marker) != 1 or b'hexagon.fp32_prefill_order' in prefix:
        raise ValueError('Require one unannotated native prefill function in the bounded IR prefix')
    return prefix.replace(marker, b'hexagon.fp32_prefill_order = "qk-fma-pv-openblas-armv8-k1024", '+marker)


def stream_policy(argv, source, expected_sha256, stderr, timeout=1200):
    def stop(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    original, supplied = hashlib.sha256(), hashlib.sha256()
    with Path(stderr).open('xb') as log:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
        timer = threading.Timer(timeout, stop, args=(process,))
        timer.start()
        try:
            with Path(source).open('rb') as stream:
                prefix = stream.read(65536)
                original.update(prefix)
                changed = annotate_prefix(prefix)
                supplied.update(changed)
                process.stdin.write(changed)
                while chunk := stream.read(1024**2):
                    original.update(chunk)
                    supplied.update(chunk)
                    process.stdin.write(chunk)
            process.stdin.close()
            code = process.wait(timeout=timeout)
            if code or original.hexdigest() != expected_sha256:
                raise ValueError('Optimizer failed or actual source bytes changed')
            return dict(command=argv, returncode=code, original_sha256=original.hexdigest(),
                        annotated_stream_sha256=supplied.hexdigest(), duplicate_raw_IR_written=False)
        finally:
            timer.cancel()
            if process.poll() is None:
                stop(process)
                process.wait(timeout=10)
            if not process.stdin.closed:
                process.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frontend', 'attention', 'attention-audit', 'build'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bound = []
    def load(path, digest):
        if sha(path) != digest:
            raise ValueError('Explicit input binding differs: '+str(path))
        bound.append(dict(path=str(path), sha256=digest))
        return json.loads(path.read_text())
    front = load(args.frontend, args.frontend_sha256)
    attention = load(args.attention, args.attention_sha256)
    audit = load(args.attention_audit, args.attention_audit_sha256)
    build = load(args.build, args.build_sha256)
    optimizer = args.build.parent/'linalg-hexagon-opt'
    if (front['status'] != 'received_native_weighted_Linalg_prepared_NOT_machine_code_or_DSP'
            or front['DSP_calls'] != 0 or front['graph_recaptured'] is not False
            or attention['status'] != 'native_new_length_attention_controls_recorded_NOT_DSP'
            or audit['status'] != 'native_new_length_strided_attention_readbacks_bitexact_NOT_DSP'
            or audit['producer_sha256'] != args.attention_sha256
            or build['status'] != 'explicit_variable_prefill_optimizer_built_NOT_DSP'
            or sha(optimizer) != build['optimizer_sha256']):
        raise ValueError('Complete actual frontend, independent native control and isolated tool required')
    args.output.mkdir(exist_ok=False)
    record = dict(status='applying_policy', DSP_calls=0, numerical_acceptance=False,
                  machine_code_compilations=0, optimizer=dict(path=str(optimizer), sha256=sha(optimizer)),
                  bindings=bound, source_sha256=sha(__file__), graphs=[])
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    def save():
        (args.output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        for index, graph in enumerate(front['graphs']):
            query = graph['query_tokens']
            controls = [r for r in attention['queries'] if r['query_tokens'] == query]
            readback = [r for r in audit['summary'] if r['query_tokens'] == query]
            if (len(controls) != 1 or controls[0]['policy_nonexact_tensors'] != 0
                    or len(readback) != 1 or readback[0]['strided_native_replays_bitexact'] != 56
                    or attention['context']['sha256'] != graph['binding']['context_sha256']
                    or sha(graph['linalg']['path']) != graph['linalg']['sha256']):
                raise ValueError('Same-shape native arithmetic policy controls required')
            folder = args.output/f'graph-{index}'
            folder.mkdir()
            target = folder/'graph.mlirbc'
            flags = ('fp32-attention-dot-order=true short-exp-sum-tree=true short-softmax-reciprocal=true '
                     'short-softmax-exp-compat=true fp32-silu-exp-compat=true')
            argv = [str(optimizer), '-', '--verify-each', '--emit-bytecode',
                    '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum{'+flags+'}),strip-debuginfo)', '-o', str(target)]
            command = stream_policy(argv, graph['linalg']['path'], graph['linalg']['sha256'], folder/'policy.log')
            structure = subprocess.run([str(optimizer), str(target), '--mlir-elide-elementsattrs-if-larger=16',
                                       '--mlir-elide-resource-strings-if-larger=16'], capture_output=True, text=True, timeout=300)
            (folder/'structure.stderr').write_text(structure.stderr)
            structure.check_returncode()
            (folder/'structure.mlir').write_text(structure.stdout)
            expected = {'linalg.matmul': 197, 'hexagon.fp32_softmax_exp_compat': 28,
                        'hexagon.fp32_silu_exp_compat': 28, 'math.exp': 0, 'math.cos': 0, 'math.sin': 0,
                        'hexagon.fp32_prefill_dot_order': 56, 'hexagon.fp32_prefill_pv_block_start': 168}
            counts = {key: structure.stdout.count(key) for key in expected}
            if counts != expected:
                raise ValueError('Full native model numerical policy coverage differs: '+str(counts))
            record['graphs'].append(dict(query_tokens=query, command=command, structure_counts=counts,
                inputs=graph['preparation']['contract']['inputs'], outputs=graph['preparation']['contract']['outputs'],
                input_update_outputs=graph['preparation']['contract']['input_update_outputs'],
                function_name=graph['linalg']['function_name'],
                graph=dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size)))
            save()
            print('Actual new-length weighted policy', query, 'verified structurally NOT DSP', flush=True)
        if any(sha(r['path']) != r['sha256'] for r in bound) or sha(optimizer) != build['optimizer_sha256']:
            raise ValueError('Bound inputs changed while optimizing')
        record['status'] = 'new_length_weighted_policy_applied_NOT_machine_code_or_DSP'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
