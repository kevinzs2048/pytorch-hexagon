"""Restore a trusted pinned Qwen AOT graph, without execution or recapture.

This experimental internal PyTorch format is not safe for untrusted models or
network clients. A SHA check is integrity only, not authentication. Optional
cache inspection only reports matching identities; it does not load DSP code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import traceback

from capture_qwen_symbolic_graph_carrier import sha, CONTEXT_SHA, GENERATION_SHA, POLICY_SHA


def restore_carrier(producer, expected_sha, *, trusted_local=False):
    if trusted_local is not True:
        raise ValueError('Explicit trusted-local artifact required before pickle loading')
    producer = Path(producer)
    if producer.is_symlink() or not 0 < producer.stat().st_size <= 4*1024**2 or sha(producer) != expected_sha:
        raise ValueError('Bounded original producer record required')
    record = json.loads(producer.read_text())
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.fx._graph_pickler import GraphPickler
    from hexagon_captured_linalg_cache import captured_identity
    from hexagon_module_scalar_binding import binding_content_signature
    from hexagon_live_compile_mailbox import LIVE_STATUS
    live = record['status'] == LIVE_STATUS
    if live:
        producer_ok = (record['source_sha256'] == sha(Path(__file__).with_name('hexagon_live_compile_mailbox.py'))
            and record.get('request_graph_executed') is False
            and type(record.get('prior_session_DSP_calls')) is int
            and record['prior_session_DSP_calls'] >= 0 and record.get('host_machine') == 'aarch64')
    else:
        producer_ok = (record['status'] == 'Qwen_symbolic_AOT_carrier_captured_NOT_executed'
            and record['source_sha256'] == sha(Path(__file__).with_name('capture_qwen_symbolic_graph_carrier.py'))
            and record['actual_AOT_callbacks'] == 1 and record['DSP_calls'] == 0
            and record['expected_pre_forward_stop'] is True)
    if (not producer_ok or record['context_sha256'] != CONTEXT_SHA
            or record['generation_sha256'] != GENERATION_SHA
            or record['decomposition_sha256'] != POLICY_SHA
            or (record['torch_version'], record['torch_git']) != (torch.__version__, torch.version.git_version)
            or record['CPU_model_forward'] is not False):
        raise ValueError('Original pinned non-executed Qwen capture required')
    rows = record['immutable_inputs']
    from models.qwen3.runtime.support.guarded_module_constants import validate_capture_constants
    expected_bindings=validate_capture_constants(record)
    if record.get('guarded_module_constants') is not None and not live:
        raise ValueError('Native constant convention requires an actual live producer')
    arity = record['graph_identity']['inputs']
    indices = [row['input_index'] for row in rows]
    if (type(arity) is not int or not 1 <= arity <= 1024 or len(rows) != expected_bindings
            or any(type(i) is not int or not 0 <= i < arity for i in indices)
            or len(set(indices)) != len(indices)
            or sorted(row['input_index'] for row in record['named_bindings']) != sorted(indices)
            or record['retained_input_indices'] != [i for i in range(arity) if i not in indices]
            or binding_content_signature(rows) != record['binding_content_sha256']):
        raise ValueError('Immutable/retained argument mapping changed')
    artifact = record['artifact']
    if artifact['path'] != 'graph.pkl' or type(artifact['bytes']) is not int:
        raise ValueError('Fixed scoped graph artifact required')
    path = producer.parent/'graph.pkl'
    if (path.is_symlink() or not 0 < path.stat().st_size <= 64*1024**2
            or path.stat().st_size != artifact['bytes'] or sha(path) != artifact['sha256']):
        raise ValueError('Graph artifact bytes changed')
    graph, examples = GraphPickler.loads(path.read_bytes(), FakeTensorMode(shape_env=ShapeEnv()))
    graph.recompile()
    if (len(examples) != arity or list(graph.named_parameters()) or list(graph.named_buffers())
            or any(not is_fake(v) for v in examples if isinstance(v, torch.Tensor))
            or captured_identity(graph) != record['graph_identity']):
        raise ValueError('Actual graph/symbolic metadata changed in transport')
    return graph, examples, record


def inspect_cache(root, record):
    """Read-only identity matches, not proof of artifact integrity/execution."""
    parameters = hashlib.sha256(json.dumps(record['immutable_inputs'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    matches = []
    for path in sorted(Path(root).glob('*.json')):
        if path.is_symlink() or path.stat().st_size > 1024**2:
            raise ValueError('Bounded frontend cache metadata required')
        entry = json.loads(path.read_text())
        request = entry.get('request', {})
        if (entry.get('status') == 'captured_AOT_Linalg_cached_NOT_machine_code'
                and request.get('graph') == record['graph_identity']
                and request.get('parameters_sha256') == parameters
                and request.get('retained_input_indices') == record['retained_input_indices']):
            matches.append(dict(path=str(path), sha256=sha(path), key=entry['key']))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--producer-sha256', required=True)
    parser.add_argument('--trusted-local-producer', action='store_true')
    parser.add_argument('--inspect-frontend-cache', type=Path)
    parser.add_argument('--bind-context', type=Path,
                        help='Optional pinned compiler-host model context; verifies every weight leaf')
    parser.add_argument('--frontend-cache', type=Path,
                        help='Existing verified cache only; paired with --bind-context')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.trusted_local_producer:
        parser.error('Explicit trusted local producer required; never load untrusted pickle')
    if bool(args.bind_context) != bool(args.frontend_cache):
        parser.error('Host weight binding and existing frontend cache must be specified together')
    args.output.mkdir(exist_ok=False)
    result = dict(status='restoring_graph_only', source_sha256=sha(__file__),
        host_machine=platform.machine(), producer=dict(path=str(args.producer), sha256=args.producer_sha256),
        DSP_calls=0, CPU_model_forward=False, torch_mlir_executed=False,
        model_acceptance=False, stable_public_interchange=False)
    try:
        graph, examples, producer = restore_carrier(args.producer, args.producer_sha256, trusted_local=True)
        result.update(status='Qwen_symbolic_AOT_carrier_restored_NOT_Linalg_or_DSP',
            graph_identity=producer['graph_identity'], producer_machine=producer['host_machine'],
            graph_metadata_exact=True, argument_mapping_checked=True,
            graph_bytes=producer['artifact']['bytes'], concrete_weights_not_bound=True)
        if args.inspect_frontend_cache:
            result.update(matching_frontend_metadata=inspect_cache(args.inspect_frontend_cache, producer),
                          cached_artifact_bytes_verified=False, cache_hit_execution=False)
        if args.bind_context:
            if platform.machine() != 'x86_64' or sha(args.bind_context) != CONTEXT_SHA:
                raise ValueError('Original pinned compiler-host context required')
            from qwen3_generate_compile_common import load_model
            from hexagon_carrier_parameter_binding import CarrierParameterBinding
            from hexagon_captured_linalg_cache import CapturedLinalgCache
            from hexagon_runtime.hexagon_compiled_artifact_cache import canonical
            cache = CapturedLinalgCache(args.frontend_cache)
            matches = inspect_cache(args.frontend_cache, producer)
            frontend_sha = hashlib.sha256(canonical(cache.dependencies)).hexdigest()
            # This admission experiment must not accidentally start a fresh
            # 1.22GB lowering. A cold-compile integration is a later step.
            matches = [m for m in matches if json.loads(Path(m['path']).read_text())['request'].get(
                'frontend_sha256') == frontend_sha and json.loads(Path(m['path']).read_text())['request'].get(
                'function_name') == 'Qwen3DynamicBoundProbe']
            if len(matches) != 1:
                raise ValueError('Exactly one current frontend identity required; no implicit cold compile')
            model, _, _ = load_model(args.bind_context)
            binding = CarrierParameterBinding(model, producer)
            # Existing cache API rehashes the actual bytecode/artifacts, not
            # merely the small index whose identity matched above.
            artifact = cache.get_or_lower(graph, binding, name='Qwen3DynamicBoundProbe')
            if artifact['cache_hit'] is not True:
                raise ValueError('Expected actual verified cache hit')
            result.update(status='Qwen_symbolic_AOT_weights_verified_frontend_cache_hit_NOT_DSP',
                concrete_weights_not_bound=False, immutable_tensors_verified=len(binding.records),
                immutable_tensor_bytes=sum(row['bytes'] for row in binding.records),
                binding_source_sha256=sha(Path(__file__).with_name('hexagon_carrier_parameter_binding.py')),
                cached_artifact_bytes_verified=True, frontend_cache_key=artifact['key'],
                frontend_invocations=artifact['frontend_invocations'], machine_code_invocations=0,
                cache_hit_execution=False)
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
