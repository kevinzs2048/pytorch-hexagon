"""Trusted, version-pinned ARM AOT -> compiler-host file handoff.

This is an experiment on a private shared directory, NOT a network service or
safe interchange for untrusted pickle. Hashes detect corruption, not attackers.
Only compiler requests use this channel; tensors for inference remain on ARM.
"""
import json
from pathlib import Path
import platform
import time

from capture_qwen_symbolic_graph_carrier import (
    sha, CONTEXT_SHA, GENERATION_SHA, POLICY_SHA, allowed_carrier_operator)

LIVE_STATUS = 'live_ARM_AOT_compile_request_v1'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    temporary.replace(path)


def named_bindings(model, dynamo_graph, dynamo_inputs, binding):
    from hexagon_module_scalar_binding import module_attribute_owner
    names = {}
    for kind, items in (('parameter', model.named_parameters(remove_duplicate=False)),
                        ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, value in items:
            names.setdefault(id(value), []).append(dict(kind=kind, name=name))
    modules = {id(value): name for name, value in model.named_modules()}
    nodes = [node for node in dynamo_graph.graph.nodes if node.op == 'placeholder']
    result = []
    for row in binding.records:
        index = row['dynamo_input']
        if row.get('kind') == 'immutable_Qwen3_module_scalar':
            owner, attribute = module_attribute_owner(nodes[index].meta['grapharg'].source,
                {'self': model}, {'variance_epsilon', 'scaling', 'attention_scaling'})
            result.append(dict(input_index=row['input_index'], module=modules[id(owner)],
                               attribute=attribute, dtype=row['spec']['dtype']))
        else:
            aliases = names.get(id(dynamo_inputs[index]))
            if not aliases:
                raise ValueError('Actual named parameter identity missing in live capture')
            result.append(dict(input_index=row['input_index'], aliases=aliases,
                               attributes=row['attributes']))
    return result


def publish_capture(directory, model, dynamo_graph, dynamo_inputs, graph, examples,
                    binding, *, prior_dsp_calls, guarded_constants=None):
    import torch
    from torch._subclasses.fake_tensor import is_fake
    from torch.fx._graph_pickler import GraphPickler, Options
    from hexagon_captured_linalg_cache import captured_identity
    from hexagon_module_scalar_binding import binding_content_signature
    if guarded_constants is not None:
        from tools.qwen3.guarded_module_constants import GuardedModuleConstants
        if not isinstance(guarded_constants,GuardedModuleConstants):
            raise ValueError('Native module constant guard required')
        guarded_constants.check()
    expected_bindings=651 if guarded_constants is None else 509
    if (type(prior_dsp_calls) is not int or prior_dsp_calls < 0
            or len(binding.records) != expected_bindings or list(graph.named_parameters())
            or list(graph.named_buffers())
            or any(not is_fake(v) for v in examples if isinstance(v, torch.Tensor))):
        raise ValueError('Lifted selected Qwen profile required for live compile request')
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    identity = captured_identity(graph)
    record = dict(status=LIVE_STATUS, source_sha256=sha(__file__), host_machine=platform.machine(),
        torch_version=torch.__version__, torch_git=torch.version.git_version,
        context_sha256=CONTEXT_SHA, generation_sha256=GENERATION_SHA, decomposition_sha256=POLICY_SHA,
        graph_identity=identity, immutable_inputs=binding.records,
        named_bindings=named_bindings(model, dynamo_graph, dynamo_inputs, binding),
        retained_input_indices=binding.retained,
        binding_content_sha256=binding_content_signature(binding.records),
        prior_session_DSP_calls=prior_dsp_calls, CPU_model_forward=False,
        request_graph_executed=False, stable_public_interchange=False)
    if guarded_constants is not None:
        from tools.qwen3 import guarded_module_constants as constants
        record['guarded_module_constants']=dict(schema=constants.SCHEMA,
            source_sha256=sha(constants.__file__),records=guarded_constants.records,
            sha256=guarded_constants.signature)
        if constants.validate_capture_constants(record)!=expected_bindings:
            raise ValueError('Actual native constant convention differs')
    payload = GraphPickler.dumps((graph, examples), Options(ops_filter=allowed_carrier_operator))
    if not 0 < len(payload) <= 64*1024**2 or captured_identity(graph) != identity:
        raise ValueError('Live graph size or identity changed')
    artifact = directory/'graph.pkl'
    artifact.write_bytes(payload)
    record['artifact'] = dict(path='graph.pkl', bytes=len(payload), sha256=sha(artifact))
    atomic_json(directory/'result.json', record)
    request_sha = sha(directory/'result.json')
    atomic_json(directory/'ready.json', dict(schema=1, producer_sha256=request_sha))
    return request_sha


def validate_response(response, request_sha, result):
    if (response.get('schema') != 1 or response.get('producer_sha256') != request_sha
            or response.get('status') != 'compiled'
            or result.get('status') != 'carried_ARM_Qwen_graph_bound_to_Hexagon_library_NOT_DSP'
            or result.get('producer', {}).get('sha256') != request_sha
            or result.get('DSP_calls') != 0 or result.get('CPU_model_forward') is not False):
        raise ValueError('Compiler response is failed, stale or not bound to this actual capture')
    return result['compiled_entry']


def wait_for_compilation(directory, request_sha, *, timeout=1800):
    directory = Path(directory)
    deadline = time.monotonic()+timeout
    response_path = directory/'response.json'
    while not response_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Compiler-host response deadline; no CPU model fallback')
        time.sleep(0.25)
    if response_path.is_symlink() or response_path.stat().st_size > 16384:
        raise ValueError('Bounded local compiler response required')
    response = json.loads(response_path.read_text())
    if response.get('status') != 'compiled':
        raise RuntimeError('Compiler-host transaction failed: '+str(response.get('error', 'unknown')))
    result_path = directory/'compiled'/'result.json'
    if (result_path.is_symlink() or result_path.stat().st_size > 4*1024**2
            or sha(result_path) != response.get('result_sha256')):
        raise ValueError('Compiler result changed after publication')
    result = json.loads(result_path.read_text())
    entry = validate_response(response, request_sha, result)
    return entry, result


def checked_library_path(root, relative, expected_sha):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Compiler library must remain inside selected machine cache')
    path = root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != expected_sha:
        raise ValueError('Actual compiled library outside cache or bytes changed')
    return path
