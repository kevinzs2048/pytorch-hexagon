"""Capture Qwen's actual dynamic AOT graph without running its forward.

An isolated transport feasibility check, not a new backend or generation loop.
Uses a pinned internal PyTorch serializer; only trusted local artifacts may be
restored. Weights are identified but NOT serialized as tensor payloads.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback


CONTEXT_SHA = '30ed4c7d3a31f97b07f73edd4c3716f47e6092314c7db2d7bc4a4cd3a88885c7'
GENERATION_SHA = '6ecaf17fc6ad5772cb2108a77082831f13311642eaf792fe8d15ac6a2125fda3'
POLICY_SHA = '2c1583eb71ef9a3f7791d312f2cef7badb413068950134a39729f19b2b7f48e5'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class StopBeforeForward(RuntimeError):
    pass


def allowed_carrier_operator(name):
    """Only native ATen plus the observed built-in dtype conversion primitive.

    Do not disable GraphPickler's op filter: custom runtime registrations are
    not automatically portable to the compiler host.
    """
    return name.startswith('torch.ops.aten.') or name == 'torch.ops.prims.convert_element_type.default'


def capture(args, record, save):
    import torch
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd
    from torch._subclasses.fake_tensor import is_fake
    from torch.fx._graph_pickler import GraphPickler, Options
    from transformers import Qwen3ForCausalLM, CompileConfig
    from transformers.generation import utils as generation_utils
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    from hexagon_native1024_cache_setup import prepare_native1024_cache
    from hexagon_native_silu import preserve_native_silu
    from hexagon_runtime_decompositions import load_runtime_decompositions
    from hexagon_dynamic_module_constants import DynamicParameterBinding
    from hexagon_module_scalar_binding import bind_module_scalar_attributes, binding_content_signature, module_attribute_owner
    from hexagon_captured_linalg_cache import captured_identity

    if sha(args.context) != CONTEXT_SHA or sha(generation_utils.__file__) != GENERATION_SHA:
        raise ValueError('Previously pinned model context and generation source required')
    context = json.loads(args.context.read_text())
    if any(importlib.metadata.version(name) != context['versions'][name]
           for name in ('torch', 'torchao', 'transformers')):
        raise ValueError('Native framework versions differ')
    if sha(args.hf/'model.safetensors') != context['checkpoint']['sha256']:
        raise ValueError('Original checkpoint required')
    table = preserve_native_silu(load_runtime_decompositions(args.decompositions,
        expected_sha256=POLICY_SHA, expected_torch_version='2.12.0+cpu',
        expected_torch_git='7661cd9c6b841b62b7f411aa52ec51f05457263b'))
    torch.set_num_threads(2)
    model = Qwen3ForCausalLM.from_pretrained(str(args.hf), local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').eval().requires_grad_(False)
    config = json.loads(json.dumps(model.config.to_dict()))
    expected = dict(context['model_config'])
    config.pop('_name_or_path', None)
    expected.pop('_name_or_path', None)
    if config != expected:
        raise ValueError('Relocated model configuration differs')
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        data = value.detach().contiguous().view(torch.uint8).numpy()
        digest.update((name+':'+hashlib.sha256(memoryview(data).cast('B')).hexdigest()+'\n').encode())
    if digest.hexdigest() != context['historical_parameter_sha256']:
        raise ValueError('Named model parameters differ')
    quantize_(model, Int8DynamicActivationInt8WeightConfig(set_inductor_config=False))
    cache = prepare_native1024_cache(model)
    original = (model.forward.__func__, model.generate.__func__)
    record.update(status='capturing_native_dynamic_AOT', torch_version=torch.__version__,
        torch_git=torch.version.git_version, context_sha256=CONTEXT_SHA,
        generation_sha256=GENERATION_SHA, decomposition_sha256=POLICY_SHA,
        checkpoint_sha256=context['checkpoint']['sha256'])
    save()
    callbacks = 0

    def backend(dynamo_graph, dynamo_inputs):
        def compiler(aten, examples):
            nonlocal callbacks
            callbacks += 1
            if callbacks != 1:
                raise ValueError('Only first real AOT capture is requested')
            tracing = torch._guards.TracingContext.get()
            binding = DynamicParameterBinding(dynamo_graph, dynamo_inputs, aten, examples,
                input_info=tracing.fw_metadata.input_info, retain_snapshots=False)
            allowed = {'variance_epsilon', 'scaling', 'attention_scaling'}
            scalar_rows = bind_module_scalar_attributes(binding, dynamo_graph, dynamo_inputs,
                aten, examples, input_info=tracing.fw_metadata.input_info, roots={'self': model},
                allowed_attributes=allowed, record_kind='immutable_Qwen3_module_scalar')
            if len(binding.records) != 651 or len(scalar_rows) != 142:
                raise ValueError('Selected Qwen immutable input profile changed')
            if (list(aten.named_parameters()) or list(aten.named_buffers())
                    or any(not is_fake(x) for x in examples if isinstance(x, torch.Tensor))):
                raise ValueError('Lifted graph and fake examples required; no model payload')
            # Typed actual source/object identities, never eval source strings.
            names = {}
            for kind, items in (('parameter', model.named_parameters(remove_duplicate=False)),
                                ('buffer', model.named_buffers(remove_duplicate=False))):
                for name, value in items:
                    names.setdefault(id(value), []).append(dict(kind=kind, name=name))
            modules = {id(value): name for name, value in model.named_modules()}
            dynamo_nodes = [n for n in dynamo_graph.graph.nodes if n.op == 'placeholder']
            named = []
            for row in binding.records:
                index = row['dynamo_input']
                if row.get('kind') == 'immutable_Qwen3_module_scalar':
                    source = dynamo_nodes[index].meta['grapharg'].source
                    owner, attribute = module_attribute_owner(source, {'self': model}, allowed)
                    named.append(dict(input_index=row['input_index'], module=modules[id(owner)],
                                      attribute=attribute, dtype=row['spec']['dtype']))
                else:
                    aliases = names.get(id(dynamo_inputs[index]))
                    if not aliases:
                        raise ValueError('Actual native named parameter identity missing')
                    named.append(dict(input_index=row['input_index'], aliases=aliases,
                                      attributes=row['attributes']))
            before = captured_identity(aten)
            record.update(actual_AOT_callbacks=callbacks, graph_identity=before,
                immutable_inputs=binding.records, named_bindings=named,
                retained_input_indices=binding.retained,
                binding_content_sha256=binding_content_signature(binding.records),
                status='serializing_lifted_graph_without_weight_payload')
            save()
            payload = GraphPickler.dumps((aten, examples), Options(ops_filter=allowed_carrier_operator))
            if not 0 < len(payload) <= 64*1024**2:
                raise ValueError('Graph-only carrier exceeds 64 MiB cap')
            target = args.output/'graph.pkl'
            target.write_bytes(payload)
            if captured_identity(aten) != before:
                raise ValueError('Serializer changed original captured graph')
            record.update(actual_AOT_callbacks=callbacks, graph_identity=before,
                immutable_inputs=binding.records, named_bindings=named,
                retained_input_indices=binding.retained,
                binding_content_sha256=binding_content_signature(binding.records),
                artifact=dict(path=target.name, sha256=sha(target), bytes=len(payload)))
            save()
            def stop(*values):
                raise StopBeforeForward('Captured only: no model forward executed')
            return make_boxed_func(stop)
        return aot_autograd(fw_compiler=compiler, inference_compiler=compiler,
                            decompositions=table)(dynamo_graph, dynamo_inputs)

    def no_eager(module, args_, kwargs):
        if not torch.compiler.is_compiling():
            raise RuntimeError('Eager CPU model execution forbidden')
    hook = model.register_forward_pre_hook(no_eager, with_kwargs=True)
    stopped = False
    try:
        cfg = CompileConfig(backend=backend, fullgraph=True, dynamic=True, mode='default')
        cfg._compile_all_devices = True
        tokens = torch.arange(10, 10+args.query).reshape(1, -1)
        with torch.no_grad(), torch._dynamo.config.patch(suppress_errors=False):
            model.generate(tokens, attention_mask=torch.ones_like(tokens),
                cache_implementation='static', do_sample=False, max_new_tokens=1,
                prefill_chunk_size=32, compile_config=cfg)
    except StopBeforeForward:
        stopped = True
    finally:
        hook.remove()
    if (not stopped or callbacks != 1 or model._cache is not cache
            or original != (model.forward.__func__, model.generate.__func__)):
        raise ValueError('Expected native capture then pre-forward stop')
    record.update(status='Qwen_symbolic_AOT_carrier_captured_NOT_executed',
                  expected_pre_forward_stop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--hf', type=Path, required=True)
    parser.add_argument('--decompositions', type=Path, required=True)
    parser.add_argument('--query', type=int, choices=range(2, 33), default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    record = dict(status='loading_native_capture_model', source_sha256=sha(__file__),
        host_machine=platform.machine(), query=args.query, DSP_calls=0,
        CPU_model_forward=False, model_acceptance=False, torch_mlir_executed=False,
        stable_public_interchange=False, trusted_pickle_only=True)
    def save():
        (args.output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        capture(args, record, save)
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save()
    print(record['status'], flush=True)


if __name__ == '__main__':
    main()
