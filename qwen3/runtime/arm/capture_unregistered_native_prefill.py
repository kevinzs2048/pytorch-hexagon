"""Capture an actual native prefill without a per-length reference graph.

Diagnostic frontend candidate ONLY: stops before executing the model. No DSP,
torch-mlir lowering, executable admission, or CPU model fallback is provided.
The existing accepted backend and its source seals remain unchanged.
"""
import argparse
from collections import Counter
import copy
import hashlib
import inspect
import json
from pathlib import Path
import platform
import traceback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_length(query, capacity=1024):
    # One new token is requested, so leave one position within the native cache.
    if type(query) is not int or not 1 <= query < capacity:
        raise ValueError('Require a positive prompt length below cache capacity')
    return query


def named_bindings(model, inputs, records):
    """Bind by real object identity, never by parsing a Dynamo display name."""
    names = {}
    for kind, items in (('parameter', model.named_parameters(remove_duplicate=False)),
                        ('buffer', model.named_buffers(remove_duplicate=False))):
        for name, tensor in items:
            names.setdefault(id(tensor), []).append(dict(kind=kind, name=name))
    result = []
    seen = set()
    for row in records:
        index = row['input_index']
        original = row['dynamo_input']
        if index in seen or type(original) is not int or not 0 <= original < len(inputs):
            raise ValueError('Invalid or duplicate parameter binding')
        seen.add(index)
        aliases = names.get(id(inputs[original]))
        if not aliases:
            raise ValueError('Captured weights must belong to the native model')
        result.append(dict(input_index=index, aliases=aliases, attributes=row['attributes']))
    return result


class BeforeModelExecution(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-tokens', type=int, required=True)
    parser.add_argument('--arm-bundle', type=Path, required=True)
    parser.add_argument('--arm-bundle-sha256', required=True)
    parser.add_argument('--runtime-check', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    query = validate_length(args.query_tokens)
    if platform.machine() != 'aarch64':
        raise ValueError('This probe requires the actual pinned ARM framework')
    args.output.mkdir(exist_ok=False)
    record = dict(status='admitting_ARM_environment', query_tokens=query, capacity=1024,
                  reference_shape_graph_required=False, prefill_chunk_size=None,
                  DSP_calls=0, CPU_model_execution=False, torch_mlir_executed=False,
                  machine_code_compilations=0, model_generation_acceptance=False,
                  source_sha256=sha(__file__))
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())

    def save():
        (args.output/'result.json').write_text(json.dumps(record, indent=2)+'\n')

    save()
    try:
        from hexagon_arm_graph_capture_setup import admit_arm_environment, load_arm_model
        bundle, pm = admit_arm_environment(args.arm_bundle, args.arm_bundle_sha256, args.runtime_check)
        import torch
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch._subclasses.fake_tensor import is_fake
        from torch.utils._python_dispatch import _disable_current_modes, TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        from transformers.cache_utils import StaticCache
        from hexagon_dynamo_parameters import ParameterBinding, resolve_descriptor, spec
        from hexagon_native_graph_identity import graph_identity
        from hexagon_native1024_cache_setup import prepare_native1024_cache
        from hexagon_runtime_decompositions import load_runtime_decompositions
        from hexagon_native_silu import preserve_native_silu
        from qwen3_generate_compile_common import generation_options

        torch.set_num_threads(2)
        deps = [Path(inspect.getsourcefile(fn)) for fn in
                (admit_arm_environment, load_arm_model, ParameterBinding, resolve_descriptor,
                 graph_identity, prepare_native1024_cache, load_runtime_decompositions,
                 preserve_native_silu, generation_options)]
        record.update(sources=[dict(path=str(p), sha256=sha(p)) for p in sorted(set(deps))],
                      arm_bundle=dict(path=str(bundle.root), sha256=bundle.sha256),
                      context=dict(path=str(bundle.root/'context.json'), sha256=sha(bundle.root/'context.json')),
                      torch_version=torch.__version__, torch_git=torch.version.git_version,
                      status='loading_original_native_model')
        save()
        model, context = load_arm_model(bundle, pm)
        cache = prepare_native1024_cache(model)
        original = (model.forward.__func__, model.generate.__func__, StaticCache.update, StaticCache.reset)
        policy = bundle.manifest['decomposition_policy']
        table = preserve_native_silu(load_runtime_decompositions(bundle.root/policy['path'],
            expected_sha256=policy['sha256'], expected_torch_version=policy['torch_version'],
            expected_torch_git=policy['torch_git']))
        tokens = torch.arange(10, 10+query, dtype=torch.int64).unsqueeze(0)
        count = 0

        class NoRealModelCompute(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if str(func) in ('aten._int_mm.default', 'aten.mm.default', 'aten.bmm.default',
                                  'aten.addmm.default', 'aten.embedding.default'):
                    if any(isinstance(v, torch.Tensor) and not is_fake(v)
                           for v in tree_leaves((args, kwargs or {}))):
                        raise RuntimeError('Real CPU model computation forbidden during export')
                return func(*args, **(kwargs or {}))

        def backend(gm, inputs):
            def compile_aten(aten, examples):
                nonlocal count
                count += 1
                if count != 1:
                    raise ValueError('Expected one complete prefill AOT callback')
                nodes = [n for n in aten.graph.nodes if n.op == 'placeholder']
                with _disable_current_modes():
                    actual = tuple(resolve_descriptor(n.meta['desc'], inputs)[1].detach() for n in nodes)
                    portable = torch.fx.GraphModule(aten, copy.deepcopy(aten.graph))
                if list(portable.named_parameters()) or list(portable.named_buffers()):
                    raise ValueError('Expected fully lifted model weights')
                binding = ParameterBinding(gm, inputs, aten, examples,
                    input_info=torch._guards.TracingContext.get().fw_metadata.input_info)
                named = named_bindings(model, inputs, binding.records)
                with _disable_current_modes():
                    witnesses = [i for i in binding.retained if actual[i].dtype == torch.int64
                                 and tuple(actual[i].shape) == (1, query)
                                 and torch.equal(actual[i], tokens)]
                if not witnesses or not binding.records:
                    raise ValueError('Actual full prompt or model parameter binding missing')
                identity = graph_identity(portable)
                record.update(status='serializing_actual_AOT_graph', actual_AOT_callbacks=count,
                    input_count=len(actual), flat_inputs=[spec(t) for t in actual],
                    parameters=binding.records, native_named_bindings=named,
                    retained_input_indices=binding.retained, prompt_input_indices=witnesses,
                    outputs=[spec(n.meta['val']) for n in list(aten.graph.nodes)[-1].args[0]],
                    operations=dict(Counter(str(n.target) for n in aten.graph.nodes if n.op == 'call_function')),
                    lifted_graph_identity=identity,
                    weight_bytes_not_serialized=sum(row['bytes'] for row in binding.records))
                (args.output/'lifted-aten.py').write_text(portable.code)
                save()
                outer = torch._guards.TracingContext.get()
                with _disable_current_modes(), torch.no_grad(), NoRealModelCompute():
                    exported = torch.export.export(portable, actual, strict=False)
                    if exported.state_dict or exported.constants:
                        raise ValueError('Graph carrier must not include model constants')
                    exported.example_inputs = None
                    if graph_identity(exported.graph_module) != identity:
                        raise ValueError('Export changed the captured native graph')
                    artifact = args.output/'native-prefill.pt2'
                    torch.export.save(exported, artifact)
                if torch._guards.TracingContext.try_get() is not outer:
                    raise ValueError('Export changed parent AOT tracing state')
                if artifact.stat().st_size > 64*1024**2:
                    raise ValueError('Graph carrier unexpectedly exceeds 64 MiB')
                record.update(artifact=dict(path=str(artifact), sha256=sha(artifact), bytes=artifact.stat().st_size),
                              native_samples_removed=True, parent_tracing_context_restored=True)
                save()

                def stop(*values):
                    binding.check(values)
                    raise BeforeModelExecution('Actual native prefill captured; no execution')
                return make_boxed_func(stop)

            def no_backward(*unused, **kwargs):
                raise RuntimeError('Inference capture only')
            return aot_autograd(fw_compiler=compile_aten, inference_compiler=compile_aten,
                               bw_compiler=no_backward, decompositions=table)(gm, inputs)

        def forbid_eager(module, values, kwargs):
            if not torch.compiler.is_compiling():
                raise RuntimeError('CPU eager model forward forbidden')

        record['status'] = 'capturing_native_generate_full_prefill'
        save()
        torch._dynamo.reset()
        handle = model.register_forward_pre_hook(forbid_eager, with_kwargs=True)
        stopped = False
        try:
            # This pinned Transformers release directly calls self() for an
            # unchunked prefill, bypassing its automatic compile_config route.
            # Use PyTorch's native in-place module compilation for BOTH paths;
            # disable only Transformers' separate automatic wrapping.
            model.compile(backend=backend, fullgraph=True, dynamic=False)
            record['compile_entry'] = 'torch.nn.Module.compile'
            options = generation_options(max_new_tokens=1)
            options['prefill_chunk_size'] = None
            with torch.no_grad(), torch._dynamo.config.patch(suppress_errors=False):
                model.generate(tokens, attention_mask=torch.ones_like(tokens), cache_implementation='static', **options)
        except BeforeModelExecution:
            stopped = True
        finally:
            handle.remove()
        if (not stopped or count != 1 or model._cache is not cache or original !=
                (model.forward.__func__, model.generate.__func__, StaticCache.update, StaticCache.reset)):
            raise ValueError('Expected unchanged native framework and pre-execution stop')
        if any(sha(row['path']) != row['sha256'] for row in record['sources']):
            raise ValueError('Capture source changed while running')
        record.update(status='unregistered_length_native_prefill_captured_on_ARM_NOT_DSP',
                      expected_pre_execution_stop=True, original_framework_unchanged=True)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(record['status'], flush=True)


if __name__ == '__main__':
    main()
