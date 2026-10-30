"""Check whether the real dynamic Qwen AOT graph can freeze model parameters.

The probe stops before model execution and before torch-mlir/Hexagon lowering.
It tests a separate prerequisite for a deployable dynamic graph: the lifted
graph currently has hundreds of explicit weight arguments.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback


class StopBeforeExecution(RuntimeError):
    pass


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--context-sha256', required=True)
    parser.add_argument('--generation-source-sha256', required=True)
    parser.add_argument('--query-tokens', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (platform.machine() != 'x86_64' or not 2 <= args.query_tokens <= 32
            or sha(args.context) != args.context_sha256):
        raise ValueError('Only pinned bounded x86 dynamic capture is admitted')
    args.output.mkdir(exist_ok=False)
    record = dict(status='admitting_dynamic_parameter_binding',
        query_tokens=args.query_tokens, dynamic=True, CPU_model_execution=False,
        DSP_calls=0, machine_code_compilations=0, torch_mlir_executed=False,
        source_sha256=sha(__file__), context_sha256=args.context_sha256)

    def save():
        (args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')

    save()
    try:
        import torch
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch_mlir import fx
        from transformers.generation import utils as generation_utils
        from qwen3_generate_compile_common import load_model, generation_options
        from hexagon_native1024_cache_setup import prepare_native1024_cache
        from hexagon_native_silu import preserve_native_silu
        from hexagon_dynamo_parameters import ParameterBinding

        torch.set_num_threads(2)
        if sha(generation_utils.__file__) != args.generation_source_sha256:
            raise ValueError('Pinned native Transformers source changed')
        record['transformers_generation_source_sha256'] = args.generation_source_sha256
        model, _, _ = load_model(args.context)
        cache = prepare_native1024_cache(model)
        original = (model.forward.__func__, model.generate.__func__)
        decompositions = preserve_native_silu(fx.get_decomposition_table())

        def backend(dynamo_graph, dynamo_inputs):
            record['dynamo_input_count'] = len(dynamo_inputs)
            save()

            def compiler(aten, examples):
                started = time.monotonic()
                tracing = torch._guards.TracingContext.get()
                if tracing is None or tracing.fw_metadata is None:
                    raise ValueError('AOT input mutation metadata is missing')
                binding = ParameterBinding(dynamo_graph, dynamo_inputs, aten, examples,
                    input_info=tracing.fw_metadata.input_info)
                before = sum(node.op == 'placeholder' for node in aten.graph.nodes)
                frozen_bytes = sum(row['bytes'] for row in binding.records)
                binding.freeze(aten)
                after = sum(node.op == 'placeholder' for node in aten.graph.nodes)
                retained = [examples[index] for index in binding.retained]
                symbolic = sum(isinstance(value, torch.Tensor)
                    and any(type(dim) is not int for dim in value.shape)
                    for value in retained)
                if (len(binding.records) != 509
                        or before != len(examples)
                        or after != len(binding.retained)
                        or after >= before
                        or frozen_bytes < 1_000_000_000
                        or symbolic < 1):
                    raise ValueError('Dynamic frozen graph parameter/shape contract differs')
                record.update(status='dynamic_qwen_AOT_parameters_frozen_NOT_Linalg_or_DSP',
                    elapsed_binding_seconds=time.monotonic() - started,
                    aot_input_count=before, frozen_parameter_leaves=len(binding.records),
                    frozen_weight_bytes=frozen_bytes,
                    retained_input_count=after,
                    symbolic_retained_tensor_count=symbolic,
                    retained_input_indices=binding.retained,
                    parameter_digest_sha256=hashlib.sha256(json.dumps(
                        binding.records, sort_keys=True).encode()).hexdigest())
                save()

                def stop(*unused):
                    raise StopBeforeExecution('Dynamic parameter binding captured; no model run')

                return make_boxed_func(stop)

            def no_backward(*unused):
                raise RuntimeError('Inference-only diagnostic')

            return aot_autograd(fw_compiler=compiler, inference_compiler=compiler,
                bw_compiler=no_backward, decompositions=decompositions)(dynamo_graph, dynamo_inputs)

        def forbid_eager(module, inputs, kwargs):
            if not torch.compiler.is_compiling():
                raise RuntimeError('CPU eager model execution forbidden')

        hook = model.register_forward_pre_hook(forbid_eager, with_kwargs=True)
        stopped = False
        try:
            tokens = torch.arange(10, 10 + args.query_tokens,
                                  dtype=torch.int64).unsqueeze(0)
            options = generation_options(backend, max_new_tokens=1)
            options['compile_config'].dynamic = True
            options['prefill_chunk_size'] = args.query_tokens
            with torch.no_grad(), torch._dynamo.config.patch(suppress_errors=False):
                model.generate(tokens, attention_mask=torch.ones_like(tokens),
                    cache_implementation='static', **options)
        except StopBeforeExecution:
            stopped = True
        finally:
            hook.remove()
        if (not stopped or model._cache is not cache
                or original != (model.forward.__func__, model.generate.__func__)
                or record['status'] != 'dynamic_qwen_AOT_parameters_frozen_NOT_Linalg_or_DSP'):
            raise ValueError('Expected original dynamic generate capture and pre-execution stop')
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__,
                      error=str(error)[:2000])
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(record['status'], flush=True)


if __name__ == '__main__':
    main()
