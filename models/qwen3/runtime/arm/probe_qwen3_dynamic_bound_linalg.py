"""Probe torch-mlir lowering after freezing the real dynamic Qwen weights.

Optionally serialize verified Linalg bytecode. Hexagon compilation, DSP
execution, and model numerical acceptance are separate stages.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback

from probe_qwen3_dynamic_weight_binding import StopBeforeExecution, sha


class BoundedBytecodeSink:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.bytes = stream, limit, 0

    def write(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('Binary MLIR bytecode required')
        if self.bytes + len(data) > self.limit:
            raise ValueError('Dynamic bound Linalg exceeded scoped bytecode cap')
        written = self.stream.write(data)
        self.bytes += written
        return written

    def flush(self):
        self.stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--context-sha256', required=True)
    parser.add_argument('--generation-source-sha256', required=True)
    parser.add_argument('--binding-result', type=Path, required=True)
    parser.add_argument('--binding-result-sha256', required=True)
    parser.add_argument('--emit-mlir', action='store_true',
                        help='Serialize verified bound Linalg with a 1.5 GiB cap')
    parser.add_argument('--bytecode-version', type=int, choices=range(7),
                        help='MLIR bytecode format version; defaults to producer version')
    parser.add_argument('--capture-runtime-abi', action='store_true',
                        help='Record concrete AOT call inputs, then stop before model execution')
    parser.add_argument('--runtime-input-output', type=Path,
                        help='Write exact retained tensor inputs into a fresh scoped directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.bytecode_version is not None and not args.emit_mlir:
        raise ValueError('--bytecode-version requires --emit-mlir')
    if bool(args.runtime_input_output) != args.capture_runtime_abi:
        raise ValueError('--runtime-input-output requires --capture-runtime-abi and vice versa')
    if (platform.machine() != 'x86_64'
            or sha(args.context) != args.context_sha256
            or sha(args.binding_result) != args.binding_result_sha256):
        raise ValueError('Only pinned bounded x86 dynamic capture is admitted')
    bound = json.loads(args.binding_result.read_text())
    if (bound['status'] != 'dynamic_qwen_AOT_parameters_frozen_NOT_Linalg_or_DSP'
            or bound['query_tokens'] != 17
            or bound['context_sha256'] != args.context_sha256
            or bound['transformers_generation_source_sha256']
                != args.generation_source_sha256):
        raise ValueError('Actual dynamic q17 binding control differs')
    args.output.mkdir(exist_ok=False)
    if args.emit_mlir and shutil.disk_usage(args.output).free < 3 * 1024**3:
        raise ValueError('At least 3 GiB host free space required for bound MLIR')
    if args.runtime_input_output is not None:
        if args.runtime_input_output.exists() or shutil.disk_usage(
                args.runtime_input_output.parent).free < 512 * 1024**2:
            raise ValueError('Fresh runtime input directory and 512 MiB space required')
    record = dict(status='admitting_dynamic_bound_Linalg', query_tokens=17,
        CPU_model_execution=False, DSP_calls=0, machine_code_compilations=0,
        source_sha256=sha(__file__), context_sha256=args.context_sha256,
        binding_result_sha256=args.binding_result_sha256)

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
        import hexagon_dynamic_module_constants as dynamic_binding_module
        import hexagon_tensor_literal_buffers as owned_constants_module
        from hexagon_dynamic_module_constants import (DynamicParameterBinding,
            bind_qwen3_dynamic_module_scalars)
        from hexagon_tensor_literal_buffers import lower_aten_graph_owned_constants

        torch.set_num_threads(2)
        if sha(generation_utils.__file__) != args.generation_source_sha256:
            raise ValueError('Pinned native Transformers source changed')
        record['dynamic_binding_source_sha256'] = sha(dynamic_binding_module.__file__)
        record['owned_constants_source_sha256'] = sha(owned_constants_module.__file__)
        save()
        model, _, _ = load_model(args.context)
        cache = prepare_native1024_cache(model)
        original = (model.forward.__func__, model.generate.__func__)
        decompositions = preserve_native_silu(fx.get_decomposition_table())

        def backend(dynamo_graph, dynamo_inputs):
            def compiler(aten, examples):
                tracing = torch._guards.TracingContext.get()
                if tracing is None or tracing.fw_metadata is None:
                    raise ValueError('AOT metadata for immutable weights is missing')
                binding = DynamicParameterBinding(dynamo_graph, dynamo_inputs, aten, examples,
                    input_info=tracing.fw_metadata.input_info)
                if (len(binding.records) != bound['frozen_parameter_leaves']
                        or binding.retained != bound['retained_input_indices']
                        or sum(row['bytes'] for row in binding.records)
                            != bound['frozen_weight_bytes']
                        or hashlib.sha256(json.dumps(binding.records,
                            sort_keys=True).encode()).hexdigest()
                            != bound['parameter_digest_sha256']):
                    raise ValueError('Frozen dynamic model parameters changed')
                scalar_summary = bind_qwen3_dynamic_module_scalars(binding,
                    dynamo_graph, dynamo_inputs, aten, examples,
                    input_info=tracing.fw_metadata.input_info)
                binding.freeze(aten)
                record.update(status='lowering_frozen_dynamic_AOT_to_Linalg',
                    retained_inputs=len(binding.retained),
                    frozen_parameter_leaves=len(binding.records),
                    scalar_binding=scalar_summary)
                save()
                started = time.monotonic()
                module, resources = lower_aten_graph_owned_constants(
                    aten, 'Qwen3DynamicBoundProbe')
                functions = [op for op in module.body.operations
                             if 'function_type' in op.attributes]
                signatures = [str(op.attributes['function_type']) for op in functions]
                verified = bool(module.operation.verify())
                resource_bytes = sum(row['bytes'] for row in resources)
                if (not verified or len(signatures) != 1 or '?' not in signatures[0]
                        or resource_bytes < 1_000_000_000
                        or any(not row['independent_readonly_snapshot'] for row in resources)):
                    raise ValueError('Frozen symbolic Qwen Linalg signature differs')
                record.update(status='dynamic_q17_frozen_weights_lowered_to_Linalg_NOT_Hexagon_or_DSP',
                    elapsed_linalg_seconds=time.monotonic() - started,
                    verified=verified, function_count=len(signatures),
                    function_type_sha256=hashlib.sha256(signatures[0].encode()).hexdigest(),
                    function_type_prefix=signatures[0][:480],
                    symbolic_dimension_count=signatures[0].count('?'),
                    owned_constant_resources=len(resources),
                    owned_constant_bytes=resource_bytes,
                    module_serialized=False)
                if args.emit_mlir:
                    target = args.output / 'dynamic-bound-linalg.mlirbc'
                    with target.open('xb') as raw:
                        sink = BoundedBytecodeSink(raw, 1536 * 1024**2)
                        module.operation.write_bytecode(
                            sink, desired_version=args.bytecode_version)
                        sink.flush()
                    with target.open('rb') as stream:
                        magic = stream.read(4)
                    if (magic != b'ML\xefR'
                            or not 1_000_000 < target.stat().st_size <= 1536 * 1024**2):
                        raise ValueError('Serialized dynamic bound Linalg size differs')
                    record.update(status='dynamic_q17_frozen_weights_Linalg_bytecode_NOT_Hexagon_or_DSP',
                        module_serialized=True,
                        linalg=dict(path=str(target), sha256=sha(target),
                                    bytes=target.stat().st_size, format='MLIR_bytecode',
                                    version=args.bytecode_version))
                del module
                save()

                def stop(*values):
                    if args.capture_runtime_abi:
                        from torch._subclasses.fake_tensor import FakeTensor
                        runtime = binding.check(values)
                        if len(runtime) != 63:
                            raise ValueError('Bound dynamic runtime input arity differs')
                        args.runtime_input_output.mkdir(exist_ok=False)
                        profiles = []
                        for index, value in enumerate(runtime):
                            if isinstance(value, torch.Tensor):
                                if (isinstance(value, FakeTensor) or value.device.type != 'cpu'
                                        or any(type(dim) is not int for dim in value.shape)):
                                    raise ValueError('Concrete CPU runtime tensor required')
                                packed = value.detach().contiguous()
                                raw = packed.view(torch.uint8).numpy()
                                target = args.runtime_input_output / f'input-{index:02d}.bin'
                                with target.open('xb') as stream:
                                    stream.write(memoryview(raw).cast('B'))
                                profiles.append(dict(index=index, kind='tensor',
                                    dtype=str(value.dtype), shape=list(value.shape),
                                    stride=list(value.stride()), bytes=target.stat().st_size,
                                    sha256=sha(target), path=str(target)))
                            elif type(value) is int or isinstance(value, torch.SymInt):
                                profiles.append(dict(index=index, kind='scalar_i64',
                                    source_type=type(value).__name__, value=int(value)))
                            else:
                                raise ValueError('Unexpected concrete AOT runtime input type')
                        if (sum(row['kind'] == 'tensor' for row in profiles) != 60
                                or sum(row['kind'] == 'scalar_i64' for row in profiles) != 3):
                            raise ValueError('Bound dynamic concrete ABI differs')
                        record.update(status='dynamic_q17_concrete_AOT_runtime_ABI_captured_NOT_DSP',
                            runtime_inputs=profiles, runtime_input_output=str(args.runtime_input_output),
                            runtime_input_bytes=sum(row.get('bytes', 0) for row in profiles))
                        save()
                    raise StopBeforeExecution('Frozen dynamic Linalg verified; no model run')

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
            tokens = torch.arange(10, 27, dtype=torch.int64).unsqueeze(0)
            options = generation_options(backend, max_new_tokens=1)
            options['compile_config'].dynamic = True
            options['prefill_chunk_size'] = 17
            with torch.no_grad(), torch._dynamo.config.patch(suppress_errors=False):
                model.generate(tokens, attention_mask=torch.ones_like(tokens),
                    cache_implementation='static', **options)
        except StopBeforeExecution:
            stopped = True
        finally:
            hook.remove()
        if (not stopped or model._cache is not cache
                or original != (model.forward.__func__, model.generate.__func__)
                or record['status'] != ('dynamic_q17_concrete_AOT_runtime_ABI_captured_NOT_DSP'
                    if args.capture_runtime_abi else
                    'dynamic_q17_frozen_weights_Linalg_bytecode_NOT_Hexagon_or_DSP'
                    if args.emit_mlir else
                    'dynamic_q17_frozen_weights_lowered_to_Linalg_NOT_Hexagon_or_DSP')):
            raise ValueError('Original model or expected compiler stop differs')
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
