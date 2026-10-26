"""Build the actual captured q1 C ABI, keeping the existing packet framing.

Dynamo's specialized graph has no leading query scalar. The singleton query
header is checked by the buffer entry but is NOT passed as an extra C argument.
The 60 tensor payloads remain byte-compatible with the dynamic q=1 packet.
"""
import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from hexagon_compiled_artifact_cache import canonical, digest_file
from hexagon_python_source import configure_triton_python
from prepare_qwen_dynamic_debug_caller import ENTRY, output_profiles, profiles_and_types
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import generate_buffer_entry, layout_for
from hexagon_qwen_shape_policy import contract_query_bounds


def captured_profiles(contract, runtime=None):
    if runtime is not None and len(contract['inputs']) != len(runtime):
        raise ValueError('Captured runtime/contract cardinality differs')
    profiles, tensor_index, scalar_index = [], 0, 0
    for index, row in enumerate(contract['inputs']):
        observed = None if runtime is None else runtime[index]
        if observed is not None and row['kind'] != observed['kind']:
            raise ValueError('Captured runtime argument kind differs')
        if row['kind'] == 'scalar':
            if type(row['value']) is not int or (observed is not None and observed['value'] != row['value']):
                raise ValueError('Specialized scalar value differs')
            profiles.append(SimpleNamespace(input_type='scalar', idx=scalar_index,
                input_id=index, value=row['value'], dtype='int64_t', rank=None, shape=None))
            scalar_index += 1
            continue
        shape = row['shape']
        dtype = {'float32': ('torch.float32', 'float'), 'int64': ('torch.int64', 'int64_t')}[row['dtype']]
        if (not shape or any(type(dim) is not int or dim <= 0 for dim in shape)
                or (observed is not None and (observed['shape'] != shape or observed['dtype'] != dtype[0]))):
            raise ValueError('Specialized runtime tensor shape/dtype differs')
        strides = [1]*len(shape)
        for axis in range(len(shape)-2, -1, -1):
            strides[axis] = strides[axis+1]*shape[axis+1]
        literal = lambda values: '{'+', '.join(map(str, values))+'}'
        profiles.append(SimpleNamespace(input_type='tensor', idx=tensor_index,
            input_id=index, value=None, dtype=dtype[1], rank=len(shape),
            shape=(literal(shape), literal(strides))))
        tensor_index += 1
    return profiles


def bounded_profiles(contract):
    """C declaration profiles; the RPC entry takes real sizes from the contract.

    Concrete initializers here belong to the unused standalone main(). They
    do not specialize compiled IR or the generated dynamic buffer entry.
    """
    sizes = {name: bounds[0] for name, bounds in contract['variables'].items()}
    rows = []
    for row in contract['inputs']:
        if row['kind'] == 'scalar':
            rows.append(dict(row, value=sizes.get(row['value'], row['value'])))
        else:
            rows.append(dict(row, shape=[sizes.get(dim, dim) for dim in row['shape']]))
    return captured_profiles(dict(inputs=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'prepared-graph', 'reference'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--triton-python', type=Path, required=True)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for name in ('capture', 'prepared_graph', 'reference'):
        if digest_file(getattr(args, name)) != getattr(args, name+'_sha256'):
            raise ValueError('Static caller input changed: '+name)
        records[name] = json.loads(getattr(args, name).read_text())
    capture, prepared, reference = [records[name] for name in ('capture', 'prepared_graph', 'reference')]
    contract = capture['buffer_contract']
    if (args.output.exists() or Path(args.remote).parent != Path('/dev/shm')
            or not Path(args.remote).name.startswith('qwen3-')
            or capture['status'] != 'native_first_forward_captured_and_lowered_NOT_executed'
            or not capture['stopped_before_execution'] or not capture['native_cache_unchanged']
            or len(capture['prompt_ids']) != 1 or capture['retained_inputs'] != 62
            or capture['CPU_model_execution'] or capture['DSP_calls'] != 0
            or prepared['status'] != 'actual_captured_Qwen_prepared_for_Hexagon_NOT_machine_code'
            or prepared['frontend_key'] != capture['frontend']['key']
            or prepared['source']['sha256'] != capture['frontend']['artifacts'][0]['sha256']
            or digest_file(prepared['output']['path']) != prepared['output']['sha256']
            or contract['variables'] != {'query': [1, 1]}
            or len(contract['inputs']) != 62 or len(contract['outputs']) != 57
            or digest_file(reference['wrapper']['path']) != reference['wrapper']['sha256']):
        raise ValueError('Actual captured q1 graph, runtime ABI and original caller required')
    layout = layout_for(contract, {'query': 1})
    if (capture['runtime_packet']['bytes'] != 16+layout['input_bytes']
            or capture['runtime_packet']['output_bytes'] != layout['output_bytes']
            or digest_file(capture['runtime_packet']['path']) != capture['runtime_packet']['sha256']):
        raise ValueError('Actual first-q1 packet differs from derived wrapper layout')
    write_caller(contract, reference, args.output, args.triton_python, args.remote,
        runtime_inputs=capture['runtime_inputs'], provenance=dict(
            capture_sha256=args.capture_sha256,
            prepared_graph_sha256=args.prepared_graph_sha256,
            graph_sha256=prepared['output']['sha256'], reference_sha256=args.reference_sha256,
            frontend_key=capture['frontend']['key']))


def write_caller(contract, reference, output, triton_python, remote, *, provenance, runtime_inputs=None):
    """Generate from the captured contract; real inputs are checked at invocation."""
    bounds = contract_query_bounds(contract)
    q1 = bounds == (1, 1)
    fixed = bounds[0] == bounds[1]
    tensor_count = sum(row['kind'] == 'tensor' for row in contract['inputs'])
    scalar_count = len(contract['inputs'])-tensor_count
    if (output.exists()
            or tensor_count != 60 or not 2 <= scalar_count <= 8 or len(contract['outputs']) != 57
            or digest_file(reference['wrapper']['path']) != reference['wrapper']['sha256']):
        raise ValueError('Selected bounded Qwen contract and original caller required')
    layout = layout_for(contract, {'query': contract['variables']['query'][1]})
    profiles = bounded_profiles(contract) if runtime_inputs is None else captured_profiles(contract, runtime_inputs)
    outputs = []
    for index, row in enumerate(contract['outputs']):
        if (row['kind'] != 'tensor' or row['dtype'] != 'float32'
                or any(type(dim) is not int or dim <= 0 for dim in row['shape'])):
            raise ValueError('Concrete FP32 specialized output required')
        outputs.append(SimpleNamespace(idx=index, dtype='float', rank=len(row['shape']), shape=row['shape']))
    configure_triton_python(str(triton_python))
    from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMlirHexagonWrapperGenerator
    options = dict(enableLWP=False, enableBufferization=True, enableConvertToHexagonmem=False)
    original_profiles, _ = profiles_and_types(reference)
    baseline = TorchMlirHexagonWrapperGenerator(original_profiles, 1, ENTRY, output_profiles(57),
                                                options, release_returned_buffers=True)
    old_text = Path(reference['wrapper']['path']).read_text()
    marker = '#include <cstring>\n'
    if old_text.count(marker) != 1:
        raise ValueError('Original wrapper boundary differs')
    old_prefix = old_text.split(marker, 1)[0]
    original_source = baseline.generate_cpp_wrapper(ENTRY, reference['remote'])
    if not old_prefix.startswith(original_source):
        raise ValueError('Original wrapper generator is not reproducible')
    runtime = old_prefix[len(original_source):]
    if (runtime.count('__hexagon_silu_exp_compat') != 2
            or runtime.count('__hexagon_softmax_exp_compat') != 2):
        raise ValueError('Existing numerical runtime suffix differs')
    wrapper = TorchMlirHexagonWrapperGenerator(profiles, 1, ENTRY, outputs, options,
                                               release_returned_buffers=True)
    source = wrapper.generate_cpp_wrapper(ENTRY, remote)+runtime+generate_buffer_entry(
        wrapper, contract, symbol='hexagon_forward_buffers')
    output.mkdir()
    target = output/'qwen_captured_q1_wrapper.cpp'
    target.write_text(source)
    header = Path(reference['wrapper']['path']).parent/'qwen3_sleef_exp_candidate.h'
    if digest_file(header) != reference['silu_runtime']['header_copy_sha256']:
        raise ValueError('Existing numerical runtime header differs')
    copied = output/header.name
    shutil.copyfile(header, copied)
    result = dict(status=('captured_q1_buffer_caller_prepared_NOT_linked_or_DSP' if q1 else
                         'captured_bounded_buffer_caller_prepared_NOT_linked_or_DSP'),
        source_sha256=digest_file(__file__), **provenance,
        wrapper=dict(path=str(target), sha256=digest_file(target)),
        runtime_header=dict(path=str(copied), sha256=digest_file(copied)),
        contract=contract, input_arguments=len(profiles), tensor_inputs=tensor_count, scalar_inputs=scalar_count,
        output_count=len(outputs), input_bytes=16+layout['input_bytes'], output_bytes=layout['output_bytes'],
        remote=remote,
        query_specialization=bounds[0] if fixed else None, DSP_calls=0, numerical_acceptance=False)
    (output/'result.json').write_bytes(canonical(result)+b'\n')
    print(result['status'])


if __name__ == '__main__':
    main()
