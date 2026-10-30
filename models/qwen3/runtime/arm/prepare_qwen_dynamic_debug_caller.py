"""Generate a Qwen diagnostic caller from the pinned 57-output ABI.

The original 57 output fields are preserved. A bounded set of FP32 diagnostics
is appended, including a rank-4 K-normalization tap. This only prepares source
and a buffer contract; it neither
compiles nor executes the model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hexagon_python_source import configure_triton_python
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import (
    generate_buffer_entry, layout_for, make_contract,
)


ENTRY = '_mlir_ciface_Qwen3DynamicBoundProbe'
K_SCALAR_NAMES = frozenset({
    'layer3_k_mean_square',
    'layer3_k_mean_square_plus_epsilon',
    'layer3_k_rsqrt',
})


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def profiles_and_types(old):
    tensor_rows = {row['arg_index']: row for row in old['tensor_inputs']}
    if (len(tensor_rows) != 60 or len(old['tensor_inputs']) != 60
            or old['scalar_inputs'] != [17, 1024, 1]):
        raise ValueError('Pinned 60-tensor, 3-scalar input ABI differs')
    arguments, profiles = [], []
    scalar_index = 0
    for arg in range(63):
        if arg not in tensor_rows:
            if arg not in (0, 4, 62):
                raise ValueError('Unexpected scalar input position')
            profiles.append(SimpleNamespace(input_type='scalar', idx=scalar_index,
                input_id=arg, value=old['scalar_inputs'][scalar_index],
                dtype='int64_t', rank=None, shape=None))
            arguments.append('i64')
            scalar_index += 1
            continue
        row = tensor_rows[arg]
        shape = row['shape']
        dtype = row['dtype']
        if row['tensor_index'] != len(profiles) - scalar_index or dtype not in ('i64', 'f32'):
            raise ValueError('Tensor index/dtype differs')
        strides = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * shape[axis + 1]
        profiles.append(SimpleNamespace(input_type='tensor', idx=row['tensor_index'],
            input_id=arg, value=None, dtype='int64_t' if dtype == 'i64' else 'float',
            rank=len(shape), shape=(str(shape).replace('[', '{').replace(']', '}'),
                                    str(strides).replace('[', '{').replace(']', '}'))))
        symbolic = list(shape)
        for axis in range(len(shape)):
            if (arg, axis) in ((1, 1), (2, 0), (3, 1), (7, 2)) or (
                    5 <= arg <= 61 and arg != 7 and axis == 2):
                symbolic[axis] = '?'
            if (arg, axis) == (7, 3):
                symbolic[axis] = '?'
        arguments.append('tensor<' + 'x'.join(map(str, symbolic)) + 'x' + dtype + '>')
    if len(profiles) != 63 or scalar_index != 3:
        raise ValueError('Dynamic Qwen input arity differs')
    return profiles, arguments


def output_profiles(count, widths=(), names=()):
    if (count not in (57, 58, 59, 60, 63, 67) or len(widths) != count - 57
            or len(names) != count - 57):
        raise ValueError('Only baseline or diagnostic output ABI')
    return [SimpleNamespace(
        idx=i, dtype='float',
        rank=4 if i < 56 or (i >= 57 and
            (names[i - 57] == 'layer3_k_norm_output' or
             names[i - 57] in K_SCALAR_NAMES))
        else 3,
        shape=([1, 8, 1024, 128] if i < 56 else
               [1, 1, 151936] if i == 56 else
               [1, 8, 17, 128] if names[i - 57] == 'layer3_k_norm_output' else
               [1, 17, 8, 1] if names[i - 57] in K_SCALAR_NAMES else
               [1, 17, widths[i - 57]])) for i in range(count)]


def debug_contract(arguments, widths, names=()):
    if names and len(names) != len(widths):
        raise ValueError('Diagnostic names and widths differ')
    names = names or ('',) * len(widths)
    count = len(widths)
    results = ['tensor<1x8x?x128xf32>'] * 56 + [
        'tensor<1x1x151936xf32>'] + [
        'tensor<1x8x?x128xf32>' if name == 'layer3_k_norm_output' else
        'tensor<1x?x8x1xf32>' if name in K_SCALAR_NAMES else
        f'tensor<1x?x{width}xf32>' for name, width in zip(names, widths)]
    symbolic = {('input', 1, 1): 'query', ('input', 2, 0): 'query',
                ('input', 3, 1): 'query', ('input', 7, 2): 'query',
                ('input', 7, 3): 1024}
    symbolic.update({('input', i, 2): 1024 for i in range(5, 62) if i != 7})
    symbolic.update({('output', i, 2): 1024 for i in range(56)})
    symbolic.update({('output', i, 2 if names[i - 57] == 'layer3_k_norm_output'
                     else 1): 'query' for i in range(57, 57 + count)})
    return make_contract(arguments, results, variables={'query': (1, 32)},
                         symbolic_dimensions=symbolic,
                         scalar_bindings={0: 'query', 4: 1024, 62: 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-caller', type=Path, required=True)
    parser.add_argument('--old-caller-sha256', required=True)
    parser.add_argument('--diagnostic-graph-record', type=Path, required=True)
    parser.add_argument('--diagnostic-graph-record-sha256', required=True)
    parser.add_argument('--triton-python', type=Path, required=True)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or not args.remote.startswith('/dev/shm/qwen3-')
            or len(Path(args.remote).parts) != 4):
        raise ValueError('Fresh scoped output and board paths required')
    if (sha(args.old_caller) != args.old_caller_sha256 or
            sha(args.diagnostic_graph_record) != args.diagnostic_graph_record_sha256):
        raise ValueError('Pinned caller/graph record drift')
    old = json.loads(args.old_caller.read_text())
    graph = json.loads(args.diagnostic_graph_record.read_text())
    checkpoints = graph.get('diagnostic_outputs', [])
    widths = graph.get('diagnostic_widths', [])
    count = len(checkpoints)
    expected_ranks = [4 if name == 'layer3_k_norm_output' or
                      name in K_SCALAR_NAMES else 3
                      for name in checkpoints]
    expected_widths = [8 if name in K_SCALAR_NAMES else
                       2048 if name in ('layer2_o_proj_input',
                                       'layer3_o_proj_input',
                                       'layer3_q_proj_output',
                                       'layer27_o_proj_input') else 3072 if name in (
        'layer3_down_proj_input', 'layer4_gate_proj_output',
        'layer4_up_proj_output', 'layer4_down_proj_input')
        else 1024 for name in checkpoints]
    if graph.get('checkpoint_spec_sha256'):
        from select_qwen_layer_observation import checkpoint_description
        expected_widths = [checkpoint_description(name)[2] for name in checkpoints]
        expected_ranks = [3] * count
    if (old['status'] != 'dynamic_q17_file_caller_linked_NOT_DSP_forward'
            or count not in (1, 2, 3, 6, 10) or widths != expected_widths
            or graph.get('diagnostic_ranks', [3] * count) != expected_ranks
            or graph['status'] !=
            f'Qwen_dynamic_{57+count}_output_diagnostic_MLIR_NOT_DSP_or_numeric'
            or old['bounded_softmax'] is not True
            or old['graph_sha256'] != graph['source_sha256']):
        raise ValueError('Diagnostic graph is not based on the accepted old ABI')
    original = Path(old['wrapper']['path'])
    if sha(original) != old['wrapper']['sha256']:
        raise ValueError('Original wrapper drift')
    inputs, arguments = profiles_and_types(old)
    configure_triton_python(args.triton_python)
    from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
        TorchMlirHexagonWrapperGenerator,
    )
    options = {'enableLWP': False, 'enableBufferization': True,
               'enableConvertToHexagonmem': False}
    baseline = TorchMlirHexagonWrapperGenerator(
        inputs, 1, ENTRY, output_profiles(57), options,
        release_returned_buffers=True)
    old_text = original.read_text()
    marker = '#include <cstring>\n'
    if old_text.count(marker) != 1:
        raise ValueError('Original wrapper buffer-entry boundary differs')
    baseline_text = baseline.generate_cpp_wrapper(ENTRY, old['remote'])
    old_prefix = old_text.split(marker, 1)[0]
    if not old_prefix.startswith(baseline_text):
        mismatch = next((i for i, (a, b) in enumerate(zip(baseline_text, old_prefix))
                         if a != b), min(len(baseline_text), len(old_prefix)))
        raise ValueError(f'Original 57-output wrapper differs at {mismatch}: '
                         f'generated={baseline_text[mismatch:mismatch+120]!r} '
                         f'old={old_prefix[mismatch:mismatch+120]!r}')
    wrapper = TorchMlirHexagonWrapperGenerator(
        inputs, 1, ENTRY, output_profiles(57 + count, widths, checkpoints), options,
        release_returned_buffers=True)
    contract = debug_contract(arguments, widths, checkpoints)
    named_runtime = old_prefix[len(baseline_text):]
    if (named_runtime.count('__hexagon_silu_exp_compat') != 2
            or named_runtime.count('__hexagon_softmax_exp_compat') != 2
            or named_runtime.count('#include "qwen3_sleef_exp_candidate.h"') != 1):
        raise ValueError('Named SiLU/Softmax runtime suffix differs')
    source = (wrapper.generate_cpp_wrapper(ENTRY, args.remote) + named_runtime
              + generate_buffer_entry(wrapper, contract,
                                      symbol='hexagon_forward_buffers'))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / f'qwen_dynamic_debug{57+count}_wrapper.cpp'
    path.write_text(source)
    result = dict(status=f'Qwen_dynamic_debug{57+count}_caller_prepared_NOT_linked_or_DSP',
                  source_sha256=sha(__file__), old_caller=str(args.old_caller),
                  old_caller_sha256=args.old_caller_sha256,
                  diagnostic_graph_record_sha256=args.diagnostic_graph_record_sha256,
                  remote=args.remote, wrapper=dict(path=str(path), sha256=sha(path)),
                  original_output_count=57, diagnostic_output_count=count,
                  diagnostic_outputs=checkpoints, diagnostic_widths=widths,
                  diagnostic_ranks=expected_ranks,
                  layouts={str(q): {k: layout_for(contract, {'query': q})[k]
                                    for k in ('input_bytes', 'output_bytes')}
                           for q in (5, 17)}, DSP_calls=0)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
