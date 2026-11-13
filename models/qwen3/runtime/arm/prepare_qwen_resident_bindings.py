"""Generate the resident DSP caller ABI from captured input provenance.

The four domains share one KV owner. This generates declarations and dispatch,
not model code or numerical acceptance. A graph admission check must still bind
each runtime capture to the corresponding compiled function.
"""
import argparse
import json
from pathlib import Path
import re


DOMAINS = ('initial_single', 'initial', 'prefill', 'decode')
PLANE = re.compile(r"L\['past_key_values'\]\.layers\[(\d+)\]\.(keys|values)")


def argument(source, mlir_type):
    source = source.replace("L['kwargs'][", "L[")
    scalar = {
        "L['input_ids'].size()[1]": 'query',
        "L['cache_position'].size()[0]": 'query',
        "L['attention_mask'].size()[1]": 'position+query',
        "L['logits_to_keep']": '1',
    }
    dimensions = re.fullmatch(PLANE.pattern + r'\.size\(\)\[2\]', source)
    if source in scalar or dimensions:
        if mlir_type != 'i64' or (dimensions and int(dimensions[1]) >= 28):
            raise ValueError('Unexpected scalar ABI: ' + source)
        return 'int64_t', scalar.get(source, 'position'), None
    tensors = {
        "L['input_ids']": (2, 'ids'),
        "L['cache_position']": (1, 'cache_position'),
        "L['position_ids']": (2, 'positions'),
        "L['attention_mask']": (2, 'mask'),
    }
    if source in tensors:
        rank, value = tensors[source]
        allowed = ('tensor<1xi64>', 'tensor<?xi64>') if rank == 1 else (
            'tensor<1x1xi64>', 'tensor<1x?xi64>')
        if mlir_type not in allowed:
            raise ValueError('Unexpected token/mask ABI: ' + source)
        return f'MemRefDescriptor<int64_t,{rank}>*', '&'+value, None
    plane = PLANE.fullmatch(source)
    if plane and int(plane[1]) < 28 and mlir_type == 'tensor<1x8x?x128xf32>':
        index = 2*int(plane[1]) + (plane[2] == 'values')
        return 'MemRefDescriptor<float,4>*', f'&kv[{index}]', index
    raise ValueError('Unsupported resident input: ' + source)


def generate(spec):
    entries = spec['domains']
    if spec.get('schema') != 1 or set(entries) != set(DOMAINS):
        raise ValueError('Exactly four captured resident domains required')
    declarations, calls, common_planes = [], {}, None
    for name in DOMAINS:
        entry = entries[name]
        function = entry['function']
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', function):
            raise ValueError('Invalid compiled function name')
        arguments, results = entry['signature'].split(') -> (')
        if not arguments.startswith('(') or not results.endswith(')'):
            raise ValueError('Expected captured MLIR function type')
        arguments = arguments[1:].split(', ')
        results = results[:-1].split(', ')
        mapping = entry['mapping']
        if (len(arguments) != len(mapping['inputs']) or len(results) != 57
                or len(mapping['outputs']) != 57
                or results[-1] != 'tensor<1x1x151936xf32>'
                or any(t not in ('tensor<1x8x?x128xf32>', 'tensor<1x8x1x128xf32>')
                       for t in results[:-1])):
            raise ValueError('Captured full-model ABI differs')
        types, values, planes = ['void*'], ['results'], []
        for row, mlir_type in zip(mapping['inputs'], arguments):
            if row['attrs']:
                raise ValueError('Indirect input provenance is unsupported')
            ctype, value, plane = argument(row['source'], mlir_type)
            types.append(ctype)
            values.append(value)
            planes.append(plane)
        resident = name in ('prefill', 'decode')
        actual_planes = [p for p in planes if p is not None]
        if sorted(actual_planes) != (list(range(56)) if resident else []):
            raise ValueError('KV inputs must bind every plane exactly once')
        output_planes = entry['output_planes']
        if sorted(output_planes) != list(range(56)):
            raise ValueError('KV results must bind every plane exactly once')
        if resident:
            derived = [planes[row['past_input']] for row in mapping['outputs'][:-1]]
            if derived != output_planes:
                raise ValueError('KV output order differs from concat provenance')
            types.append('int64_t')
            values.append('capacity')
        if common_planes is None:
            common_planes = output_planes
        elif common_planes != output_planes:
            raise ValueError('Shared caller requires a common output plane order')
        symbol = '_mlir_ciface_' + function
        declarations.append(f'extern "C" void {symbol}('+','.join(types)+');')
        calls[name] = symbol+'('+','.join(values)+');'
    return '\n'.join([
        '// Generated from captured ABI metadata; do not edit.',
        '#pragma once',
        '#include "tensor.h"',
        '#include "returned_memref_owner.h"',
        'struct Results { MemRefDescriptor<float,4> delta[56]; MemRefDescriptor<float,3> logits; };',
        *declarations,
        'static constexpr int output_plane[56]={'+','.join(map(str, common_planes))+'};',
        'static inline void qwen_resident_forward(Results *results, int64_t query,',
        '    int64_t position, int64_t capacity, MemRefDescriptor<int64_t,2> &ids,',
        '    MemRefDescriptor<int64_t,1> &cache_position, MemRefDescriptor<int64_t,2> &positions,',
        '    MemRefDescriptor<int64_t,2> &mask, MemRefDescriptor<float,4> *kv) {',
        '  if (!position && query==1) '+calls['initial_single'],
        '  else if (!position) '+calls['initial'],
        '  else if (query==1) '+calls['decode'],
        '  else '+calls['prefill'],
        '}', '',
        'static inline bool qwen_adopt_results(OwnedReturnedMemrefs<57> &owner,',
        '    Results &results, MemRefDescriptor<float,4> *kv,',
        '    MemRefDescriptor<int64_t,2> &ids, MemRefDescriptor<int64_t,1> &cache_position,',
        '    MemRefDescriptor<int64_t,2> &positions, MemRefDescriptor<int64_t,2> &mask) {',
        '  return owner.adopt({'+','.join(
            [f'results.delta[{i}].allocated' for i in range(56)]
            + ['results.logits.allocated'])+'},',
        '    {'+','.join([f'kv[{i}].allocated' for i in range(56)]
            + ['ids.allocated', 'cache_position.allocated', 'positions.allocated',
               'mask.allocated'])+'});',
        '}', '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = generate(json.loads(args.spec.read_text()))
    with args.output.open('x') as stream:
        stream.write(source)


if __name__ == '__main__':
    main()
