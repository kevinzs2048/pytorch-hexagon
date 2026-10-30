"""Select norm/projection observation points from the actual printed dataflow.

No query length or generated SSA number is baked in. The elided inspection
text is ONLY used for selectors; it must never become a compilation input.
The subsequent instrumentation validates selectors against the full graph.
"""
import argparse
import json
from pathlib import Path
import re

from hexagon_runtime.hexagon_compiled_artifact_cache import canonical, digest_file


DEF = re.compile(r'^    (%[A-Za-z0-9_]+)(?::[0-9]+)? = ([A-Za-z0-9_.]+) ')
VALUE = re.compile(r'%[A-Za-z0-9_]+')
SHAPE = 'tensor<1x?x1024xf32>'


def checkpoint_description(name):
    match = re.fullmatch(r'layer([0-9]+)_(input_norm_input|input_norm_output|k_proj_output|attention_input|attention_residual|o_proj_input|o_proj_output|down_proj_input|down_proj_output)', name)
    if match is None or not 0 <= int(match[1]) < 28:
        raise ValueError('Unsupported Qwen dataflow checkpoint')
    layer, suffix = match.groups()
    base = f'model.layers.{layer}.'
    if suffix == 'attention_residual':
        return base+'post_attention_layernorm', 'input', 1024
    if suffix == 'attention_input':
        return base+'input_layernorm', 'input', 1024
    if suffix.startswith('o_proj_'):
        role = suffix.rsplit('_', 1)[1]
        return base+'self_attn.o_proj', role, 2048 if role == 'input' else 1024
    if suffix.startswith('down_proj_'):
        role = suffix.rsplit('_', 1)[1]
        return base+'mlp.down_proj', role, 3072 if role == 'input' else 1024
    if suffix == 'k_proj_output':
        return base+'self_attn.k_proj', 'output', 1024
    return base+'input_layernorm', suffix.rsplit('_', 1)[1], 1024


def select(text, layer, kind='input_norm'):
    if type(layer) is not int or not 0 <= layer < 28:
        raise ValueError('Qwen layer must be 0..27')
    if kind == 'layer_transition':
        # Both sides of one decoder boundary, selected independently from
        # the same graph. No generated SSA names or query length assumptions.
        return select(text, layer, 'previous_mlp') + select(text, layer, 'input_norm')
    if kind not in ('input_norm', 'previous_mlp', 'previous_attention') or (kind != 'input_norm' and layer == 0):
        raise ValueError('Invalid observation kind or missing previous layer')
    nodes = {}
    current = None
    for line in text.splitlines():
        match = DEF.match(line)
        if match:
            value, operation = match.groups()
            if value in nodes:
                raise ValueError('Duplicate top-level SSA definition')
            current = dict(value=value, operation=operation, header=line,
                           order=len(nodes), lines=[line])
            nodes[value] = current
        elif current is not None:
            if line.startswith('    return ') or line == '  }':
                current = None
            else:
                current['lines'].append(line)
    for node in nodes.values():
        operands = set(VALUE.findall('\n'.join(node['lines'])))
        node['parents'] = [nodes[value] for value in operands
                           if value in nodes and nodes[value]['order'] < node['order']]

    def ancestors(node):
        found = {}
        pending = list(node['parents'])
        while pending:
            parent = pending.pop()
            if parent['value'] in found:
                continue
            found[parent['value']] = parent
            pending.extend(parent['parents'])
        return list(found.values())

    def output_type(node):
        types = re.findall(r'tensor<[^>]+>', node['header'])
        return types[-1] if types else None

    def nearest_hidden(node):
        candidates = [parent for parent in ancestors(node) if output_type(parent) == SHAPE]
        if not candidates:
            raise ValueError('Missing hidden-state ancestor')
        return max(candidates, key=lambda parent: parent['order'])

    scatters = [node for node in nodes.values() if node['operation'] == 'tm_tensor.scatter']
    if len(scatters) != 56:
        raise ValueError('Expected 56 KV writes in this Qwen graph')
    key = scatters[2*layer]
    expected_arg = 5 if layer == 0 else 6 + 2*layer
    if f'outs(%arg{expected_arg} :' not in key['header']:
        raise ValueError('Layer K-cache argument order differs')
    projection = nearest_hidden(key)
    if projection['operation'] != 'tensor.expand_shape':
        raise ValueError('Expected projected K before head reshape')
    def sole_hidden_input(node, *, multiplication=True, width=1024):
        match = re.search(r'ins\((.*?) : ', node['header'])
        if (match is None or node['operation'] != 'linalg.generic'
                or (multiplication and '\n'.join(node['lines']).count('arith.mulf ') != 1)):
            raise ValueError('Expected the native RMSNorm multiplication chain')
        candidates = [nodes[value] for value in VALUE.findall(match.group(1))
                      if value in nodes and output_type(nodes[value]) == f'tensor<1x?x{width}xf32>']
        if len(candidates) != 1:
            raise ValueError('Ambiguous native RMSNorm hidden input')
        return candidates[0]

    def projection_input(node, width):
        rounds = [parent for parent in ancestors(node)
                  if output_type(parent) == f'tensor<1x?x{width}xf32>'
                  and '\n'.join(parent['lines']).count('math.roundeven ') == 1]
        if not rounds:
            raise ValueError('Expected native dynamic activation quantizer')
        rounded = max(rounds, key=lambda parent: parent['order'])
        scaled = sole_hidden_input(rounded, multiplication=False, width=width)
        return sole_hidden_input(scaled, width=width)

    normalized = projection_input(projection, 1024)
    unit_normalized = sole_hidden_input(normalized)
    residual = sole_hidden_input(unit_normalized)
    points = [('input_norm_input', residual, 1024),
              ('input_norm_output', normalized, 1024), ('k_proj_output', projection, 1024)]
    if kind != 'input_norm':
        marker = re.search(r'ins\((.*?) : ', residual['header'])
        if (marker is None or residual['operation'] != 'linalg.generic'
                or '\n'.join(residual['lines']).count('arith.addf ') != 1):
            raise ValueError('Expected previous MLP residual addition')
        parents = [nodes[value] for value in VALUE.findall(marker.group(1)) if value in nodes]
        if len(parents) != 2 or any(output_type(node) != SHAPE for node in parents):
            raise ValueError('Expected two hidden-state residual operands')
        down = [node for node in parents if node['operation'] == 'tensor.expand_shape']
        if len(down) != 1:
            raise ValueError('Ambiguous previous down projection')
        down = down[0]
        attention = next(node for node in parents if node is not down)
        down_input = projection_input(down, 3072)
        # The down-projection activation must descend from this residual branch.
        if attention['value'] not in {node['value'] for node in ancestors(down_input)}:
            raise ValueError('MLP activation does not descend from residual branch')
        points = [('attention_residual', attention, 1024),
                  ('down_proj_input', down_input, 3072), ('down_proj_output', down, 1024)]
        if kind == 'previous_attention':
            marker = re.search(r'ins\((.*?) : ', attention['header'])
            if (marker is None or attention['operation'] != 'linalg.generic'
                    or '\n'.join(attention['lines']).count('arith.addf ') != 1):
                raise ValueError('Expected attention residual addition')
            parents = [nodes[value] for value in VALUE.findall(marker.group(1)) if value in nodes]
            projected = [node for node in parents if node['operation'] == 'tensor.expand_shape']
            if (len(parents) != 2 or len(projected) != 1
                    or any(output_type(node) != SHAPE for node in parents)):
                raise ValueError('Ambiguous attention output projection')
            projected = projected[0]
            before = next(node for node in parents if node is not projected)
            pv = projection_input(projected, 2048)
            if before['value'] not in {node['value'] for node in ancestors(pv)}:
                raise ValueError('Attention output does not descend from residual branch')
            points = [('attention_input', before, 1024),
                      ('o_proj_input', pv, 2048), ('o_proj_output', projected, 1024)]
        layer -= 1
    rows = []
    for suffix, node, width in points:
        if node['operation'] == 'linalg.generic':
            marker = re.search(r'ins\(.*? : ', node['header'])
            if marker is None:
                raise ValueError('Missing selected generic inputs')
            source = marker.group()
        elif node['operation'] == 'tensor.expand_shape':
            source = node['operation'] + ' ' + VALUE.findall(node['header'])[1] + ' '
        elif node['operation'] == 'tensor.reshape':
            source = node['operation'] + ' ' + VALUE.findall(node['header'])[1] + '('
        else:
            raise ValueError('Unsupported selected checkpoint operation')
        rows.append(dict(name=f'layer{layer}_'+suffix, value=node['value'],
                         source=source, width=width, operation=node['operation']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--inspection-sha256', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--kind', choices=('input_norm', 'previous_mlp', 'previous_attention', 'layer_transition'), default='input_norm')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if digest_file(args.inspection) != args.inspection_sha256:
        raise ValueError('Printed inspection source changed')
    result = dict(status='dataflow_selected_observation_points_NOT_model_input',
                  inspection_sha256=args.inspection_sha256, layer=args.layer,
                  source_sha256=digest_file(__file__),
                  kind=args.kind, checkpoints=select(args.inspection.read_text(), args.layer, args.kind))
    with args.output.open('xb') as stream:
        stream.write(canonical(result)+b'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
