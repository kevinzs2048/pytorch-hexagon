"""Freeze only provenance-checked immutable Qwen module scalar tensor leaves.

Dynamic Dynamo exposes model epsilon/scaling attributes as 142 rank-zero
float64 runtime tensors. They are model-owned arithmetic constants, not user
inputs, but ordinary ParamBufferSource binding does not recognize them. This
extension is intentionally Qwen3-0.6B-specific and rejects any changed scalar
before execution through the existing ParameterBinding.check() path.
"""

import hashlib
import json

import torch

from hexagon_dynamo_parameters import (ParameterBinding, digest as tensor_digest,
                                       resolve_descriptor, spec)


def dynamic_tensor_digest(tensor):
    """Preserve existing tensor hashes, adding the missing rank-zero case."""
    if tensor.ndim != 0:
        return tensor_digest(tensor)
    raw = tensor.detach().reshape(1).contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast('B')).hexdigest()


class DynamicParameterBinding(ParameterBinding):
    """Use the existing frozen-input contract with rank-zero scalar support."""

    def check(self, inputs):
        if len(inputs) != self.input_count:
            raise ValueError('Runtime parameter input arity differs')
        for row in self.records:
            value = inputs[row['input_index']]
            if spec(value) != row['spec'] or dynamic_tensor_digest(value) != row['sha256']:
                raise ValueError('Compiled parameter changed; create a fresh compile session before DSP execution')
        return [inputs[index] for index in self.retained]


def expected_module_scalar_sources():
    prefix = "L['self']._modules['model']"
    names = {prefix + "._modules['rotary_emb'].attention_scaling",
             prefix + "._modules['norm'].variance_epsilon"}
    for index in range(28):
        layer = prefix + f"._modules['layers']._modules['{index}']"
        names.update({
            layer + "._modules['input_layernorm'].variance_epsilon",
            layer + "._modules['post_attention_layernorm'].variance_epsilon",
            layer + "._modules['self_attn'].scaling",
            layer + "._modules['self_attn']._modules['q_norm'].variance_epsilon",
            layer + "._modules['self_attn']._modules['k_norm'].variance_epsilon",
        })
    if len(names) != 142:
        raise ValueError('Qwen3-0.6B scalar-source catalog differs')
    return names


def bind_qwen3_dynamic_module_scalars(binding, dynamo_gm, dynamo_inputs,
                                      aten_gm, examples, *, input_info,
                                      expected_retained_input_count=63):
    """Extend an un-frozen ParameterBinding; no value-based source guessing."""
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.utils._python_dispatch import _disable_current_modes

    if expected_retained_input_count not in (62, 63):
        raise ValueError('Only the admitted q1 or bounded-prefill ABI is supported')
    nodes = [node for node in aten_gm.graph.nodes if node.op == 'placeholder']
    dynamo_nodes = [node for node in dynamo_gm.graph.nodes if node.op == 'placeholder']
    if (len(nodes) != binding.input_count or len(examples) != len(nodes)
            or len(input_info) != len(nodes)
            or len(dynamo_nodes) != len(dynamo_inputs)
            or len(binding.records) != 509 or len(binding.values) != 509):
        raise ValueError('Dynamic Qwen model input binding differs')
    expected = expected_module_scalar_sources()
    added = []
    for index, node in enumerate(nodes):
        if index in binding.values:
            continue
        example = examples[index]
        if (not isinstance(example, torch.Tensor)
                or example.dtype != torch.float64 or example.ndim != 0):
            continue
        original, value, attrs = resolve_descriptor(node.meta.get('desc'), dynamo_inputs)
        source = getattr(dynamo_nodes[original].meta.get('grapharg'), 'source', None)
        name = getattr(source, 'name', None)
        if name not in expected:
            continue
        chain = []
        cursor = source
        while cursor is not None:
            chain.append(type(cursor).__name__)
            cursor = getattr(cursor, 'base', None)
        info = input_info[index]
        if (name in {row['source'] for row in added}
                or not chain or chain[0] != 'AttrSource'
                or chain[-1] != 'LocalSource'
                or any(kind not in ('AttrSource', 'UnspecializedNNModuleSource',
                                     'DictGetItemSource', 'LocalSource') for kind in chain)
                or attrs or info.mutates_data or info.mutates_metadata
                or not isinstance(value, torch.Tensor)
                or isinstance(value, FakeTensor)
                or value.dtype != torch.float64 or value.ndim != 0
                or spec(value) != spec(example)):
            raise ValueError('Qwen scalar is not an immutable direct module tensor attribute')
        with _disable_current_modes():
            snapshot = value.detach().clone(memory_format=torch.preserve_format)
            value_digest = dynamic_tensor_digest(snapshot)
        binding.values[index] = snapshot
        row = dict(input_index=index, dynamo_input=original, source=name,
                   attributes=[], spec=spec(value), bytes=value.element_size(),
                   sha256=value_digest, kind='immutable_Qwen3_module_scalar')
        binding.records.append(row)
        added.append(row)
    if {row['source'] for row in added} != expected:
        raise ValueError('Dynamic Qwen scalar source set is incomplete or changed')
    binding.retained = [index for index in range(binding.input_count)
                        if index not in binding.values]
    if (len(binding.retained) != expected_retained_input_count
            or len(binding.records) != 651):
        raise ValueError('Dynamic Qwen frozen/retained arity differs')
    return dict(status='Qwen3_dynamic_module_scalars_bound', count=len(added),
                bytes=sum(row['bytes'] for row in added),
                source_set_sha256=hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest(),
                records_sha256=hashlib.sha256(json.dumps(added, sort_keys=True).encode()).hexdigest())
