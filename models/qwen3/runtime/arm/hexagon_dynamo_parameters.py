"""Bound actual Dynamo parameter sources to AOT tensor leaves, without heuristics.

Version-sensitive PyTorch integration, static inference only. The framework's
input descriptors identify quantized tensor leaves. Immutable snapshots become
compiler constants. A byte check rejects changed parameters before DSP use;
it deliberately includes .data writes which can bypass the version counter.
"""
import hashlib
import torch


def parameter_sources(gm):
    from torch._dynamo.source import ParamBufferSource, UnspecializedParamBufferSource
    nodes = [n for n in gm.graph.nodes if n.op == 'placeholder']
    result = {}
    for index, node in enumerate(nodes):
        source = getattr(node.meta.get('grapharg'), 'source', None)
        cursor = source
        while cursor is not None:
            if isinstance(cursor, (ParamBufferSource, UnspecializedParamBufferSource)):
                if not isinstance(source.name, str):
                    raise ValueError('Unsupported PyTorch source-name interface')
                result[index] = source.name
                break
            cursor = getattr(cursor, 'base', None)
    return result


def resolve_descriptor(desc, inputs):
    from torch._functorch._aot_autograd.descriptors import PlainAOTInput, SubclassGetAttrAOTInput
    attrs = []
    while isinstance(desc, SubclassGetAttrAOTInput):
        attrs.append(desc.attr)
        desc = desc.base
    if not isinstance(desc, PlainAOTInput) or not 0 <= desc.idx < len(inputs):
        raise ValueError('Unsupported AOT input descriptor; no inferred parameter binding')
    value = inputs[desc.idx]
    for attr in reversed(attrs):
        value = getattr(value, attr)
    return desc.idx, value, list(reversed(attrs))


def spec(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
            or tensor.layout != torch.strided or any(type(n) is not int for n in tensor.shape)):
        raise ValueError('Concrete static CPU tensor required')
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), stride=list(tensor.stride()))


def digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data).cast('B')).hexdigest()


class ParameterBinding:
    def __init__(self, dynamo_gm, dynamo_inputs, aten_gm, examples, *, input_info,
                 retain_snapshots=True):
        from torch._subclasses.fake_tensor import FakeTensor
        sources = parameter_sources(dynamo_gm)
        nodes = [n for n in aten_gm.graph.nodes if n.op == 'placeholder']
        if (len(nodes) != len(examples) or len(input_info) != len(examples)
                or list(dynamo_gm.named_parameters()) or list(dynamo_gm.named_buffers())):
            raise ValueError('Only direct Dynamo inputs with unambiguous AOT metadata are supported')
        self.records, self.values = [], {}
        for index, (node, example) in enumerate(zip(nodes, examples)):
            original, value, attrs = resolve_descriptor(node.meta.get('desc'), dynamo_inputs)
            if original not in sources:
                continue
            info = input_info[index]
            if info.mutates_data or info.mutates_metadata:
                raise ValueError('Mutated parameters/buffers cannot be frozen')
            if isinstance(value, FakeTensor) or spec(value) != spec(example):
                raise ValueError('Concrete parameter leaf differs from captured AOT input')
            # Compilation needs immutable snapshots, but a metadata-only
            # diagnostic may hash the concrete values without another copy.
            # Neither mode specializes user input values or KV as weights.
            from torch.utils._python_dispatch import _disable_current_modes
            with _disable_current_modes():
                snapshot = (value.detach().clone(memory_format=torch.preserve_format)
                            if retain_snapshots else None)
                sha256 = digest(snapshot if snapshot is not None else value)
            self.values[index] = snapshot
            self.records.append(dict(input_index=index, dynamo_input=original, source=sources[original],
                attributes=attrs, spec=spec(value), bytes=value.numel()*value.element_size(), sha256=sha256))
        self.retained = [i for i in range(len(nodes)) if i not in self.values]
        self.input_count = len(nodes)

    def freeze(self, gm):
        from torch._inductor.constant_folding import replace_node_with_constant
        if any(value is None for value in self.values.values()):
            raise ValueError('Cannot freeze metadata-only parameter binding')
        nodes = [n for n in gm.graph.nodes if n.op == 'placeholder']
        if len(nodes) != self.input_count:
            raise ValueError('AOT input arity drift')
        for index, tensor in self.values.items():
            replace_node_with_constant(gm, nodes[index], tensor)
        gm.graph.lint()
        gm.recompile()
        return gm

    def check(self, inputs):
        if len(inputs) != self.input_count:
            raise ValueError('Runtime parameter input arity differs')
        for row in self.records:
            value = inputs[row['input_index']]
            if spec(value) != row['spec'] or digest(value) != row['sha256']:
                raise ValueError('Compiled parameter changed; create a fresh compile session before DSP execution')
        return [inputs[i] for i in self.retained]
