"""Freeze explicitly allowed scalar attributes using actual module provenance.

No model name, layer count, source-string eval, or value-based input guessing.
Runtime checks remain those of DynamicParameterBinding, including .data writes.
This helper does not freeze arbitrary user scalars or mutable cache state.
"""
import math
import hashlib
import json

import torch

from hexagon_dynamo_parameters import resolve_descriptor, spec
from hexagon_dynamic_module_constants import dynamic_tensor_digest


def binding_content_signature(records):
    """Immutable source/content identity independent of AOT argument numbering."""
    rows = [{key: row[key] for key in ('source', 'attributes', 'spec', 'bytes', 'sha256')}
            | {'kind': row.get('kind', 'parameter')} for row in records]
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def module_attribute_owner(source, roots, allowed_attributes):
    """Resolve only typed Dynamo paths through registered nn.Module children."""
    from torch._dynamo.source import (AttrSource, DictGetItemSource, LocalSource,
                                      UnspecializedNNModuleSource)
    if type(source) is not AttrSource or source.member not in allowed_attributes:
        return None
    def module_path(node):
        if type(node) is LocalSource:
            value = roots.get(node.local_name)
            return value if isinstance(value, torch.nn.Module) else None
        if type(node) is UnspecializedNNModuleSource:
            return module_path(node.base)
        if (type(node) is DictGetItemSource and type(node.index) is str
                and type(node.base) is AttrSource and node.base.member == '_modules'):
            parent = module_path(node.base.base)
            if parent is not None:
                child = parent.__dict__.get('_modules', {}).get(node.index)
                return child if isinstance(child, torch.nn.Module) else None
        # Direct module attributes may be used by simpler Dynamo captures.
        if type(node) is AttrSource:
            parent = module_path(node.base)
            if parent is not None:
                child = parent.__dict__.get('_modules', {}).get(node.member)
                return child if isinstance(child, torch.nn.Module) else None
        return None
    owner = module_path(source.base)
    if owner is None or source.member not in owner.__dict__:
        return None
    return owner, source.member


def bind_module_scalar_attributes(binding, dynamo_graph, dynamo_inputs,
                                  aten_graph, examples, *, input_info, roots,
                                  allowed_attributes,
                                  record_kind='immutable_module_scalar'):
    """Transactionally add only source-proven, unmutated scalar attributes."""
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.utils._python_dispatch import _disable_current_modes
    if (not roots or any(not isinstance(value, torch.nn.Module) for value in roots.values())
            or not allowed_attributes or any(not isinstance(name, str) or not name.isidentifier()
                                             or name.startswith('_') for name in allowed_attributes)):
        raise ValueError('Explicit module roots and public scalar attribute policy required')
    nodes = [node for node in aten_graph.graph.nodes if node.op == 'placeholder']
    dynamo_nodes = [node for node in dynamo_graph.graph.nodes if node.op == 'placeholder']
    if (len(nodes) != binding.input_count or len(examples) != len(nodes)
            or len(input_info) != len(nodes) or len(dynamo_nodes) != len(dynamo_inputs)):
        raise ValueError('Dynamo/AOT scalar binding cardinality differs')
    added, snapshots, seen = [], {}, set()
    for index, (node, example, info) in enumerate(zip(nodes, examples, input_info)):
        if index in binding.values or not isinstance(example, torch.Tensor) or example.ndim != 0:
            continue
        original, value, attrs = resolve_descriptor(node.meta.get('desc'), dynamo_inputs)
        source = getattr(dynamo_nodes[original].meta.get('grapharg'), 'source', None)
        resolved = module_attribute_owner(source, roots, allowed_attributes)
        if resolved is None:
            continue
        owner, attribute = resolved
        actual = owner.__dict__[attribute]
        name = source.name
        if (not isinstance(name, str) or name in seen or attrs
                or info.mutates_data or info.mutates_metadata
                or not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor)
                or value.dtype not in (torch.float32, torch.float64) or value.ndim != 0
                or spec(value) != spec(example)):
            raise ValueError('Allowed module scalar is not an immutable direct floating input')
        with _disable_current_modes():
            if isinstance(actual, torch.Tensor):
                if (actual is not value or not torch.isfinite(actual).all()):
                    raise ValueError('Dynamo scalar tensor is not the original module attribute')
            elif type(actual) is float:
                if (value.dtype != torch.float64 or not math.isfinite(actual)
                        or value.item() != actual):
                    raise ValueError('Tensorified Python module float differs from its source')
            else:
                raise ValueError('Only floating scalar module attributes are admitted')
            snapshot = value.detach().clone(memory_format=torch.preserve_format)
            value_digest = dynamic_tensor_digest(snapshot)
        seen.add(name)
        snapshots[index] = snapshot
        added.append(dict(input_index=index, dynamo_input=original, source=name,
            attributes=[], spec=spec(value), bytes=value.element_size(),
            sha256=value_digest, kind=record_kind))
    binding.values.update(snapshots)
    binding.records.extend(added)
    binding.retained = [index for index in range(binding.input_count) if index not in binding.values]
    return added
