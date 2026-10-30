"""Bind a trusted lifted graph to local model tensors by checked native names.

No model forward, source-string eval, value-based weight search, or graph
recapture. Every leaf is checked against producer shape/stride/dtype/bytes/SHA
before taking an immutable snapshot for the existing compiler binding API.
"""
import copy
import math

import torch

from hexagon_dynamo_parameters import spec
from hexagon_dynamic_module_constants import DynamicParameterBinding, dynamic_tensor_digest


class CarrierParameterBinding(DynamicParameterBinding):
    def __init__(self, model, record):
        if not isinstance(model, torch.nn.Module):
            raise ValueError('Native model module required')
        declaration=record.get('guarded_module_constants')
        if declaration is not None:
            from models.qwen3.runtime.support.guarded_module_constants import GuardedModuleConstants,validate_capture_constants
            validate_capture_constants(record)
            self.guarded_constants=GuardedModuleConstants(model,declaration['records'])
            self.cache_constants_signature=self.guarded_constants.signature
        self.input_count = record['graph_identity']['inputs']
        rows = copy.deepcopy(record['immutable_inputs'])
        named = record['named_bindings']
        if (type(self.input_count) is not int or not 1 <= self.input_count <= 1024
                or len(rows) != len(named)):
            raise ValueError('Complete explicit input mapping required')
        mapping = {}
        for item in named:
            index = item['input_index']
            if type(index) is not int or not 0 <= index < self.input_count or index in mapping:
                raise ValueError('Invalid or duplicate immutable input index')
            mapping[index] = item
        values = {}
        for row in rows:
            index = row['input_index']
            if type(index) is not int or index not in mapping or index in values:
                raise ValueError('Immutable input cardinality differs')
            item = mapping[index]
            if 'aliases' in item:
                aliases = item['aliases']
                if not isinstance(aliases, list) or not aliases:
                    raise ValueError('At least one actual native tensor name required')
                bases = []
                for alias in aliases:
                    if alias['kind'] == 'parameter':
                        bases.append(model.get_parameter(alias['name']))
                    elif alias['kind'] == 'buffer':
                        bases.append(model.get_buffer(alias['name']))
                    else:
                        raise ValueError('Native parameter/buffer kind required')
                if any(base is not bases[0] for base in bases):
                    raise ValueError('Native alias identity changed')
                value = bases[0]
                attributes = item['attributes']
                if attributes != row['attributes'] or not isinstance(attributes, list):
                    raise ValueError('Tensor leaf attribute path changed')
                for attribute in attributes:
                    if type(attribute) is not str or not attribute.isidentifier() or attribute.startswith('_'):
                        raise ValueError('Only explicit public tensor leaf attributes allowed')
                    value = getattr(value, attribute)
            else:
                if (row.get('kind') != 'immutable_Qwen3_module_scalar' or row['attributes']
                        or item['attribute'] not in {'variance_epsilon', 'scaling', 'attention_scaling'}
                        or item['dtype'] not in {'torch.float32', 'torch.float64'}):
                    raise ValueError('Explicit immutable module scalar policy required')
                owner = model.get_submodule(item['module'])
                value = owner.__dict__.get(item['attribute'])
                dtype = {'torch.float32': torch.float32, 'torch.float64': torch.float64}[item['dtype']]
                if type(value) is float and math.isfinite(value):
                    value = torch.tensor(value, dtype=dtype)
                elif not isinstance(value, torch.Tensor) or value.ndim != 0 or value.dtype != dtype:
                    raise ValueError('Native scalar attribute type changed')
            if (spec(value) != row['spec'] or value.numel()*value.element_size() != row['bytes']
                    or dynamic_tensor_digest(value) != row['sha256']):
                raise ValueError('Local immutable tensor bytes or metadata differ from producer')
            values[index] = value.detach().clone(memory_format=torch.preserve_format)
        retained = [i for i in range(self.input_count) if i not in values]
        if retained != record['retained_input_indices']:
            raise ValueError('Retained runtime input order changed')
        self.records, self.values, self.retained = rows, values, retained
