"""Exact source and FP64 byte guards for native specialized model floats.

Same fixed Qwen scalar contract as the tensor representation. The ledger never
substitutes values into a graph or changes floating-point arithmetic.
"""
import copy
import hashlib
import json
import math
import struct

SCHEMA='native_guarded_Qwen_python_float_constants_v1'
ATTRIBUTES={'variance_epsilon','scaling','attention_scaling'}


def source_name(module,attribute):
    return "L['self']"+''.join('._modules['+repr(part)+']' for part in module.split('.'))+'.'+attribute


def constant_signature(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def selected_model_content_signature(tensors,constants):
    """Keep the original complete model identity across scalar representations.

    This constructs metadata only, never graph inputs or arithmetic. Python
    floats retain exactly the original rank-zero FP64 source/content identity.
    """
    from hexagon_module_scalar_binding import binding_content_signature
    validate_constant_rows(constants)
    if (len(tensors)!=509 or any(row.get('kind')=='immutable_Qwen3_module_scalar'
                              for row in tensors)):
        raise ValueError('All original tensor bindings required for selected model identity')
    scalars=[dict(source=row['source'],attributes=[],
        spec=dict(shape=[],dtype='torch.float64',stride=[]),bytes=8,
        sha256=row['sha256'],kind='immutable_Qwen3_module_scalar') for row in constants]
    return binding_content_signature(list(tensors)+scalars)


def validate_constant_rows(rows):
    from hexagon_dynamic_module_constants import expected_module_scalar_sources
    expected=expected_module_scalar_sources()
    if type(rows) is not list or len(rows)!=len(expected):
        raise ValueError('Complete original module constant source set required')
    sources=[]
    for row in rows:
        if (type(row) is not dict or set(row)!={'source','module','attribute','hex_le','bytes','sha256'}
                or type(row['module']) is not str or not row['module']
                or any(not (part.isidentifier() or part.isdecimal()) for part in row['module'].split('.'))
                or type(row['attribute']) is not str or row['attribute'] not in ATTRIBUTES
                or type(row['bytes']) is not int or row['bytes']!=8
                or type(row['hex_le']) is not str or len(row['hex_le'])!=16
                or row['source']!=source_name(row['module'],row['attribute'])):
            raise ValueError('Typed original module constant ledger required')
        try:raw=bytes.fromhex(row['hex_le'])
        except ValueError as error:raise ValueError('Exact FP64 constant bytes required') from error
        if (len(raw)!=8 or not math.isfinite(struct.unpack('<d',raw)[0])
                or hashlib.sha256(raw).hexdigest()!=row['sha256']):
            raise ValueError('Finite original FP64 bytes and digest required')
        sources.append(row['source'])
    if set(sources)!=expected or len(set(sources))!=len(sources) or sources!=sorted(sources):
        raise ValueError('Original module constant sources changed or duplicated')


def model_constant_rows(model):
    rows=[]
    for name,module in model.named_modules():
        for attribute in sorted(ATTRIBUTES):
            if attribute not in module.__dict__:continue
            value=module.__dict__[attribute]
            if type(value) is not float or not math.isfinite(value):
                raise ValueError('Original finite native Python model float required')
            raw=struct.pack('<d',value)
            rows.append(dict(source=source_name(name,attribute),module=name,attribute=attribute,
                hex_le=raw.hex(),bytes=8,sha256=hashlib.sha256(raw).hexdigest()))
    rows.sort(key=lambda r:r['source']);validate_constant_rows(rows)
    return rows


class GuardedModuleConstants:
    def __init__(self,model,rows=None):
        actual=model_constant_rows(model)
        if rows is not None:
            validate_constant_rows(rows)
            if rows!=actual:raise ValueError('Compiler-host module constant bytes differ from producer')
        self._records=copy.deepcopy(actual)
        self.signature=constant_signature(actual)
        # Private immutable expected bytes, separate from any exported ledger.
        self._model=model
        self._references=tuple((row['module'],model.get_submodule(row['module']),row['attribute'],
            bytes.fromhex(row['hex_le'])) for row in actual)

    @property
    def records(self):
        return copy.deepcopy(self._records)

    def check(self):
        modules=dict(self._model.named_modules())
        for name,owner,attribute,expected in self._references:
            value=owner.__dict__.get(attribute)
            if (modules.get(name) is not owner or type(value) is not float
                    or not math.isfinite(value) or struct.pack('<d',value)!=expected):
                raise ValueError('Frozen native module constant changed before DSP execution')


def validate_capture_constants(record):
    """Validate a new live convention explicitly; legacy tensor mode stays strict."""
    declaration=record.get('guarded_module_constants')
    if declaration is None:return 651
    from pathlib import Path
    if (type(declaration) is not dict or set(declaration)!={'schema','source_sha256','records','sha256'}
            or declaration['schema']!=SCHEMA
            or declaration['source_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('Pinned native guarded constant convention required')
    validate_constant_rows(declaration['records'])
    if declaration['sha256']!=constant_signature(declaration['records']):
        raise ValueError('Native constant ledger digest differs')
    if (len(record['immutable_inputs'])!=509 or len(record['named_bindings'])!=509
            or any(row.get('kind')=='immutable_Qwen3_module_scalar' for row in record['immutable_inputs'])
            or any('aliases' not in row for row in record['named_bindings'])):
        raise ValueError('Native constant convention must keep all tensor parameter bindings')
    return 509
