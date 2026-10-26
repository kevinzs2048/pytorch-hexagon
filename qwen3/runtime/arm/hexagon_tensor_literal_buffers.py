"""Per-import owned tensor buffers, avoiding large tensor.tolist() allocations.

Uses FxImporterHooks; does not patch torch-mlir globals or enable a DSP backend.
The importer/caller still decides which values are constants. Runtime parameter
change checks remain required. No dtype conversion or reduced precision here.
"""
import hashlib
import sys

import numpy as np
import torch
from torch_mlir import fx, ir
from torch_mlir.compiler_utils import OutputType
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_importer import FxImporter, FxImporterHooks, create_mlir_tensor_type


SUPPORTED = frozenset((torch.int8, torch.uint8, torch.int16, torch.int32,
                       torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64))


def owned_tensor_bytes(tensor):
    """Independent, read-only logical C-order bytes, not a view of caller memory."""
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.utils._python_dispatch import _disable_current_modes
    if (not isinstance(tensor, torch.Tensor) or isinstance(tensor, FakeTensor)
            or tensor.device.type != 'cpu' or tensor.layout != torch.strided
            or tensor.dtype not in SUPPORTED or sys.byteorder != 'little'):
        raise ValueError('Supported concrete little-endian CPU tensor required')
    if tensor.numel() <= 1 or any(type(n) is not int or n <= 0 for n in tensor.shape):
        raise ValueError('Scalar, empty and symbolic tensors use the native importer')
    with _disable_current_modes():
        concrete = tensor.detach().resolve_conj().resolve_neg().contiguous()
        # uint8 exposes BF16 and every other supported dtype without a NumPy
        # numerical cast. np.array(copy=True) creates the required owned snapshot.
        raw = concrete.view(torch.uint8).numpy()
        owned = np.array(raw, dtype=np.uint8, copy=True, order='C').reshape(-1)
    if (owned.nbytes != tensor.numel()*tensor.element_size()
            or owned.ctypes.data % tensor.element_size() != 0):
        raise ValueError('Constant byte count or actual buffer alignment differs')
    owned.flags.writeable = False
    return owned


class OwnedTensorLiteralHooks(FxImporterHooks):
    def __init__(self):
        self.records = []
        self.context = None

    def resolve_literal(self, gni, literal, info):
        if not isinstance(literal, torch.Tensor):
            return None
        # Keep native scalar/splat, empty and unsupported-type behavior. This is
        # importer delegation, not a model-computation CPU fallback.
        if literal.dtype not in SUPPORTED or literal.numel() <= 1:
            return None
        if self.context is not None and self.context != gni._c:
            raise ValueError('One literal hook instance must not span MLIR contexts')
        self.context = gni._c
        mapping = gni.fx_importer._py_attr_tracker.track(literal)
        if mapping.is_empty:
            owned = owned_tensor_bytes(literal)
            shape = list(literal.shape)
            name = 'hexagon_owned_tensor_' + '_'.join(map(str, shape)) + '_' + str(literal.dtype)
            attribute = ir.DenseResourceElementsAttr.get_from_buffer(
                owned, name, create_mlir_tensor_type(literal),
                alignment=literal.element_size(), is_mutable=False)
            mapping.value = attribute
            self.records.append(dict(shape=shape, dtype=str(literal.dtype),
                bytes=owned.nbytes, sha256=hashlib.sha256(memoryview(owned)).hexdigest(),
                independent_readonly_snapshot=True, per_element_Python_list=False))
        return ir.Operation.create('torch.vtensor.literal',
            results=[gni._cc.tensor_to_vtensor_type(literal)],
            attributes={'value': mapping.value}).result


def import_graph_owned_constants(gm, name):
    context = ir.Context()
    torch_d.register_dialect(context)
    hooks = OwnedTensorLiteralHooks()
    importer = FxImporter(context=context, hooks=hooks)
    importer.import_graph_module(gm, func_name=name)
    return importer, hooks


def lower_aten_graph_owned_constants(gm, name):
    importer, hooks = import_graph_owned_constants(gm, name)
    module = fx._module_lowering(False, False, OutputType.LINALG_ON_TENSORS, importer.module)
    return module, hooks.records
