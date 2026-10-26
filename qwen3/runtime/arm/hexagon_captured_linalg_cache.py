"""Lower the actual AOT graph on a frontend-cache miss.

No model recapture, expected-output hash, or concrete symbolic shape hint is
used as a cache identity. This caches torch-mlir output only: callers still
need Hexagon code generation, linking, and independent device acceptance.
"""

import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from hexagon_compiled_artifact_cache import canonical, describe_artifacts, digest_file


def captured_identity(graph):
    """Include code, symbolic shape algebra/ranges, strides and literal tensors.

    Symbol spellings are deliberately retained. Renaming can cause an extra
    miss, but cannot cause a wrong hit. Hints such as this request's q=5 are
    excluded; Dynamo guards and the runtime ABI continue to validate calls.
    """
    import torch
    from torch._subclasses.fake_tensor import is_fake
    from hexagon_dynamic_module_constants import dynamic_tensor_digest

    def scalar(value):
        if type(value) in (int, bool, str) or value is None:
            return value
        if type(value) is float:
            return dict(float_hex=value.hex())
        if isinstance(value, (torch.SymInt, torch.SymBool, torch.SymFloat)):
            expression = value.node.expr
            ranges = value.node.shape_env.var_to_range
            return dict(symbolic_type=type(value).__name__, expression=str(expression),
                        ranges={str(symbol): [str(ranges[symbol].lower),
                                              str(ranges[symbol].upper)]
                                for symbol in sorted(expression.free_symbols, key=str)
                                if symbol in ranges})
        raise ValueError('Unsupported captured metadata: ' + type(value).__name__)

    def metadata(value):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise ValueError('Strided captured tensors required')
            return dict(dtype=str(value.dtype), device=str(value.device),
                        layout=str(value.layout), shape=[scalar(x) for x in value.shape],
                        stride=[scalar(x) for x in value.stride()],
                        storage_offset=scalar(value.storage_offset()),
                        requires_grad=value.requires_grad)
        if isinstance(value, (tuple, list)):
            return dict(sequence_type=type(value).__name__,
                        values=[metadata(x) for x in value])
        return scalar(value)

    rows = []
    for node in graph.graph.nodes:
        row = dict(op=node.op, target=str(node.target))
        if node.op == 'get_attr':
            value = graph
            for name in node.target.split('.'):
                value = getattr(value, name)
            if not isinstance(value, torch.Tensor) or is_fake(value):
                raise ValueError('Concrete captured tensor constants required')
            # AOT invokes the compiler under FakeTensorMode, but get_attr
            # literals here are checked concrete tensors, not example inputs.
            from torch.utils._python_dispatch import _disable_current_modes
            with _disable_current_modes():
                row['constant_sha256'] = dynamic_tensor_digest(value)
        if node.op != 'output':
            if 'val' not in node.meta:
                raise ValueError('Captured node lacks authoritative value metadata')
            row['metadata'] = metadata(node.meta['val'])
        rows.append(row)
    payload = canonical(dict(code=graph.code, nodes=rows))
    return dict(sha256=hashlib.sha256(payload).hexdigest(), nodes=len(rows),
                inputs=sum(n.op == 'placeholder' for n in graph.graph.nodes))


def frontend_dependencies():
    """Fingerprint the actual installed importer/lowering code and bindings."""
    import torch
    names = ('torch_mlir.fx', 'torch_mlir.compiler_utils',
             'torch_mlir.extras.fx_importer', 'torch_mlir._mlir_libs._mlir',
             'torch_mlir._mlir_libs._torchMlir', 'hexagon_tensor_literal_buffers',
             'hexagon_dynamic_module_constants', 'hexagon_dynamo_parameters')
    rows = [dict(module=name, sha256=digest_file(importlib.import_module(name).__file__))
            for name in names]
    rows.append(dict(module=__name__, sha256=digest_file(__file__)))
    return dict(torch_version=torch.__version__, torch_git=torch.version.git_version,
                sources=rows)


class CapturedLinalgCache:
    def __init__(self, root):
        root = Path(root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError('Existing explicit captured-frontend cache directory required')
        self.root = root.resolve()
        self.dependencies = frontend_dependencies()

    def _read(self, key, request):
        entry = self.root / (key + '.json')
        if not entry.exists():
            if entry.is_symlink():
                raise ValueError('Dangling frontend cache entry')
            return None
        if entry.is_symlink() or not entry.is_file():
            raise ValueError('Regular frontend manifest required')
        record = json.loads(entry.read_text())
        build_name = record.get('build', '')
        if (record.get('status') != 'captured_AOT_Linalg_cached_NOT_machine_code'
                or record.get('request') != request or record.get('key') != key
                or not isinstance(build_name, str) or Path(build_name).name != build_name
                or not build_name.startswith(key + '-')):
            raise ValueError('Captured frontend cache identity drifted')
        folder = self.root / build_name
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError('Original frontend transaction required')
        rows = record.get('artifacts')
        if (not isinstance(rows, list) or not rows
                or rows != describe_artifacts(folder, [x['path'] for x in rows])):
            raise ValueError('Cached Linalg bytes drifted; refusing execution')
        return dict(record, directory=str(folder))

    def get_or_lower(self, graph, binding, *, name):
        """Compile a private copy; preserve the graph AOTAutograd owns."""
        import torch
        from hexagon_tensor_literal_buffers import lower_aten_graph_owned_constants
        from probe_qwen3_dynamic_bound_linalg import BoundedBytecodeSink

        before = captured_identity(graph)
        if before['inputs'] != binding.input_count:
            raise ValueError('Captured graph and parameter binding arity differ')
        request = dict(schema=1, graph=before, function_name=name,
                       parameters_sha256=hashlib.sha256(canonical(binding.records)).hexdigest(),
                       retained_input_indices=list(binding.retained),
                       frontend_sha256=hashlib.sha256(canonical(self.dependencies)).hexdigest())
        if hasattr(binding,'cache_constants_signature'):
            request['guarded_module_constants_sha256']=binding.cache_constants_signature
        key = hashlib.sha256(canonical(request)).hexdigest()
        fd = os.open(self.root / (key + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            hit = self._read(key, request)
            if hit is not None:
                return dict(hit, cache_hit=True, frontend_invocations=0,
                            machine_code_invocations=0)
            frozen_bytes = sum(row['bytes'] for row in binding.records)
            margin = min(512 * 1024**2, max(16 * 1024**2, frozen_bytes // 2))
            required = frozen_bytes + margin
            if shutil.disk_usage(self.root).free < required:
                raise ValueError('Insufficient space for actual captured Linalg and margin')
            folder = Path(tempfile.mkdtemp(prefix=key + '-', dir=self.root))
            started = time.monotonic()
            try:
                private = torch.fx.GraphModule(graph, copy.deepcopy(graph.graph))
                binding.freeze(private)
                module, resources = lower_aten_graph_owned_constants(private, name)
                if not module.operation.verify():
                    raise ValueError('Actual captured graph did not lower to verified Linalg')
                target = folder / 'captured-linalg.mlirbc'
                with target.open('xb') as stream:
                    sink = BoundedBytecodeSink(stream, max(required, 1024**2))
                    module.operation.write_bytecode(sink)
                    sink.flush()
                with target.open('rb') as stream:
                    if stream.read(4) != b'ML\xefR':
                        raise ValueError('Expected actual MLIR bytecode')
                signatures = [str(op.attributes['function_type']) for op in module.body.operations
                              if 'function_type' in op.attributes]
                if len(signatures) != 1 or captured_identity(graph) != before:
                    raise ValueError('Frontend altered native graph or returned unexpected functions')
                evidence = dict(actual_AOT_graph_lowered=True, original_graph_unchanged=True,
                                signatures=signatures, resources=resources,
                                frontend_seconds=time.monotonic() - started,
                                dependencies=self.dependencies, machine_code_invocations=0,
                                DSP_calls=0, numerical_acceptance=False)
                (folder / 'lowering.json').write_bytes(canonical(evidence) + b'\n')
                del module, private
                record = dict(status='captured_AOT_Linalg_cached_NOT_machine_code',
                              key=key, request=request, build=folder.name,
                              artifacts=describe_artifacts(folder, ['captured-linalg.mlirbc', 'lowering.json']))
                pending = folder / 'entry.pending.json'
                with pending.open('xb') as stream:
                    stream.write(canonical(record) + b'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(pending, self.root / (key + '.json'))
                return dict(self._read(key, request), cache_hit=False, frontend_invocations=1,
                            machine_code_invocations=0)
            except BaseException as error:
                (folder / 'failure.json').write_bytes(canonical(
                    dict(error_type=type(error).__name__, error=str(error))) + b'\n')
                raise
