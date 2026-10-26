"""Load an unchanged, pinned Python decomposition policy without MLIR binaries.

The policy is executable trusted deployment code, not an untrusted model file.
Its digest must come from the reviewed build manifest. This is integrity checking,
not a sandbox. The compiler policy remains torch-mlir's; no new decompositions,
model arithmetic, generation loop or implicit CPU fallback are introduced here.
"""
import hashlib
from pathlib import Path
import types


def load_runtime_decompositions(path, *, expected_sha256, expected_torch_version,
                                expected_torch_git):
    import torch
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Explicit lowercase SHA256 policy binding required')
    if (not expected_torch_version or not expected_torch_git
            or torch.__version__ != expected_torch_version
            or torch.version.git_version != expected_torch_git):
        raise ValueError('Pinned PyTorch version and source revision required')
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024:
        raise ValueError('Bounded Python policy file required')
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('Decomposition policy bytes changed')
    # Compile the bytes already checked; do not import torch_mlir or write pyc.
    module = types.ModuleType('_hexagon_pinned_decomposition_' + expected_sha256)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    factory = getattr(module, 'get_decomposition_table', None)
    if not callable(factory):
        raise ValueError('Policy must expose get_decomposition_table')
    table = factory()
    if (not isinstance(table, dict) or not table
            or any(not isinstance(op, torch._ops.OpOverload) or not callable(fn)
                   for op, fn in table.items())):
        raise ValueError('Actual operator-to-callable decomposition table required')
    return table
