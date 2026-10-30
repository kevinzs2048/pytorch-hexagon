"""Reject missing/misdirected explicit Triton Python sources before execution."""
import importlib.util
from pathlib import Path
import sys
from models.qwen3.bundle import sha


def configure_triton_python(directory):
    root=Path(directory).resolve(strict=True)
    package=root/'triton/__init__.py'
    if not package.is_file(): raise ValueError('Explicit Triton Python directory must contain triton/__init__.py')
    expected=package.resolve()
    loaded=sys.modules.get('triton')
    if loaded is not None and Path(getattr(loaded,'__file__','')).resolve()!=expected:
        raise ValueError('An already-loaded Triton package differs from the explicit source')
    sys.path.insert(0,str(root))
    try:
        spec=importlib.util.find_spec('triton')
        if spec is None or spec.origin is None or Path(spec.origin).resolve()!=expected:
            raise ValueError('Triton module resolution differs from the explicit source')
    except BaseException:
        sys.path.pop(0)
        raise
    return dict(root=str(root),package_source=str(expected),package_source_sha256=sha(expected))
