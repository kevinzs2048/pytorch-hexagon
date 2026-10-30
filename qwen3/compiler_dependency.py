"""Resolve the separately maintained, pinned Hexagon-MLIR compiler checkout."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def compiler_root():
    lock = json.loads((ROOT / "dependencies/hexagon-mlir.lock.json").read_text())
    root = ROOT / lock["checkout"]
    if not (root / "qcom_hexagon_backend").is_dir():
        raise ValueError("Compiler dependency is missing. Initialize third_party/hexagon-mlir; see README.md.")
    for name, expected in lock["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned compiler source differs: " + name)
    return root


def native_root():
    """Qwen-specific compiler extensions and DSP helpers belong to this app."""
    return ROOT / "native"


def activate():
    path = str(compiler_root())
    if path not in sys.path:
        sys.path.insert(0, path)
