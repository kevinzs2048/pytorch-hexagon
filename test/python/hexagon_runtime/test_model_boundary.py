"""Model dispatch stays lazy; transport admission is supplied by its caller."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hexagon_runtime.hexagon_buffer_transport import HexagonLibraryBufferStream

ROOT = Path(__file__).resolve().parents[3]


def test_shared_primitives_do_not_import_model_adapters():
    for path in (ROOT / 'hexagon_runtime').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                     [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            assert not any(name.split('.')[0] in ('models', 'qwen3') for name in names), path


@pytest.mark.parametrize('module,args', [
    ('hexagon_runtime', ['qwen3', 'generate', '--help']),
    ('models.qwen3', ['generate', '--help']),
    ('qwen3', ['generate', '--help']),
])
def test_commands_work_without_private_pythonpath(module, args):
    result = subprocess.run([sys.executable, '-m', module, *args], cwd=ROOT,
                            env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '--prompt' in result.stdout


def test_unknown_model_is_rejected_before_execution():
    result = subprocess.run([sys.executable, '-m', 'hexagon_runtime', 'unknown', 'generate'],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr


@pytest.mark.parametrize('prefix', ['/dev/shm/qwen3-', '/dev/shm/another-adapter-'])
def test_library_namespace_is_explicit_and_cannot_be_crossed(prefix):
    validate = HexagonLibraryBufferStream.validate_library_path
    validate(prefix + 'session/model.so', prefix=prefix)
    for path in ('/tmp/model.so', prefix + '../model.so', prefix + 'model.so\nselect other',
                 '/dev/shm/unrelated/model.so'):
        with pytest.raises(ValueError):
            validate(path, prefix=prefix)
