"""Runtime isolation, cache invalidation and maintainer entry regressions."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from qwen3 import cli

ROOT = Path(__file__).resolve().parents[3]


def test_cache_tracks_runtime_changes_but_not_diagnostic_edits(tmp_path, monkeypatch):
    manifest = ROOT/'qwen3/runtime-sources.json'
    files = json.loads(manifest.read_text())['files']
    for name in [*files, 'qwen3/runtime-sources.json']:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    original = cli.implementation_digest()
    diagnostic = tmp_path/'tools/qwen3/unrelated_probe.py'
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text('raise RuntimeError("not runtime code")\n')
    assert cli.implementation_digest() == original
    runtime = tmp_path/'qwen3/runtime/support/native_aot_output.py'
    runtime.write_text(runtime.read_text() + '\n# changed adapter\n')
    assert cli.implementation_digest() != original
    runtime.unlink()
    with pytest.raises(FileNotFoundError):
        cli.implementation_digest()


def test_execution_manifest_covers_local_imports_without_maintainer_tools():
    manifest = json.loads((ROOT/'qwen3/runtime-sources.json').read_text())
    files = set(manifest['files'])
    assert set(manifest['roots']) <= files
    assert len(files) == len(manifest['files'])
    assert not any(p.startswith('tools/') for p in files)
    search = [ROOT/'qwen3/runtime/arm', ROOT/'tools/qwen3', ROOT,
              ROOT/'third_party/hexagon-mlir']

    def resolve(name, source, level=0):
        if level:
            base = source.parent
            for _ in range(level - 1):
                base = base.parent
            bases = [base]
        else:
            bases = search
        for base in bases:
            path = base.joinpath(*name.split('.')) if name else base
            for candidate in (path.with_suffix('.py'), path/'__init__.py'):
                if candidate.is_file():
                    return candidate

    for name in files:
        source = ROOT/name
        for parent in source.parents:
            if not parent.is_relative_to(ROOT):
                break
            initializer = parent/'__init__.py'
            if initializer.is_file():
                assert str(initializer.relative_to(ROOT)) in files
        for node in ast.walk(ast.parse(source.read_text())):
            targets = []
            if isinstance(node, ast.Import):
                targets = [resolve(a.name, source) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = resolve(node.module or '', source, node.level)
                targets = [base]
                if base is None or base.name == '__init__.py':
                    targets += [resolve((node.module + '.' if node.module else '') + a.name,
                                        source, node.level) for a in node.names]
            for target in filter(None, targets):
                assert str(target.relative_to(ROOT)) in files, (name, str(target))


def test_native_entry_imports_no_maintainer_scripts():
    code = '''
import sys
from pathlib import Path
from qwen3.cli import ROOT, runtime_paths
runtime_paths()
import run_qwen_arm_dynamic_warm_generate
from qwen3.runtime.support import native_aot_output
from qwen3 import runtime_checks
for module in list(sys.modules.values()):
    path = getattr(module, '__file__', None)
    assert not path or not Path(path).resolve().is_relative_to(ROOT/'tools'), path
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('tool', ['build', 'verify', 'build_q2_tail_model_sdk', 'profile_model_memory'])
def test_maintainer_entry_needs_no_private_pythonpath(tool):
    result = subprocess.run([sys.executable, '-m', 'qwen3.tools', tool, '--help'], cwd=ROOT,
                            env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_comparison_still_rejects_changed_full_output(tmp_path):
    reference = ROOT/'qwen3/validation/reference16.json'
    candidate = json.loads(reference.read_text())
    path = tmp_path/'result.json'
    path.write_text(json.dumps(candidate))
    assert cli.compare(reference, path)['status'] == 'identical'
    candidate['forwards'][0]['output_sha256'] = '0' * 64
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='output_sha256'):
        cli.compare(reference, path)
