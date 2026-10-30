"""Prevent dirty source exports from silently reusing a compiler build."""
import json
from pathlib import Path
import subprocess

import pytest

from qwen3.build import export_source, product_paths, sha, source_identity, verify_products


def test_verification_rejects_mixed_optimizer_and_helper_checkouts(tmp_path, monkeypatch):
    from tools.qwen3 import verify
    pinned, candidate = tmp_path / 'pinned', tmp_path / 'candidate'
    pinned.mkdir()
    candidate.mkdir()
    for path in (pinned, candidate):
        subprocess.run(['git', 'init', '-q', str(path)], check=True)
    (pinned / 'helper.c').write_text('old implementation')
    (candidate / 'helper.c').write_text('candidate implementation')
    monkeypatch.setattr(verify, 'compiler_root', lambda: pinned)
    with pytest.raises(ValueError, match='would mix compiler checkouts'):
        verify.require_matching_test_checkout(candidate)


def test_verification_accepts_same_checkout_through_symlink(tmp_path, monkeypatch):
    from tools.qwen3 import verify
    pinned, alias = tmp_path / 'pinned', tmp_path / 'alias'
    pinned.mkdir()
    alias.symlink_to(pinned, target_is_directory=True)
    monkeypatch.setattr(verify, 'compiler_root', lambda: pinned)
    assert verify.require_matching_test_checkout(alias) == pinned.resolve()


def test_verification_accepts_identical_checkout_copy(tmp_path, monkeypatch):
    from tools.qwen3 import verify
    pinned, candidate = tmp_path / 'pinned', tmp_path / 'candidate'
    for path in (pinned, candidate):
        path.mkdir()
        subprocess.run(['git', 'init', '-q', str(path)], check=True)
        (path / 'helper.c').write_text('same implementation')
    monkeypatch.setattr(verify, 'compiler_root', lambda: pinned)
    assert verify.require_matching_test_checkout(candidate) == pinned.resolve()


@pytest.fixture
def source(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo/'input.txt').write_text('pinned source\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'source'], check=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    return repo, revision, tmp_path/'export'


def test_export_excludes_dirty_checkout_and_verifies_resume(source):
    repo, revision, output = source
    (repo/'input.txt').write_text('uncommitted experiment\n')
    export_source(repo, revision, output)
    assert (output/'input.txt').read_text() == 'pinned source\n'
    export_source(repo, revision, output)
    (output/'input.txt').write_text('changed after export\n')
    with pytest.raises(ValueError, match='differs from pinned Git tree'):
        export_source(repo, revision, output)


def test_resume_rejects_injected_source(source):
    repo, revision, output = source
    export_source(repo, revision, output)
    (output/'additional.cpp').write_text('unexpected build input')
    with pytest.raises(ValueError, match='Untracked files'):
        export_source(repo, revision, output)


def test_model_build_rejects_changed_compiler_products(source, tmp_path, monkeypatch):
    repo, _, _ = source
    output = tmp_path/'build'
    output.mkdir()
    config = tmp_path/'toolchain.json'
    config.write_text('{}')
    dependency = {'receipt_sha256': 'qualified LLVM identity'}
    native = {'native/runtime/helper.c': 'qualified native helper'}
    monkeypatch.setattr('qwen3.build.native_identity', lambda: native)
    monkeypatch.setattr('qwen3.build.llvm_identity', lambda *args: dependency)
    for path in product_paths(output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original build product')
    record = dict(status='compiler_built_NOT_qualified', configuration_sha256=sha(config),
                  compiler_source=str(repo), compiler_sources_sha256=source_identity(repo)['sha256'],
                  products={str(p.relative_to(output)): sha(p) for p in product_paths(output)},
                  LLVM_identity=dependency, native_sources=dict(native))
    (output/'build.json').write_text(json.dumps(record))
    assert verify_products(output, repo, config) == record
    native['native/runtime/helper.c'] = 'changed helper'
    with pytest.raises(ValueError, match='native extension sources changed'):
        verify_products(output, repo, config)
    native['native/runtime/helper.c'] = 'qualified native helper'
    monkeypatch.setattr('qwen3.build.llvm_identity', lambda *args: {'receipt_sha256': 'other LLVM'})
    with pytest.raises(ValueError, match='LLVM dependency identity differs'):
        verify_products(output, repo, config)
    monkeypatch.setattr('qwen3.build.llvm_identity', lambda *args: dependency)
    product_paths(output)[0].write_bytes(b'replaced compiler')
    with pytest.raises(ValueError, match='products changed'):
        verify_products(output, repo, config)


def test_model_build_rejects_source_edits_after_compiler_build(source, tmp_path):
    repo, _, _ = source
    output = tmp_path/'build'
    output.mkdir()
    config = tmp_path/'toolchain.json'
    config.write_text('{}')
    record = dict(status='compiler_built_NOT_qualified', configuration_sha256=sha(config),
                  compiler_source=str(repo), compiler_sources_sha256=source_identity(repo)['sha256'])
    (output/'build.json').write_text(json.dumps(record))
    (repo/'input.txt').write_text('source edited after build')
    with pytest.raises(ValueError, match='build inputs changed'):
        verify_products(output, repo, config)
