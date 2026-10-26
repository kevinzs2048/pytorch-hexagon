"""Prevent dirty source exports from silently reusing a compiler build."""
import json
from pathlib import Path
import subprocess

import pytest

from qwen3.build import export_source, product_paths, sha, source_identity, verify_products


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


def test_model_build_rejects_changed_compiler_products(source, tmp_path):
    repo, _, _ = source
    output = tmp_path/'build'
    output.mkdir()
    config = tmp_path/'toolchain.json'
    config.write_text('{}')
    for path in product_paths(output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original build product')
    record = dict(status='compiler_built_NOT_qualified', configuration_sha256=sha(config),
                  compiler_source=str(repo), compiler_sources_sha256=source_identity(repo)['sha256'],
                  products={str(p.relative_to(output)): sha(p) for p in product_paths(output)})
    (output/'build.json').write_text(json.dumps(record))
    assert verify_products(output, repo, config) == record
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
