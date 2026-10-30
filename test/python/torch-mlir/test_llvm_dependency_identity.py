"""A base Git revision alone must not admit changed LLVM libraries or patches."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('llvm_dependency', ROOT/'dependencies/llvm/toolchain.py')
dependency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependency)


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    source, build, recipe = (tmp_path/name for name in ('source', 'build', 'recipe'))
    source.mkdir()
    recipe.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source/'llvm').mkdir()
    header = source/'llvm/input.h'
    header.write_text('base\n')
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base'], check=True)
    revision = dependency.git(source, 'rev-parse', 'HEAD')
    header.write_text('qualified patch\n')
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary', 'HEAD'])
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    tree = dependency.git(source, 'write-tree')
    subprocess.run(['git', '-C', str(source), 'reset', '-q', 'HEAD'], check=True)
    (recipe/'qualified.patch').write_bytes(patch)
    (recipe/'recipe.json').write_text(json.dumps(dict(schema=1, base_revision=revision,
        source_tree=tree, patch='qualified.patch', patch_sha256=hashlib.sha256(patch).hexdigest())))
    monkeypatch.setattr(dependency, 'HERE', recipe)
    config = dict(llvm_source=str(source), llvm_build=str(build),
                  llvm_receipt=str(tmp_path/'receipt.json'))
    for key in ('host_clang', 'host_clangxx', 'lld'):
        path = tmp_path/key
        path.write_bytes(b'qualified host tool')
        config[key] = str(path)
    cache = dict(CMAKE_HOME_DIRECTORY=str(source/'llvm'),
        LLVM_TARGETS_TO_BUILD='AMDGPU;NVPTX;X86;Hexagon', LLVM_ENABLE_ASSERTIONS='ON',
        LLVM_ENABLE_EH='ON', LLVM_ENABLE_RTTI='ON', CMAKE_C_COMPILER=config['host_clang'],
        CMAKE_CXX_COMPILER=config['host_clangxx'], CMAKE_MAKE_PROGRAM=config['host_clang'],
        CMAKE_COMMAND=config['host_clang'])
    files = ['bin/'+name for name in dependency.TOOLS]
    files += ['lib/libLLVMCore.a', 'lib/libLLVMHexagonCodeGen.a', 'lib/libMLIRIR.a',
              'lib/liblldELF.a', 'include/llvm/Config/llvm-config.h',
              'tools/mlir/include/mlir/IR/BuiltinOps.h.inc',
              'lib/cmake/llvm/LLVMConfig.cmake', 'lib/cmake/mlir/MLIRConfig.cmake',
              'lib/cmake/lld/LLDConfig.cmake']
    for name in files:
        p = build/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'qualified product')
    (build/'CMakeCache.txt').write_text('\n'.join(k+':STRING='+v for k, v in cache.items()))
    dependency.seal(config, 'qualified_prebuilt_snapshot')
    return config


def test_prepare_replays_patch_without_modifying_original_index(toolchain, tmp_path):
    source = Path(toolchain['llvm_source'])
    before = dependency.git(source, 'status', '--porcelain')
    destination = tmp_path/'fresh-source'
    dependency.prepare(source, destination)
    assert (destination/'llvm/input.h').read_bytes() == (source/'llvm/input.h').read_bytes()
    assert dependency.git(source, 'status', '--porcelain') == before
    dependency.prepare(source, destination)


def test_snapshot_is_explicitly_distinct_from_a_source_build(toolchain):
    assert dependency.verify(toolchain)['origin'] == 'qualified_prebuilt_snapshot'


@pytest.mark.parametrize('name', [
    'lib/libLLVMCore.a', 'bin/llc', 'include/llvm/Config/llvm-config.h',
    'tools/mlir/include/mlir/IR/BuiltinOps.h.inc', 'lib/cmake/mlir/MLIRConfig.cmake'])
def test_same_revision_with_changed_products_is_rejected(toolchain, name):
    (Path(toolchain['llvm_build'])/name).write_bytes(b'other toolchain')
    with pytest.raises(ValueError, match='changed after sealing'):
        dependency.verify(toolchain)


@pytest.mark.parametrize('name', ['llvm/input.h', 'llvm/unrecorded.cpp'])
def test_same_revision_with_different_local_sources_is_rejected(toolchain, name):
    (Path(toolchain['llvm_source'])/name).write_text('different local patch\n')
    with pytest.raises(ValueError, match='complete patched tree'):
        dependency.verify(toolchain)


def test_replaced_external_linker_is_rejected(toolchain):
    Path(toolchain['lld']).write_bytes(b'other linker')
    with pytest.raises(ValueError, match='changed after sealing'):
        dependency.verify(toolchain)


def test_missing_identity_is_not_silently_adopted(toolchain):
    del toolchain['llvm_receipt']
    with pytest.raises(ValueError, match='receipt required'):
        dependency.verify(toolchain)


def test_build_directory_from_another_source_is_rejected(toolchain):
    p = Path(toolchain['llvm_build'])/'CMakeCache.txt'
    p.write_text(p.read_text().replace(toolchain['llvm_source'], '/different/source'))
    with pytest.raises(ValueError, match='different source tree'):
        dependency.verify(toolchain)


def test_loader_resolution_tracks_library_bytes_and_rejects_missing_libraries(tmp_path, monkeypatch):
    executable, library = tmp_path/'tool', tmp_path/'libdep.so'
    executable.write_bytes(b'\x7fELFfixture')
    library.write_bytes(b'original library')
    result = SimpleNamespace(returncode=0, stdout='libdep.so => '+str(library)+' (0x123)\n\n')
    monkeypatch.setattr(dependency.subprocess, 'run', lambda *args, **kwargs: result)
    before = dependency.dynamic_libraries([executable], {})
    library.write_bytes(b'changed library')
    assert dependency.dynamic_libraries([executable], {}) != before
    result.stdout = 'libdep.so => not found\n'
    with pytest.raises(ValueError, match='Missing LLVM host dependency'):
        dependency.dynamic_libraries([executable], {})
