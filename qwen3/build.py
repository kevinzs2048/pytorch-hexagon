"""Build the pinned compiler from sources in one isolated output directory.

Usage: python -m qwen3.build --config toolchain.json --output .local/build
The accepted execution bundle is a reference, never a source of object files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from .bundle import write
from .compiler_dependency import compiler_root


def product_paths(output):
    paths = [output/'compiler/libtriton.so',
             output/'compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt']
    paths += [output/'compiler/profiles'/p/'libtriton.so' for p in ('q1', 'q2')]
    paths += [output/'policy'/name for name in ('qwen-policy-first-opt', 'qwen-policy-sums-opt',
                                              'qwen-packed-i8-opt', 'qwen-silu-effects-opt')]
    paths += [output/'runtime'/(name+'.o') for name in (
        'packed-int8-runtime', 'packed-multi-i8-runtime', 'guarded-copy-runtime',
        'fixed512-memcpy', 'preserved-memcpy', 'caller-byte-copy', 'global-hvx-copy',
        'q1-pv-guard', 'q1-qk-guard', 'q2-qk-guard', 'q2-zero-guard',
        'trig_lane', 'rempitab', 'trig_runtime')]
    return paths


def verify_products(output, source, config):
    record = json.loads((output/'build.json').read_text())
    if (record['status'] != 'compiler_built_NOT_qualified'
            or record['configuration_sha256'] != sha(config)
            or record['compiler_source'] != str(source)
            or record['compiler_sources_sha256'] != source_identity(source)['sha256']):
        raise ValueError('Compiler build inputs changed; rerun the source build')
    actual = {str(p.relative_to(output)): sha(p) for p in product_paths(output)}
    if actual != record.get('products'):
        raise ValueError('Compiler build products changed or are not sealed')
    return record


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_source(repository, revision, destination):
    """Materialize exactly the requested Git tree, excluding local experiments."""
    expected = dict(repository=str(repository), revision=revision)
    tree = subprocess.check_output(['git', '-C', str(repository), 'ls-tree', '-rz', revision])
    def verify():
        expected_paths = {'.qwen-source.json'}
        for row in tree.split(b'\0'):
            if not row:
                continue
            descriptor, raw_name = row.split(b'\t', 1)
            mode, kind, object_id = descriptor.split()
            if kind == b'commit':
                raise ValueError('Triton export contains an uninitialized Git submodule')
            path = destination / os.fsdecode(raw_name)
            expected_paths.add(os.fsdecode(raw_name))
            if mode == b'120000':
                if not path.is_symlink():
                    raise ValueError('Exported source symlink changed: ' + str(path))
                data = os.fsencode(os.readlink(path))
            else:
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Exported source file missing: ' + str(path))
                if bool(path.stat().st_mode & 0o111) != (mode == b'100755'):
                    raise ValueError('Exported source executable mode changed: ' + str(path))
                data = path.read_bytes()
            identity = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if identity != object_id.decode():
                raise ValueError('Exported source differs from pinned Git tree: ' + str(path))
        extra = {str(p.relative_to(destination)) for p in destination.rglob('*')
                 if p.is_file() or p.is_symlink()} - expected_paths
        if extra:
            raise ValueError('Untracked files in exported source: ' + ', '.join(sorted(extra)))
    if destination.exists():
        marker = destination / '.qwen-source.json'
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise ValueError('Build source directory belongs to a different input')
        verify()
        return
    destination.mkdir(parents=True)
    archive = destination.parent / (destination.name + '.tar')
    try:
        subprocess.run(['git', '-C', str(repository), 'archive', revision,
                        '-o', str(archive)], check=True)
        with tarfile.open(archive) as source:
            source.extractall(destination, filter='data')
        verify()
        write(destination / '.qwen-source.json', expected)
    finally:
        # A partial export has no marker and cannot be admitted on resume.
        archive.unlink(missing_ok=True)


def source_identity(repository):
    names = subprocess.check_output(['git', '-C', str(repository), 'ls-files', '-z',
                                     '--cached', '--others', '--exclude-standard'])
    files = {}
    for name in sorted(set(filter(None, names.decode().split('\0')))):
        path = repository / name
        if path.is_symlink():
            files[name] = {'symlink': os.readlink(path)}
        elif path.is_file():
            files[name] = sha(path)
        elif not path.exists():
            files[name] = None
    encoded = json.dumps(files, sort_keys=True).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--configure-only', action='store_true')
    parser.add_argument('--models', type=Path,
                        help='JSON with context, hf, catalog and q1/q2 producer path+sha256 records')
    parser.add_argument('--trusted-local-producers', action='store_true')
    parser.add_argument('--compiler-source', type=Path,
                        help='Explicit development checkout; defaults to the pinned compiler dependency')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('jobs must be positive')
    if args.models and (args.configure_only or not args.trusted_local_producers):
        parser.error('Model builds require explicitly trusted local captures and a complete compiler build')
    config = json.loads(args.config.read_text())
    if Path(config['lld']).name != 'ld.lld':
        raise ValueError('The Hexagon compiler driver requires linker basename ld.lld')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.compiler_source.resolve() if args.compiler_source else compiler_root()
    if not (source/'qcom_hexagon_backend/CMakeLists.txt').is_file():
        raise ValueError('Expected a Hexagon-MLIR source checkout')
    llvm = Path(config['llvm_build']).resolve()
    locks = json.loads((source/'dependencies/sources.lock.json').read_text())
    llvm_revision = subprocess.check_output(['git', '-C', config['llvm_source'],
                                            'rev-parse', 'HEAD'], text=True).strip()
    if llvm_revision != locks['llvm']['revision']:
        raise ValueError('LLVM source revision differs from the compiler lock')
    triton_revision = locks['triton']['revision']
    sdk = Path(config['hexagon_sdk']).resolve()
    tools = Path(config['hexagon_tools']).resolve()
    triton = output / 'sources/triton'
    export_source(Path(config['triton_source']).resolve(), triton_revision, triton)
    build = output / 'compiler'
    environment = dict(os.environ, HEXAGON_SDK_ROOT=str(sdk),
                       HEXAGON_TOOLS=str(tools), HEXAGON_ARCH_VERSION='73',
                       HEXKL_ROOT=str(Path(config['hexkl']).resolve()),
                       HEXAGON_MLIR_ROOT=str(source), LLVM_SYSPATH=str(llvm),
                       LD_LIBRARY_PATH=os.pathsep.join(config.get('library_path', [])))
    python = Path(config.get('python', sys.executable)).resolve()
    pybind = subprocess.check_output([str(python), '-c',
                                     'import pybind11; print(pybind11.get_cmake_dir())'], text=True).strip()
    configure = [config.get('cmake', 'cmake'), '-S', str(triton), '-B', str(build), '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_FLAGS_RELEASE=-O2 -g0 -DNDEBUG',
                 '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -g0 -DNDEBUG',
                 '-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--strip-debug',
                 '-DCMAKE_EXE_LINKER_FLAGS=-Wl,--strip-debug',
                 '-DCMAKE_C_COMPILER='+config['host_clang'],
                 '-DCMAKE_CXX_COMPILER='+config['host_clangxx'],
                 '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                 '-DLLVM_SYSPATH='+str(llvm), '-DLLVM_LIBRARY_DIR='+str(llvm/'lib'),
                 '-DMLIR_DIR='+str(llvm/'lib/cmake/mlir'),
                 '-DLLVM_DIR='+str(llvm/'lib/cmake/llvm'),
                 '-DLLD_DIR='+str(llvm/'lib/cmake/lld'),
                 '-DPython3_EXECUTABLE='+str(python), '-Dpybind11_DIR='+pybind,
                 '-DTRITON_BUILD_PYTHON_MODULE=ON', '-DTRITON_BUILD_PROTON=OFF',
                 '-DTRITON_BUILD_UT=OFF', '-DTRITON_BUILD_WITH_CCACHE=OFF',
                 '-DTRITON_CODEGEN_BACKENDS=nvidia;amd',
                 '-DTRITON_PLUGIN_DIRS='+str(source/'qcom_hexagon_backend'),
                 '-DTRITON_WHEEL_DIR='+str(output/'wheel'),
                 '-DTRITON_PARALLEL_LINK_JOBS=1']
    configure.append('-DQWEN_BUILD_COMPILER_PROFILES=ON')
    record = dict(schema=1, status='building', started_unix=time.time(),
                  configuration_sha256=sha(args.config), compiler_source=str(source),
                  compiler_revision=subprocess.check_output(
                      ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                  compiler_dirty=bool(subprocess.check_output(
                      ['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()),
                  triton_revision=triton_revision, commands=[], compiler_rebuild_verified=False,
                  model_rebuild_verified=False, DSP_calls=0)
    identity = source_identity(source)
    write(output/'compiler-sources.json', identity)
    record['compiler_sources_sha256'] = identity['sha256']
    record['LLVM_rebuilt'] = False
    record['LLVM_build'] = str(llvm)
    record['LLVM_revision'] = llvm_revision
    def run(name, command):
        row = dict(stage=name, argv=command, started_unix=time.time())
        record['commands'].append(row)
        write(output/'build.json', record)
        with (output/(name+'.log')).open('w') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode, completed_unix=time.time())
        write(output/'build.json', record)
        result.check_returncode()
    try:
        run('configure', configure)
        if not args.configure_only:
            run('compiler', [config.get('cmake', 'cmake'), '--build', str(build),
                             '--target', 'triton', 'linalg-hexagon-opt',
                             'qwen-compiler-q1', 'qwen-compiler-q2',
                             '--parallel', str(args.jobs), '--', '-k', '0'])
            policy = output / 'policy'
            run('policy-configure', [config.get('cmake', 'cmake'),
                '-S', str(source/'tools/qwen3/policy'), '-B', str(policy), '-G', 'Ninja',
                '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -g0 -DNDEBUG',
                '-DCMAKE_C_COMPILER='+config['host_clang'],
                '-DCMAKE_CXX_COMPILER='+config['host_clangxx'],
                '-DMLIR_DIR='+str(llvm/'lib/cmake/mlir'),
                '-DLLVM_DIR='+str(llvm/'lib/cmake/llvm'),
                '-DQWEN_BUILD_PACKED_INT8_EXPERIMENT=ON',
                '-DQWEN_BUILD_SILU_EFFECTS_EXPERIMENT=ON'])
            run('policy', [config.get('cmake', 'cmake'), '--build', str(policy),
                           '--parallel', str(args.jobs), '--', '-k', '0'])
            runtime = output / 'runtime'
            run('runtime-configure', [config.get('cmake', 'cmake'),
                '-S', str(source/'tools/qwen3/runtime'), '-B', str(runtime), '-G', 'Ninja',
                '-DHEXAGON_TOOLS='+str(tools), '-DHEXAGON_SDK='+str(sdk),
                '-DHEXAGON_LLD='+config['lld']])
            run('runtime', [config.get('cmake', 'cmake'), '--build', str(runtime),
                            '--parallel', str(args.jobs), '--', '-k', '0'])
        record['status'] = 'configured' if args.configure_only else 'compiler_built_NOT_qualified'
        if source_identity(source) != identity:
            raise ValueError('Compiler sources changed during the build; rerun before qualification')
        if not args.configure_only:
            record['products'] = {str(p.relative_to(output)): sha(p) for p in product_paths(output)}
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['completed_unix'] = time.time()
        write(output/'build.json', record)
    if args.models:
        inputs = json.loads(args.models.read_text())
        if set(inputs['producers']) != {'q1', 'q2'}:
            raise ValueError('Both q1 and q2 captures are required')
        for profile in ('q1', 'q2'):
            producer = inputs['producers'][profile]
            command = [str(python), '-m', 'qwen3.build_model', '--config', str(args.config.resolve()),
                       '--build', str(output), '--compiler-source', str(source),
                       '--producer', producer['path'], '--producer-sha256', producer['sha256'],
                       '--trusted-local-producer', '--output', str(output/'models'/profile)]
            for name in ('context', 'hf', 'catalog'):
                command += ['--'+name, inputs[name]]
            with (output/('model-'+profile+'.log')).open('w') as log:
                subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output/'models'/profile/'result.json').read_text())
            if result.get('profile') != profile:
                raise ValueError('Producer was assigned to the wrong model domain: '+profile)


if __name__ == '__main__':
    main()
