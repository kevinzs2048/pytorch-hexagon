"""Prepare, build and verify the explicit LLVM dependency (Python stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
TOOLS = ('llvm-config', 'llvm-tblgen', 'mlir-tblgen', 'mlir-opt', 'llc',
         'FileCheck', 'llvm-mc', 'llvm-objdump', 'llvm-readobj', 'llvm-ar',
         'llvm-as', 'llvm-dis', 'opt', 'ld.lld', 'llvm-lit')
_HASHES = {}


def sha(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
           stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASHES:
        with path.open('rb') as stream:
            value = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = path.stat()
        after_key = (str(path.resolve()), after.st_dev, after.st_ino, after.st_size,
                     after.st_mtime_ns, after.st_ctime_ns)
        if key != after_key:
            raise ValueError('Dependency changed while hashing: ' + str(path))
        _HASHES[key] = value
    return _HASHES[key]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def recipe():
    result = json.loads((HERE/'recipe.json').read_text())
    if result['schema'] != 1 or sha(HERE/result['patch']) != result['patch_sha256']:
        raise ValueError('LLVM dependency recipe/patch changed')
    return result


def git(source, *args, env=None):
    return subprocess.check_output(['git', '-C', str(source), *args],
                                   env=env, text=True).strip()


def source_tree(source, expected):
    """Hash actual working sources without staging or editing the user's index."""
    if git(source, 'rev-parse', 'HEAD') != expected['base_revision']:
        raise ValueError('LLVM base revision differs from recipe')
    index = Path(git(source, 'rev-parse', '--path-format=absolute', '--git-path', 'index'))
    with tempfile.TemporaryDirectory(prefix='qwen-llvm-index-') as folder:
        temporary = Path(folder)/'index'
        shutil.copy2(index, temporary)
        env = dict(os.environ, GIT_INDEX_FILE=str(temporary))
        subprocess.run(['git', '-C', str(source), 'add', '-A'], env=env, check=True)
        actual = git(source, 'write-tree', env=env)
    if actual != expected['source_tree']:
        raise ValueError('LLVM working sources differ from the complete patched tree')
    return actual


def prepare(repository, source):
    expected = recipe()
    if not source.exists():
        subprocess.run(['git', '-C', str(repository), 'worktree', 'add', '--detach',
                        str(source), expected['base_revision']], check=True)
        subprocess.run(['git', '-C', str(source), 'apply', '--check',
                        str(HERE/expected['patch'])], check=True)
        subprocess.run(['git', '-C', str(source), 'apply',
                        str(HERE/expected['patch'])], check=True)
    source_tree(source, expected)


def cache(build):
    values = {}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line:
            name, value = line.split('=', 1)
            values[name.split(':', 1)[0]] = value
    return values


def entry(path):
    result = {'sha256': sha(path)}
    if path.is_symlink():
        result['symlink'] = os.readlink(path)
    return result


def dynamic_libraries(paths, config):
    """Record the loader's actual dependency resolution, without ASLR addresses."""
    libraries = {}
    environment = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(config.get('library_path', [])))
    for path in sorted({p.resolve() for p in paths}):
        with path.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                continue
        result = subprocess.run(['ldd', str(path)], env=environment, capture_output=True, text=True)
        if 'not found' in result.stdout:
            raise ValueError('Missing LLVM host dependency: ' + result.stdout)
        if result.returncode:
            raise ValueError('Cannot inspect LLVM host dependencies: ' + str(path))
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            name = line.split('=>')[-1].strip().split()[0]
            if name.startswith('/'):
                libraries[name] = entry(Path(name))
    return libraries


def inventory(config):
    build = Path(config['llvm_build']).resolve()
    source = Path(config['llvm_source']).resolve()
    values = cache(build)
    if Path(values['CMAKE_HOME_DIRECTORY']).resolve() != source/'llvm':
        raise ValueError('LLVM cache belongs to a different source tree')
    if set(values['LLVM_TARGETS_TO_BUILD'].split(';')) != {'AMDGPU', 'NVPTX', 'X86', 'Hexagon'}:
        raise ValueError('LLVM target set differs from the supported dependency')
    for name in ('LLVM_ENABLE_ASSERTIONS', 'LLVM_ENABLE_EH', 'LLVM_ENABLE_RTTI'):
        if values[name] != 'ON':
            raise ValueError('Required LLVM ABI/build option changed: ' + name)
    files = set(build/'bin'/name for name in TOOLS)
    files.add(build/'CMakeCache.txt')
    files.update(build/name for name in (
        'lib/libLLVMCore.a', 'lib/libLLVMHexagonCodeGen.a', 'lib/libMLIRIR.a',
        'lib/liblldELF.a', 'include/llvm/Config/llvm-config.h',
        'tools/mlir/include/mlir/IR/BuiltinOps.h.inc',
        'lib/cmake/llvm/LLVMConfig.cmake', 'lib/cmake/mlir/MLIRConfig.cmake',
        'lib/cmake/lld/LLDConfig.cmake'))
    archives = list((build/'lib').glob('*.a'))
    if not archives:
        raise ValueError('LLVM static libraries are missing')
    files.update(archives)
    files.update((build/'lib').glob('*.so*'))
    for directory in ('include', 'tools/mlir/include', 'tools/lld/include', 'lib/cmake'):
        files.update(p for p in (build/directory).rglob('*') if p.is_file() or p.is_symlink())
    products = {str(p.relative_to(build)): entry(p) for p in sorted(files)}
    external = {name: dict(path=str(Path(config[name]).absolute()), **entry(Path(config[name])))
                for name in ('host_clang', 'host_clangxx', 'lld')}
    for name in ('CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_MAKE_PROGRAM', 'CMAKE_COMMAND'):
        p = Path(values[name])
        external[name] = dict(path=str(p), **entry(p))
    executables = [build/'bin'/name for name in TOOLS]
    executables += [Path(row['path']) for row in external.values()]
    return dict(build=str(build), source=str(source), products=products, tools=external,
                dynamic_libraries=dynamic_libraries(executables, config))


def seal(config, origin):
    expected = recipe()
    source = Path(config['llvm_source']).resolve()
    tree = source_tree(source, expected)
    actual = inventory(config)
    source_tree(source, expected)
    result = dict(schema=1, origin=origin, source_tree=tree,
                  recipe_sha256=sha(HERE/'recipe.json'), **actual)
    write(Path(config['llvm_receipt']), result)
    return result


def verify(config):
    if not config.get('llvm_receipt'):
        raise ValueError('LLVM identity receipt required; see dependencies/llvm/README.md')
    path = Path(config['llvm_receipt'])
    record = json.loads(path.read_text())
    expected = recipe()
    if (record.get('schema') != 1
            or record.get('origin') not in ('source_build', 'qualified_prebuilt_snapshot')
            or record.get('recipe_sha256') != sha(HERE/'recipe.json')
            or record.get('source_tree') != expected['source_tree']):
        raise ValueError('LLVM receipt does not match the dependency recipe')
    source_tree(Path(config['llvm_source']).resolve(), expected)
    actual = inventory(config)
    if any(record.get(key) != value for key, value in actual.items()):
        raise ValueError('LLVM libraries, headers, tools or configuration changed after sealing')
    return dict(base_revision=expected['base_revision'],
                recipe_sha256=record['recipe_sha256'], source_tree=record['source_tree'],
                receipt_sha256=sha(path), origin=record['origin'])


def build(config, jobs):
    expected = recipe()
    source = Path(config['llvm_source']).resolve()
    output = Path(config['llvm_build']).resolve()
    source_tree(source, expected)
    if Path(config['lld']).absolute() != output/'bin/ld.lld':
        raise ValueError('A source-built dependency must use its own bin/ld.lld')
    marker = output/'qwen-llvm-build.json'
    identity = dict(recipe_sha256=sha(HERE/'recipe.json'), source=str(source),
                    host_clang=entry(Path(config['host_clang'])),
                    host_clangxx=entry(Path(config['host_clangxx'])),
                    library_path=config.get('library_path', []))
    if output.exists() and any(output.iterdir()):
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError('LLVM build directory belongs to different inputs')
    write(marker, identity)
    environment = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(config.get('library_path', [])))
    cmake = config.get('cmake', 'cmake')
    configure = [cmake, '-S', str(source/'llvm'), '-B', str(output), '-G', 'Ninja',
                 '-DCMAKE_C_COMPILER='+config['host_clang'],
                 '-DCMAKE_CXX_COMPILER='+config['host_clangxx']]
    configure += ['-D'+name+'='+value for name, value in expected['cmake'].items()]
    commands = [configure, [cmake, '--build', str(output), '--parallel', str(jobs),
                            '--target', *expected['targets']]]
    for name, command in zip(('configure', 'build'), commands):
        print('LLVM ' + name + ': ' + str(output), flush=True)
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    return seal(config, 'source_build')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    for name in ('build', 'seal-existing', 'verify'):
        p = commands.add_parser(name)
        p.add_argument('--config', type=Path, required=True)
        if name == 'build':
            p.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.repository.resolve(), args.source.resolve())
        return
    config = json.loads(args.config.read_text())
    if args.action == 'build':
        if args.jobs < 1:
            parser.error('jobs must be positive')
        build(config, args.jobs)
    elif args.action == 'seal-existing':
        if Path(config['llvm_receipt']).exists():
            parser.error('Choose a new receipt; existing identities are preserved')
        seal(config, 'qualified_prebuilt_snapshot')
    else:
        print(json.dumps(verify(config)))


if __name__ == '__main__':
    main()
