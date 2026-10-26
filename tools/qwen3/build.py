#!/usr/bin/env python3
"""Build the two owned policy stages and RoPE runtime, then bind a pinned SDK.

The LLVM/Hexagon SDK and immutable model constants are external dependencies.
No policy object, generated caller or compiled graph is imported from an old run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Also support the existing direct build command and its Python subprocesses.
sys.path.insert(0, str(ROOT))
from qwen3.compiler_dependency import compiler_tools


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def entry(path):
    return dict(path=str(Path(path).absolute()), sha256=sha(path))


def verify_dependencies(config, lock):
    # Hexagon clang selects the LLD argument dialect by executable basename.
    # Renaming identical bytes can select the GNU Hexagon linker dialect.
    if Path(config['lld']).name != 'ld.lld':
        raise ValueError('Hexagon clang requires the pinned linker basename ld.lld')
    checked = []
    for row in lock['files']:
        root = Path(config[row['root']])
        path = root / row['relative'] if row['relative'] else root
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f"Pinned SDK dependency changed: {row['root']}/{row['relative']}")
        checked.append(dict(root=row['root'], relative=row['relative'], sha256=row['sha256']))
    for row in json.loads((HERE/'model-constants.lock.json').read_text())['libraries']:
        path = Path(config['constants_dir']) / row['name']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f"Immutable model constants changed: {row['name']}")
    return checked


def bind_recipe(config, output):
    metadata = output/'recipe'
    metadata.mkdir()
    reference = json.loads((HERE/'abi-template.json').read_text())
    wrapper = compiler_tools()/'runtime/reference-wrapper.cpp'
    if sha(wrapper) != reference.pop('wrapper_sha256'):
        raise ValueError('Versioned ABI wrapper changed')
    header = compiler_tools()/'runtime/qwen3_sleef_exp_candidate.h'
    if sha(header) != reference.pop('runtime_header_sha256'):
        raise ValueError('Versioned exp runtime changed')
    constants = json.loads((HERE/'model-constants.lock.json').read_text())
    for row in constants['libraries']:
        row['path'] = str(Path(config['constants_dir'])/row.pop('name'))
    constants['inputs'] = dict(lld=entry(config['lld']))
    write(metadata/'constants.json', constants)
    # These two absent paths are argument placeholders. The linker replaces them
    # with the fresh graph object/output before execution; neither is an input.
    reference.update(status='versioned_Qwen_ABI_template_NOT_artifact',
        wrapper=entry(wrapper), silu_runtime=dict(header_copy_sha256=sha(header)),
        library=dict(path=str(metadata/'reference-output.so')),
        constants_result_sha256=sha(metadata/'constants.json'))
    write(metadata/'reference.json', reference)
    replacements = dict(config, runtime_include=str(compiler_tools()/'runtime/dsp-include'),
        reference_wrapper=str(wrapper), reference_object=str(metadata/'bounded_rmsnorm_qwen.o'),
        aligned_alloc_shim=str(compiler_tools()/'runtime/aligned_alloc_shim.cpp'),
        trig_runtime=str(output/'rope/trig_runtime.o'),
        reference_output=str(metadata/'reference-output.so'))
    argv = [arg.format(**replacements) for arg in json.loads((HERE/'link-template.json').read_text())['argv']]
    write(metadata/'manifest.json', dict(link_command=shlex.join(argv)))
    paths = dict(attention_optimizer=output/'policy/qwen-policy-first-opt',
        sums_optimizer=output/'policy/qwen-policy-sums-opt',
        knorm_optimizer=output/'policy/qwen-policy-sums-opt', compiler=config['compiler'],
        options_result=HERE/'options_result.json', expected_objects_result=HERE/'expected_objects_result.json',
        reference=metadata/'reference.json', manifest=metadata/'manifest.json',
        constants=metadata/'constants.json')
    recipe = dict(schema=1, triton_python=config['triton_python'],
                  inputs={key: entry(path) for key, path in paths.items()}, arm_rope_trig_compat=True)
    write(output/'machine-recipe.json', recipe)
    return entry(output/'machine-recipe.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    lock = json.loads((HERE/'dependencies.lock.json').read_text())
    checked = verify_dependencies(config, lock)
    if args.verify_only:
        print(f'Verified {len(checked)} pinned SDK files and three model constant libraries')
        return
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(config['library_path']),
               PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get('PYTHONPATH', '')]))
    commands = [
        [config['cmake'], '-S', str(compiler_tools()/'policy'), '-B', str(output/'policy'), '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER='+config['host_clang'],
         '-DCMAKE_CXX_COMPILER='+config['host_clangxx'],
         '-DCMAKE_MAKE_PROGRAM='+config['ninja'],
         '-DMLIR_DIR='+config['llvm_build']+'/lib/cmake/mlir',
         '-DLLVM_DIR='+config['llvm_build']+'/lib/cmake/llvm'],
        [config['cmake'], '--build', str(output/'policy'), '--parallel', str(args.jobs)],
        [sys.executable, str(ROOT/'qwen3/runtime/arm/build_arm_rope_trig_runtime.py'),
         '--hexagon-clang', config['hexagon_clang'], '--lld', config['lld'],
         '--output', str(output/'rope')]]
    record = dict(status='building_NOT_DSP', source_root=str(ROOT), config_sha256=sha(args.config),
                  dependency_lock_sha256=sha(HERE/'dependencies.lock.json'), commands=commands,
                  verified_dependencies=checked, DSP_calls=0)
    try:
        for index, command in enumerate(commands):
            print(f'Build step {index+1}/{len(commands)}', flush=True)
            with (output/f'build-{index+1}.log').open('xb') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        record.update(status='owned_policy_and_RoPE_built_NOT_DSP', recipe=bind_recipe(config, output),
            outputs={name: entry(output/path) for name, path in
                [('first', 'policy/qwen-policy-first-opt'), ('sums', 'policy/qwen-policy-sums-opt'),
                 ('rope', 'rope/trig_runtime.o')]})
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output/'build-result.json', record)
    print(json.dumps(dict(status=record['status'], recipe=record['recipe'])), flush=True)


if __name__ == '__main__':
    main()
