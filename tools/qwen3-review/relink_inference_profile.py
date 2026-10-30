"""Instrument an already compiled downstream model; never compile its graph.

This is a link-only diagnostic. It requires a pinned clean downstream source,
actual recorded primary/runtime objects, and a byte-identical uninstrumented
library replay. It does not validate or restore a complete compiler SDK build.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from qwen3.compiler_dependency import native_root, compiler_root


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'machine', 'config', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--mode', choices=('external', 'attention', 'guard'), default='external')
    a = p.parse_args()
    git = lambda *args: subprocess.check_output(['git', '-C', str(a.source), *args], text=True).strip()
    if (git('rev-parse', 'HEAD') != a.revision or git('status', '--porcelain')
            or not a.output.is_absolute() or a.output.exists()):
        raise ValueError('Pinned clean source and fresh diagnostic output required')
    sys.path[:0] = [str(a.source/'tools/qwen3'), str(a.source/'qwen3/runtime/arm'), str(a.source), str(compiler_root())]
    helpers = importlib.import_module('profile_model_memory')
    cfg = json.loads(a.config.read_text())
    read = lambda path: json.loads(Path(path).read_text())
    link = read(a.machine/'linked/result.json')
    caller = read(a.machine/'caller/result.json')
    objects = read(a.machine/'objects/result.json')
    dynamic = link['query_specialization'] is None
    if (link['query_specialization'] not in (None, 1) or link['output_bytes'] != 235488768
            or a.mode in ('attention', 'guard') and not dynamic):
        raise ValueError('Original q1 or bounded dynamic q2 model required')
    primary = helpers.checked(objects['objects'][0])
    baseline = helpers.checked(link['library'])
    original = helpers.linked_caller(caller, link)
    header = helpers.checked(caller['runtime_header'])
    command = list(link['command'])
    if sha(command[0]) != link['compiler_sha256'] or sha(cfg['lld']) != link['linker_sha256']:
        raise ValueError('Original wrapper compiler and linker required')
    if command.count(str(primary)) != 1 or command.count(str(original)) != 1 or command.count('-o') != 1:
        raise ValueError('One unchanged primary, original caller and output required')
    constants = read(a.source/'tools/qwen3/model-constants.lock.json')['libraries']
    constant_refs = []
    for row in constants:
        path = Path(cfg['constants_dir'])/row['name']
        if sha(path) != row['sha256'] or objects['objects'][row['index']]['sha256'] != row['input_object_sha256']:
            raise ValueError('Original model constants required')
        constant_refs.append(dict(row, path=str(path)))
    a.output.mkdir(parents=True)
    record = dict(status='linking_diagnostic_NOT_device_acceptance', source_root=str(a.source),
        source_revision=a.revision, source_sha256=sha(__file__), mode=a.mode,
        dependency_scope='actual_link_inputs_and_exact_baseline_replay_NOT_full_compiler_build',
        complete_compiler_SDK_rebuild_claimed=False, new_model_PRIMARY_compilations=0, DSP_calls=0,
        model_object_unchanged=True, model_graph_changed=False, power_votes_changed=False,
        original_link_receipt=dict(path=str(a.machine/'linked/result.json'), sha256=sha(a.machine/'linked/result.json')),
        model_object=dict(path=str(primary), sha256=sha(primary)), baseline_library=link['library'],
        original_wrapper=dict(path=str(original), sha256=sha(original)), constants=constant_refs)
    save = lambda: (a.output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(cfg['library_path']))
    def run(name, argv):
        start = time.monotonic()
        proc = subprocess.run(argv, capture_output=True, env=env, timeout=180)
        (a.output/(name+'.stdout.txt')).write_bytes(proc.stdout)
        (a.output/(name+'.stderr.txt')).write_bytes(proc.stderr)
        record[name] = dict(command=argv, returncode=proc.returncode, seconds=time.monotonic()-start)
        save()
        proc.check_returncode()
    save()
    try:
        replay = a.output/'uninstrumented-replay.so'
        argv = list(command); argv[argv.index('-o')+1] = str(replay)
        run('baseline_replay', argv)
        if sha(replay) != sha(baseline):
            raise ValueError('Exact uninstrumented library replay failed')
        record['uninstrumented_replay_byte_identical'] = True
        (a.output/header.name).write_bytes(header.read_bytes())
        if 'silu_effects' in caller:
            guard = helpers.checked(caller['silu_effects']['guard_header'])
            (a.output/guard.name).write_bytes(guard.read_bytes())
        wrapper = a.output/original.name
        wrapper.write_text(helpers.instrument(original.read_text()))
        runtime_name = 'model_integer_timeline.c' if a.mode == 'attention' else 'model_external_profile.c'
        runtime = native_root()/'runtime/profile'/runtime_name
        memory_runtime = runtime.with_name('model_memory_profile.c')
        (a.output/memory_runtime.name).write_bytes(memory_runtime.read_bytes())
        if a.mode == 'guard':
            runtime = native_root()/'diagnostics/guard_profile.c'
        copied = a.output/runtime.name; copied.write_bytes(runtime.read_bytes())
        tools = Path(cfg['hexagon_tools'])/'bin'
        preserved = None if a.mode != 'external' else helpers.preserve_wrapped_memcpy(
            command, tools/'llvm-nm', tools/'hexagon-llvm-objcopy', a.output)
        obj = a.output/'profile.o'
        compile_args = [str(tools/'hexagon-clang'), '-O2', '-Wall', '-Wextra', '-Werror',
            '-fPIC', '-fvisibility=hidden', '-fno-builtin', '-mv73', '-mhvx', '-G0',
            '-I'+str(Path(cfg['hexagon_sdk'])/'incs'), '-I'+str(Path(cfg['hexagon_sdk'])/'incs/stddef'),
            '-c', str(copied), '-o', str(obj)]
        if preserved:
            compile_args.insert(1, '-DQWEN_PROFILE_MEMCPY_TARGET=qwen_profile_preserved_memcpy')
        if dynamic:
            compile_args.insert(1, '-DQWEN_PROFILE_MULTI_INT8')
        if a.mode == 'guard':
            symbols = ['_mlir_ciface_qwen_q2_qk_positive_zero_extent']
        elif a.mode == 'attention':
            compile_args.insert(1, '-DQWEN_TIMELINE_ATTENTION_PHASES')
            symbols = ['_mlir_ciface___qwen_i8_multi_matmul', '_mlir_ciface___hexagon_softmax_exp_compat']
        else:
            compile_args.insert(1, '-DQWEN_PROFILE_ALLOCATIONS')
            symbols = ['memcpy', 'memset', '__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes',
                '__hexagon_sqrtf', 'expf', '_mlir_ciface___hexagon_silu_exp_compat',
                '_mlir_ciface___hexagon_softmax_exp_compat', '_mlir_ciface___hexagon_arm_sin4_lane',
                '_mlir_ciface___hexagon_arm_cos4_lane',
                '_mlir_ciface___qwen_i8_multi_matmul' if dynamic else '_mlir_ciface___qwen_i8_matmul',
                '__hexagon_divdi3', '__hexagon_udivdi3', 'malloc', 'calloc', 'posix_memalign', 'free']
        run('profiler_compile', compile_args)
        library = a.output/'memory-profile.so'
        command[command.index(str(original))] = str(wrapper)
        command[command.index('-o')+1] = str(library)
        command[1:1] = [str(obj)]+['-Wl,--wrap='+name for name in symbols if '-Wl,--wrap='+name not in command]
        run('diagnostic_link', command)
        undefined = subprocess.check_output([str(tools/'llvm-nm'), '-u', str(library)], text=True)
        if any(line.split()[-1].startswith('__real_') for line in undefined.splitlines() if line.split()):
            raise ValueError('Unresolved original profiler callee')
        relocations = subprocess.check_output([str(tools/'llvm-readobj'), '--relocations', str(library)], text=True)
        local_names = ['qwen_memory_profile_'+name for name in ('begin', 'end', 'query')]
        local_names += ['__wrap_'+name for name in symbols]
        if any(name in relocations for name in local_names):
            raise ValueError('Diagnostic state and wrappers must bind locally')
        if git('rev-parse', 'HEAD') != a.revision or git('status', '--porcelain'):
            raise ValueError('Downstream source changed during diagnostic build')
        record.update(status='diagnostic_model_memory_library_built_NOT_device_acceptance',
            library=dict(path=str(library), sha256=sha(library)), input_bytes=link['input_bytes'], output_bytes=link['output_bytes'],
            external_profile_symbols={} if a.mode == 'attention' else {str(i+1):s for i,s in enumerate(symbols)},
            external_profile_kinds=[] if a.mode == 'attention' else list(range(1,len(symbols)+1)),
            required_profile_kinds=[] if a.mode == 'attention' else [1,2,3,10],
            integer_call_timeline=a.mode == 'attention', attention_phase_timeline=a.mode == 'attention')
        if dynamic:
            record.update(query_bounds=[2,32], input_bytes_are_maximum_capacity=True)
        if a.mode == 'guard':
            record.update(external_profile_symbols={'13':'_mlir_ciface_qwen_q2_qk_positive_zero_extent'},
                external_profile_kinds=[13], required_profile_kinds=[13])
    except BaseException as exc:
        record.update(status='failed', error=repr(exc)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
