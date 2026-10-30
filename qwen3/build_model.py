"""Build model callers and libraries from explicit source-build outputs.

Callers contain only the runtime buffer entry. No saved reference wrapper,
standalone main, installed Triton Python package or historical link receipt is
needed to generate the C ABI.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .bundle import sha, write
from .compiler_dependency import native_root

ROOT = Path(__file__).resolve().parents[1]


def caller_source(contract):
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import generate_buffer_entry, layout_for
    from hexagon_caller_byte_copy import rewrite_caller
    from hexagon_silu_helper_effects import guard_wrapper

    bounds = tuple(contract['variables'].get('query', ()))
    if set(contract['variables']) != {'query'} or bounds not in ((1, 1), (2, 32)):
        raise ValueError('Expected the supported q1 or q2..32 model contract')
    layout_for(contract, {'query': bounds[1]})
    inputs, outputs, arguments = [], [], []
    tensor_index = 0
    for row in contract['inputs']:
        if row['kind'] == 'scalar':
            inputs.append(SimpleNamespace(input_type='scalar', dtype='int64_t'))
            arguments.append('int64_t')
        else:
            dtype = {'float32': 'float', 'int64': 'int64_t'}[row['dtype']]
            rank = len(row['shape'])
            inputs.append(SimpleNamespace(input_type='tensor', dtype=dtype,
                                          rank=rank, idx=tensor_index))
            arguments.append(f'MemRefDescriptor<{dtype}, {rank}> *')
            tensor_index += 1
    if tensor_index != 60 or not 2 <= len(inputs)-tensor_index <= 8 or len(contract['outputs']) != 57:
        raise ValueError('Expected the complete 28-layer Qwen3 model ABI')
    fields = []
    for index, row in enumerate(contract['outputs']):
        if row['kind'] != 'tensor' or row['dtype'] != 'float32':
            raise ValueError('Expected FP32 logits and KV results')
        rank = len(row['shape'])
        outputs.append(SimpleNamespace(idx=index, dtype='float', rank=rank))
        fields.append(f'  MemRefDescriptor<float, {rank}> r{index};')
    entry = '_mlir_ciface_Qwen3DynamicBoundProbe'
    wrapper = SimpleNamespace(func_name=entry, iterations=1, release_returned_buffers=True,
                              input_profs=inputs, output_profs=outputs)
    source = '''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "common.h"
#include "tensor.h"
#include "hexagon_benchmark.h"
#include "CRunnerUtils.cpp"
// The SDK loader requires this symbol even for buffer-only libraries.
// Inference enters hexagon_forward_buffers; file-based standalone mode is unsupported.
int main() { return 2; }
'''
    source += f'extern "C" void {entry}(void *, '+', '.join(arguments)+');\n'
    source += 'struct FuncResult {\n'+'\n'.join(fields)+'\n};\n'
    source += '''#pragma clang fp contract(off)
#define QWEN_INCLUDE_SOFTMAX_EXP_COMPAT 1
#include "qwen3_sleef_exp_candidate.h"
'''
    for name in ('__hexagon_silu_exp_compat', '_mlir_ciface___hexagon_silu_exp_compat',
                 '__hexagon_softmax_exp_compat', '_mlir_ciface___hexagon_softmax_exp_compat'):
        source += f'extern "C" float {name}(float value) {{ return qwen3_sleef_exp_candidate(value); }}\n'
    source += generate_buffer_entry(wrapper, contract, symbol='hexagon_forward_buffers')
    return rewrite_caller(guard_wrapper(source))


def link_command(config, compiler_source, build, profile, caller, model_object,
                 constants, destination):
    """Explicit SDK link; every model/runtime object comes from this build."""
    if profile not in ('q1', 'q2'):
        raise ValueError('Unknown model profile')
    sdk, tools = Path(config['hexagon_sdk']), Path(config['hexagon_tools'])
    runtime = native_root()/'runtime'
    includes = [runtime/'dsp-include', runtime, sdk/'incs', sdk/'incs/stddef',
                sdk/'rtos/qurt/computev73/include/qurt', sdk/'rtos/qurt/computev73/include/posix']
    helpers = ['global-hvx-copy', profile+'-qk-guard',
               'q1-pv-guard' if profile == 'q1' else 'q2-zero-guard',
               'caller-byte-copy', 'preserved-memcpy', 'trig_runtime',
               'packed-int8-runtime' if profile == 'q1' else 'packed-multi-i8-runtime',
               'guarded-copy-runtime']
    command = [str(tools/'bin/hexagon-clang++'), '-O3', '-mv73', '-mhvx',
               '--ld-path='+str(Path(config['lld']).absolute()),
               '-Wl,-z,separate-loadable-segments', '-mlong-calls', '-nostdlib++']
    command += ['-I'+str(path) for path in includes]
    command += [str(caller), str(model_object)]
    command += [str(build/'runtime'/(name+'.o')) for name in helpers]
    command += ['-L'+str(sdk/'libs/qhl_hvx/prebuilt/hexagon_toolv87_v73'),
                '-L'+str(Path(config['hexkl'])/'lib/hexagon_toolv19_v73'),
                '-lqhmath_hvx', '-lhexkl_micro', '-shared', '-fPIC', '-G0',
                '-L'+str(constants)]
    command += ['-l_mlir_ciface_Qwen3DynamicBoundProbe-consts-'+str(i) for i in (1, 2, 3)]
    command += [str(runtime/'aligned_alloc_shim.cpp'),
                str(tools/'target/hexagon/lib/v73/G0/pic/libc++.a'),
                str(tools/'target/hexagon/lib/v73/G0/pic/libc++abi.a'),
                '-Wl,--wrap=memcpy', '-Wl,--wrap=memrefCopy', '-o', str(destination)]
    return command


def lower_capture(producer_path, producer_sha256, context, hf, catalog, cache):
    """Bind real immutable weights to the captured graph; never run a CPU forward."""
    from restore_qwen_symbolic_graph_carrier import restore_carrier
    from capture_qwen_symbolic_graph_carrier import CONTEXT_SHA
    from qwen3_generate_compile_common import load_model
    from hexagon_carrier_parameter_binding import CarrierParameterBinding
    from hexagon_captured_linalg_cache import CapturedLinalgCache
    from hexagon_arm_dynamic_catalog import select_actual_capture
    from hexagon_qwen_shape_policy import query_bounds
    from qcom_hexagon_backend.backend.hexagon_captured_buffer_contract import (
        scalar_policy_from_tensor_contract, derive_captured_contract)
    import torch

    if sha(context) != CONTEXT_SHA:
        raise ValueError('Model context differs from the captured configuration')
    graph, _, producer = restore_carrier(producer_path, producer_sha256, trusted_local=True)
    model, _, _ = load_model(context, hf=hf)
    binding = CarrierParameterBinding(model, producer)
    selected = select_actual_capture(catalog, producer['graph_identity'], binding)
    cache.mkdir(parents=True, exist_ok=True)
    frontend = CapturedLinalgCache(cache).get_or_lower(graph, binding, name='Qwen3DynamicBoundProbe')
    evidence = json.loads((Path(frontend['directory'])/'lowering.json').read_text())
    placeholders = [node for node in graph.graph.nodes if node.op == 'placeholder']
    tensors = [placeholders[i].meta['val'] for i in binding.retained
               if isinstance(placeholders[i].meta['val'], torch.Tensor)]
    if len(evidence['signatures']) != 1 or len(tensors) != 60:
        raise ValueError('One complete captured model signature required')
    bounds = query_bounds(tensors[0].shape[1])
    scalar_policy = scalar_policy_from_tensor_contract(graph, binding.retained,
        [row for row in selected['contract']['inputs'] if row['kind'] == 'tensor'],
        variables={'query': bounds}, allowed_literals={1, 1024})
    contract = json.loads(json.dumps(derive_captured_contract(graph, binding.retained,
        evidence['signatures'][0], variables={'query': bounds}, scalar_bindings=scalar_policy)))
    if selected['contract'] != contract:
        raise ValueError('Fresh frontend ABI differs from the selected model catalog')
    profile = {(1, 1): 'q1', (2, 32): 'q2'}[tuple(contract['variables']['query'])]
    return profile, contract, frontend


def prepare_command(python, entry, build, profile, output):
    command = [str(python), str(ROOT/'qwen3/runtime/arm/prepare_captured_qwen_hexagon.py'),
               '--entry', str(entry), '--entry-sha256', sha(entry), '--output', str(output), '--arm-rope-trig']
    packed = 'packed-i8-optimizer' if profile == 'q1' else 'packed-multi-i8-optimizer'
    effects = 'silu-effects-optimizer' if profile == 'q1' else 'multi-silu-effects-optimizer'
    for option, name in [('attention-optimizer', 'qwen-policy-first-opt'),
                         ('sums-optimizer', 'qwen-policy-sums-opt'),
                         ('knorm-optimizer', 'qwen-policy-sums-opt'),
                         (packed, 'qwen-packed-i8-opt'), (effects, 'qwen-silu-effects-opt')]:
        path = build/'policy'/name
        command += ['--'+option, str(path), '--'+option+'-sha256', sha(path)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'build', 'producer', 'context', 'hf', 'catalog', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--producer-sha256', required=True)
    parser.add_argument('--trusted-local-producer', action='store_true')
    parser.add_argument('--compiler-source', type=Path, required=True)
    parser.add_argument('--frontend-only', action='store_true')
    args = parser.parse_args()
    if not args.trusted_local_producer:
        parser.error('Only explicitly trusted local graph captures may be restored')
    source, build, output = args.compiler_source.resolve(), args.build.resolve(), args.output.resolve()
    from .build import verify_products
    compiler_build = verify_products(build, source, args.config)
    sys.path[:0] = [str(source), str(ROOT/'qwen3/runtime/arm')]
    config = json.loads(args.config.read_text())
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(source),
        str(ROOT/'qwen3/runtime/arm')]), LD_LIBRARY_PATH=os.pathsep.join(config.get('library_path', [])))
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='building', producer=dict(path=str(args.producer.resolve()),
        sha256=args.producer_sha256), started_unix=time.time(), commands=[],
        CPU_model_forward=False, DSP_calls=0, numerical_acceptance=False)
    record['builder_sha256'] = sha(Path(__file__))
    record['compiler_build_sha256'] = sha(build/'build.json')
    record['compiler_products'] = compiler_build['products']
    def run(stage, command):
        row = dict(stage=stage, argv=command, started_unix=time.time())
        record['commands'].append(row)
        write(output/'result.json', record)
        with (output/(stage+'.log')).open('w') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode, completed_unix=time.time())
        write(output/'result.json', record)
        result.check_returncode()
    try:
        profile, contract, frontend = lower_capture(args.producer, args.producer_sha256,
            args.context, args.hf, json.loads(args.catalog.read_text()), build/'frontend')
        record.update(profile=profile, contract=contract, frontend=frontend)
        write(output/'result.json', record)
        if args.frontend_only:
            record['status'] = 'frontend_built_NOT_model_code'
            return
        python = config.get('python', sys.executable)
        run('prepare', prepare_command(python, build/'frontend'/(frontend['key']+'.json'),
                                       build, profile, output/'prepared'))
        run('compile', [str(python), str(ROOT/'qwen3/runtime/arm/compile_qwen3_artifact.py'),
            '--input', str(output/'prepared/hexagon-linalg.mlirbc'), '--output-bundle', str(output/'objects'),
            '--libtriton', str(build/'compiler/profiles'/profile/'libtriton.so'),
            '--backend-source', str(source), '--options-from', str(ROOT/'qwen3/config/compiler-options.json'),
            '--geninsert-policy', 'process', '--progress'])
        from qwen3_object_bundle import load_bundle
        metadata, objects = load_bundle(output/'objects')
        if metadata.get('codegen_policy') != dict(scope='isolated_process',
                llvm_option='hexagon-insert', enabled=False):
            raise ValueError('Model compilation did not use the required process codegen policy')
        record['codegen_policy'] = metadata['codegen_policy']
        if len(objects) != 4:
            raise ValueError('Expected main code and three constant objects')
        constants = output/'constants'
        constants.mkdir()
        record['constants'] = []
        for index, obj in enumerate(objects[1:], 1):
            target = constants/f'lib_mlir_ciface_Qwen3DynamicBoundProbe-consts-{index}.so'
            run('constant-'+str(index), [config['hexagon_tools']+'/bin/hexagon-clang++',
                '-O3', '-mv73', '-mhvx', '--ld-path='+config['lld'],
                '-Wl,-z,separate-loadable-segments', str(obj), '-shared', '-fPIC', '-G0', '-o', str(target)])
            record['constants'].append(dict(path=str(target), sha256=sha(target)))
        caller = output/'caller.cpp'
        caller.write_text(caller_source(contract))
        library = output/'model.so'
        run('link', link_command(config, source, build, profile, caller, objects[0], constants, library))
        run('loader-exports', [config['hexagon_tools']+'/bin/hexagon-nm', '-D', '--defined-only', str(library)])
        exports = {line.split()[-1] for line in (output/'loader-exports.log').read_text().splitlines()
                   if line.split()}
        if not {'main', 'hexagon_forward_buffers'} <= exports:
            raise ValueError('DSP loader and buffer entry exports are required')
        if sha(Path(__file__)) != record['builder_sha256']:
            raise ValueError('Model builder changed during compilation; rerun with stable source')
        verify_products(build, source, args.config)
        record.update(status='source_model_built_NOT_DSP_qualified',
            caller=dict(path=str(caller), sha256=sha(caller)),
            library=dict(path=str(library), sha256=sha(library)))
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['completed_unix'] = time.time()
        write(output/'result.json', record)


if __name__ == '__main__':
    main()
