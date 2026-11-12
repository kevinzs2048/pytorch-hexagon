"""Build model callers and libraries from explicit source-build outputs.

Callers contain only the runtime buffer entry. No saved reference wrapper,
standalone main, installed Triton Python package or historical link receipt is
needed to generate the C ABI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .bundle import sha, write
from models.qwen3.paths import native_root

ROOT = Path(__file__).resolve().parents[2]


def constant_symbols(paths, readobj, environment):
    """Identify immutable globals by bytes, independently of compiler numbering."""
    result = {}
    for path in paths:
        metadata = json.loads(subprocess.check_output(
            [str(readobj), '--elf-output-style=JSON', '--sections', '--symbols', str(path)],
            env=environment))[0]
        if metadata['FileSummary']['Format'] != 'elf32-hexagon':
            raise ValueError('Expected Hexagon constant objects')
        sections = {row['Section']['Index']: row['Section'] for row in metadata['Sections']}
        with path.open('rb') as stream:
            for row in metadata['Symbols']:
                symbol = row['Symbol']
                if symbol['Binding']['Name'] == 'Local' or symbol['Section']['Name'] == 'Undefined':
                    continue
                section = sections.get(symbol['Section']['Value'])
                name, size = symbol['Name']['Name'], symbol['Size']
                if (symbol['Binding']['Name'] != 'Global' or symbol['Type']['Name'] != 'Object'
                        or section is None or section['Type']['Name'] != 'SHT_PROGBITS'
                        or size <= 0 or name in result or symbol['Value'] < 0
                        or symbol['Value'] + size > section['Size']):
                    raise ValueError('Unexpected constant definition: ' + name)
                stream.seek(section['Offset'] + symbol['Value'])
                digest = hashlib.sha256()
                remaining = size
                while remaining:
                    chunk = stream.read(min(remaining, 1024*1024))
                    if not chunk:
                        raise ValueError('Truncated constant object')
                    digest.update(chunk)
                    remaining -= len(chunk)
                result[name] = dict(bytes=size, sha256=digest.hexdigest())
    if not result:
        raise ValueError('Constant object set is empty')
    return result


# Fused-graph passes per domain, in the order of one policy invocation (the
# order the accepted candidates were prepared in). Counts are per domain.
FUSED_GRAPH_PASSES = (
    ('int8_dequant', '--qwen-fuse-i8-dequant=expected-count=197 expected-residual=56 expected-product=28',
     ('initial', 'prefill', 'decode')),
    ('parallel_rope', '--qwen-fuse-parallel-rope=expected-count=56', ('initial', 'prefill', 'decode')),
    # Decode's single-row norms are not loops; it keeps the in-graph norm.
    ('head_norm_rope', '--qwen-fuse-head-norm-rope=expected-count=56', ('initial', 'prefill')),
    ('parallel_trig', '--qwen-fuse-parallel-trig=expected-count=2', ('initial', 'prefill')),
    ('causal_mask', '--qwen-fuse-causal-mask=expected-count=1', ('prefill',)),
    ('kv_storage', '--qwen-fp16-kv-storage=expected-calls=28', ('initial', 'prefill', 'decode')),
    ('embedding_rows', '--qwen-embedding-rows=expected-count=1', ('initial_single', 'initial', 'prefill', 'decode')),
)
FUSED_GRAPH_KEYS = {'int8_dequant', 'parallel_rope', 'head_norm_rope', 'parallel_trig', 'causal_mask',
                    'kv_storage', 'kv_dma', 'embedding_rows',
                    'q8_0_projections'}


def fused_graph_passes(name, graph):
    """Policy flags of the selected fused-graph features for one domain. With
    Q8_0 projections every domain fuses its INT8 dequantization first."""
    passes = []
    for key, flag, domains in FUSED_GRAPH_PASSES:
        selected = graph.get(key) not in (None, False)
        if key == 'int8_dequant' and graph.get('q8_0_projections'):
            domains = domains + ('initial_single',)
        if selected and name in domains:
            passes.append(flag)
    return passes


def prepare_resident(name, entry, build, output, environment, integer_weights="plain",
                     silu_effects=False, activation_reuse=False, parallel_silu=False,
                     vectorize_activation=False, batch_arm_rope=False, graph=None):
    """Stream a pinned actual Linalg capture through the effective-length policies."""
    sys.path.insert(0, str(ROOT/'models/qwen3/runtime/arm'))
    from apply_dynamic_bound_qk_policy import FunctionPolicyPatch
    from streamed_mlir_interchange import stream_mlir_pipeline
    from hexagon_qwen_numerical_policy import policy_counts
    from .runtime.arm.prepare_qwen_resident_bindings import PLANE

    source = Path(entry['linalg']).resolve()
    if sha(source) != entry['linalg_sha256']:
        raise ValueError('Captured Linalg changed: ' + name)
    output.mkdir()
    tools = build/'policy'
    reader = tools/'qwen-policy-sums-opt'
    resident = name in ('prefill', 'decode')
    single = name in ('initial_single', 'decode')
    planes = {}
    for i, row in enumerate(entry['mapping']['inputs']):
        match = PLANE.fullmatch(row['source'].replace("L['kwargs'][", 'L['))
        if match:
            planes[i] = 2*int(match[1]) + (match[2] == 'values')
    if sorted(planes.values()) != (list(range(56)) if resident else []):
        raise ValueError('Captured cache provenance differs: ' + name)

    class Source:
        def print(self, *, file, binary):
            if not binary:
                raise ValueError('Complete binary text interchange required')
            patch = FunctionPolicyPatch(file.write, function_name=entry['function'],
                effective_attention='decode' if single else 'prefill',
                resident_planes=planes if resident else None, silu=True,
                softmax_exp=True, bounded_rmsnorm=True, arm_rope_trig=True)
            with (output/'read.log').open('wb') as log:
                process = subprocess.Popen([str(reader), str(source), '-o', '-'],
                    stdout=subprocess.PIPE, stderr=log, env=environment)
                try:
                    while chunk := process.stdout.read(65536):
                        patch.write(chunk)
                    patch.finish()
                    if process.wait() != 0:
                        raise RuntimeError('Captured Linalg reader failed')
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    process.stdout.close()

    target = output/'hexagon-linalg.mlirbc'
    common = ['-', '--verify-each', '--emit-bytecode']
    commands = [[str(tools/'qwen-policy-first-opt'), *common,
        '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum{fp32-silu-exp-compat=true}))', '-o', '-'],
        [str(reader), *common, '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum))', '-o', '-']]
    if name == 'decode':
        # The admitted decoder normalizes RMSNorm/K-norm with the sums tool
        # before first-stage attention/SiLU. Preserve that preparation order.
        commands = [commands[1], commands[1], *commands]
    if resident:
        commands.append([str(tools/'qwen-resident-kv-opt'), *common,
                         '--qwen-resident-kv=expected-count=56', '-o', '-'])
    commands.append([str(tools/'qwen-packed-i8-opt'), *common,
                    '--qwen-packed-i8-matmul=multi-row=true expected-count=197', '-o', str(target)])
    record = dict(domain=name, input_sha256=entry['linalg_sha256'],
        interchange=stream_mlir_pipeline(Source(), commands,
            archive_path=output/'adapted-source.mlir.gz', log_path=output/'policy.log',
            timeout=1800, env=environment))
    inspection = output/'inspection.mlir'
    with (output/'inspection.log').open('wb') as log:
        subprocess.run([str(tools/'qwen-packed-i8-opt'), str(target), '--verify-each',
            '--mlir-elide-elementsattrs-if-larger=16', '--mlir-elide-resource-strings-if-larger=16',
            '-o', str(inspection)], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    text = inspection.read_text()
    counts = policy_counts(text)
    counts['effective_decode_order'] = text.count('hexagon.fp32_effective_decode_order')
    counts['effective_PV'] = text.count('hexagon.fp32_effective_prefill_pv_order')
    counts['resident_key_head_major'] = text.count('hexagon.resident_key_head_major')
    expected = dict(norm_rows=113, softmax_rows=28, silu_rows=28,
                    silu_exponents=28, softmax_exponents=28, rope_trig_attributes=3)
    expected.update(dict(effective_decode_order=56) if single else
                    dict(attention_contractions=28, effective_PV=28, unconverted_fp32_attention=0))
    expected['resident_key_head_major'] = 28 if resident else 0
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError('Incomplete resident numerical policy coverage: ' + repr(counts))
    strategy = entry.get('attention', 'fp32-reference')
    if strategy != 'fp32-reference':
        allowed = ('initial', 'prefill', 'decode')
        if strategy != 'fp32-ordered-workers-v1' or name not in allowed:
            raise ValueError('Unsupported attention execution strategy/domain: ' + name)
        fused = output/'fused.mlirbc'
        with (output/'fusion.log').open('wb') as log:
            subprocess.run([str(tools/'qwen-fused-attention-opt'), str(target),
                '--qwen-fused-attention=expected-count=28 implementation='+strategy, '--canonicalize',
                '--verify-each', '--emit-bytecode', '-o', str(fused)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        fused.replace(target)
    if integer_weights not in ('plain', 'hvx-i8-prepacked-v1'):
        raise ValueError('Unsupported integer weight layout: ' + integer_weights)
    if integer_weights == 'hvx-i8-prepacked-v1':
        packed = output/'prepacked.mlirbc'
        with (output/'prepack.log').open('wb') as log:
            subprocess.run([str(tools/'qwen-packed-i8-opt'), str(target),
                '--hexagon-prepack-i8-weights=expected-count=197', '--canonicalize',
                '--verify-each', '--emit-bytecode', '-o', str(packed)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        packed.replace(target)
    if silu_effects:
        annotated = output/'silu-effects.mlirbc'
        multi = 'true' if name in ('initial', 'prefill') else 'false'
        with (output/'silu-effects.log').open('wb') as log:
            subprocess.run([str(tools/'qwen-silu-effects-opt'), str(target),
                '--qwen-silu-effects=resident-graphs=true expected-count=28 multi-row='+multi,
                '--verify-each', '--emit-bytecode', '-o', str(annotated)], env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        annotated.replace(target)
    if activation_reuse:
        shared = output/'activation-reuse.mlirbc'
        with (output/'activation-reuse.log').open('wb') as log:
            subprocess.run([str(tools/'qwen-packed-i8-opt'), str(target),
                '--qwen-reuse-activation=expected-count=84', '--cse',
                '--verify-each', '--emit-bytecode', '-o', str(shared)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        shared.replace(target)
    if parallel_silu:
        parallel = output/'parallel-silu.mlirbc'
        with (output/'parallel-silu.log').open('wb') as log:
            subprocess.run([str(tools/'qwen-silu-effects-opt'), str(target),
                '--qwen-parallel-silu=expected-count=28', '--verify-each',
                '--emit-bytecode', '-o', str(parallel)], env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        parallel.replace(target)
    if vectorize_activation:
        vectorized = output/'vectorized-activation.mlirbc'
        with (output/'vectorized-activation.log').open('wb') as log:
            subprocess.run([str(tools/'qwen-packed-i8-opt'), str(target),
                '--qwen-vectorize-activation=expected-count=113', '--verify-each',
                '--emit-bytecode', '-o', str(vectorized)], env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        vectorized.replace(target)
    if batch_arm_rope:
        batched = output/'batched-arm-rope.mlirbc'
        with (output/'batched-arm-rope.log').open('wb') as log:
            subprocess.run([str(tools/'qwen-silu-effects-opt'), str(target),
                '--qwen-batch-arm-rope=expected-count=2', '--verify-each',
                '--emit-bytecode', '-o', str(batched)], env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        batched.replace(target)
    graph = graph or {}
    passes = fused_graph_passes(name, graph)
    if passes or graph.get('q8_0_projections'):
        fused = output/'fused-graph.mlirbc'
        log = output/'fused-graph.log'
        if graph.get('q8_0_projections'):
            from .q8_0.builder import rewrite_domain
            record['q8_0_projections'] = rewrite_domain(tools/'qwen-packed-i8-opt', target, fused, name,
                                                        output/'q8_0-dump', environment, log, passes)
        else:
            with log.open('wb') as stream:
                subprocess.run([str(tools/'qwen-packed-i8-opt'), str(target), *passes, '--canonicalize',
                                '--verify-each', '--emit-bytecode', '-o', str(fused)],
                               env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        fused.replace(target)
    record['fused_graph_passes'] = passes
    record['batch_arm_rope'] = batch_arm_rope
    record['vectorize_activation'] = vectorize_activation
    record['parallel_silu'] = parallel_silu
    record['activation_reuse'] = activation_reuse
    record['silu_effects'] = silu_effects
    record['integer_weights'] = integer_weights
    record['attention'] = strategy
    record.update(counts=counts, output=dict(path=str(target), sha256=sha(target)),
                  numerical_acceptance=False)
    write(output/'preparation.json', record)
    return record


def complete_ir_identity(path, reader, environment, log_path):
    """Hash complete verified IR without debug locations; never elide weights."""
    digest, size = hashlib.sha256(), 0
    with log_path.open('wb') as log:
        process = subprocess.Popen([str(reader), str(path), '--verify-each', '-o', '-'],
            env=environment, stdout=subprocess.PIPE, stderr=log)
        try:
            while chunk := process.stdout.read(1024*1024):
                digest.update(chunk)
                size += len(chunk)
            if process.wait() != 0:
                raise RuntimeError('Complete IR comparison failed; inspect ' + str(log_path))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
    return dict(bytes=size, sha256=digest.hexdigest())


def link_resident(args):
    """Prepare captured domains, optionally compile, then link a resident session."""
    from .runtime.arm.prepare_qwen_resident_bindings import generate, DOMAINS
    from .runtime.arm.qwen3_object_bundle import load_bundle
    from .resident import load_session
    from .build import native_identity, verify_products
    from hexagon_runtime.compiler_dependency import compiler_root
    import shutil

    config = json.loads(args.config.read_text())
    source = args.compiler_source.resolve() if args.compiler_source else compiler_root()
    build = args.build.resolve()
    compiler_build = verify_products(build, source, args.config)
    plan = json.loads(args.resident_spec.read_text())
    temporary_buffer_bytes = plan.get('temporary_buffer_bytes', 0)
    if type(temporary_buffer_bytes) is not int or not 0 <= temporary_buffer_bytes <= 128 * 1024 * 1024:
        raise ValueError('temporary_buffer_bytes must be an integer in [0, 134217728]')
    compile_resident = args.compile_resident
    if compile_resident and any('linalg' not in entry for entry in plan['domains'].values()):
        raise ValueError('Resident compilation requires captured Linalg for every domain')
    benchmark_max_performance = plan.get('benchmark_max_performance', False)
    if type(benchmark_max_performance) is not bool:
        raise ValueError('benchmark_max_performance must be a boolean')
    silu_effects = plan.get('silu_effects', False)
    if type(silu_effects) is not bool:
        raise ValueError('silu_effects must be a boolean')
    activation_reuse = plan.get('activation_reuse', False)
    if type(activation_reuse) is not bool:
        raise ValueError('activation_reuse must be a boolean')
    vectorize_activation = plan.get('vectorize_activation', False)
    if type(vectorize_activation) is not bool:
        raise ValueError('vectorize_activation must be a boolean')
    if vectorize_activation and not activation_reuse:
        raise ValueError('vectorize_activation requires activation_reuse')
    parallel_silu = plan.get('parallel_silu', False)
    if type(parallel_silu) is not bool:
        raise ValueError('parallel_silu must be a boolean')
    batch_arm_rope = plan.get('batch_arm_rope', False)
    if type(batch_arm_rope) is not bool:
        raise ValueError('batch_arm_rope must be a boolean')
    row_minmax = plan.get('row_minmax', False)
    if type(row_minmax) is not bool:
        raise ValueError('row_minmax must be a boolean')
    integer_weights = plan.get('integer_weights', 'plain')
    if integer_weights not in ('plain', 'hvx-i8-prepacked-v1'):
        raise ValueError('Unsupported integer weight layout: ' + integer_weights)
    if (integer_weights != 'plain' or silu_effects or activation_reuse or parallel_silu or vectorize_activation or batch_arm_rope) and any(
            'linalg' not in entry for entry in plan['domains'].values()):
        raise ValueError('Selected numerical policies require captured Linalg for every domain')
    integer_threads = plan.get('integer_threads', 1)
    if integer_threads not in (1, 4):
        raise ValueError('Integer workers must be 1 or 4')
    if integer_threads == 4 and (integer_weights != 'hvx-i8-prepacked-v1' or not args.llama_source):
        raise ValueError('Four integer workers require prepacked weights and the pinned llama runtime')
    int8_kernels = plan.get('triton_int8_prefill')
    int8_inputs = {}
    if int8_kernels is not None:
        if (integer_weights != 'hvx-i8-prepacked-v1' or integer_threads != 4 or
                not isinstance(int8_kernels, dict) or set(int8_kernels) != {'pack', 'accumulate'}):
            raise ValueError('triton_int8_prefill requires prepacked weights, four workers and pack/accumulate objects')
        for kind, entry in int8_kernels.items():
            if not isinstance(entry, dict) or set(entry) != {'object', 'object_sha256', 'manifest_sha256'}:
                raise ValueError('Invalid INT8 kernel record: ' + kind)
            obj = Path(entry['object']).resolve(strict=True)
            manifest = obj.with_name('manifest.json')
            receipt = json.loads(manifest.read_text())
            if (sha(obj) != entry['object_sha256'] or sha(manifest) != entry['manifest_sha256'] or
                    receipt.get('object_sha256') != entry['object_sha256'] or
                    receipt.get('profile') != 'strict-int8-' + kind or
                    receipt.get('lwp', {}).get('enabled') is not False):
                raise ValueError('INT8 kernel identity/profile differs: ' + kind)
            int8_inputs[kind] = (obj, manifest)
            int8_kernels[kind] = dict(entry, object=str(obj))
    features = plan.get('fused_graph', {})
    if not isinstance(features, dict) or set(features) - FUSED_GRAPH_KEYS:
        raise ValueError('Unsupported fused_graph features')
    graph_inputs = {}

    def kernel_record(entry, profile, label):
        if not isinstance(entry, dict) or set(entry) != {'object', 'object_sha256', 'manifest_sha256'}:
            raise ValueError('Invalid kernel record: ' + label)
        obj = Path(entry['object']).resolve(strict=True)
        manifest = obj.with_name('manifest.json')
        receipt = json.loads(manifest.read_text())
        if (sha(obj) != entry['object_sha256'] or sha(manifest) != entry['manifest_sha256'] or
                receipt.get('object_sha256') != entry['object_sha256'] or receipt.get('profile') != profile or
                receipt.get('lwp', {}).get('enabled') is not False):
            raise ValueError('Kernel identity/profile differs: ' + label)
        graph_inputs[label] = (obj, manifest, entry)
        return dict(entry, object=str(obj))
    if features.get('int8_dequant') is not None:
        dequant = features['int8_dequant']
        if int8_kernels is None or not isinstance(dequant, dict) or set(dequant) != {'dequant', 'residual', 'product'}:
            raise ValueError('int8_dequant requires the INT8 HMX kernels and dequant/residual/product objects')
        features['int8_dequant'] = {kind: kernel_record(dequant[kind], 'strict-int8-dequant' +
                                                     ('' if kind == 'dequant' else '-' + kind), 'dequant-' + kind)
                                 for kind in ('dequant', 'residual', 'product')}
    if features.get('parallel_rope') is not None:
        features['parallel_rope'] = kernel_record(features['parallel_rope'], 'strict-rope-heads', 'rope')
    if features.get('head_norm_rope') is not None:
        if features.get('parallel_rope') is None:
            raise ValueError('head_norm_rope requires parallel_rope')
        features['head_norm_rope'] = kernel_record(features['head_norm_rope'], 'strict-rope-heads-norm', 'rope-norm')
    if features.get('parallel_trig') not in (None, True):
        raise ValueError('parallel_trig must be true when present')
    if features.get('causal_mask') is not None:
        features['causal_mask'] = kernel_record(features['causal_mask'], 'strict-causal-mask', 'mask')
    if features.get('kv_storage') not in (None, 'fp16'):
        raise ValueError('kv_storage must be fp16 when present')
    if features.get('embedding_rows') not in (None, True):
        raise ValueError('embedding_rows must be true when present')
    if features.get('kv_dma') not in (None, True) or (features.get('kv_dma') and features.get('kv_storage') != 'fp16'):
        raise ValueError('kv_dma requires kv_storage fp16')
    q8_0 = features.get('q8_0_projections')
    if q8_0 is not None:
        if (features.get('int8_dequant') is None or not isinstance(q8_0, dict) or set(q8_0) != {'hf_checkpoint'} or
                not args.llama_source or not compile_resident):
            raise ValueError('q8_0_projections requires int8_dequant, the HF checkpoint, '
                             '--llama-source and resident compilation')
        q8_0 = dict(hf_checkpoint=str(Path(q8_0['hf_checkpoint']).resolve(strict=True)))
        features['q8_0_projections'] = q8_0
    if features and not compile_resident:
        raise ValueError('fused_graph features require resident compilation')
    plan['fused_graph'] = features
    sdk_hmx_linker = None
    if int8_kernels is not None:
        path = Path(config['hexagon_tools'])/'bin/hexagon-link'
        sdk_hmx_linker = dict(path=str(path.resolve(strict=True)), sha256=sha(path))
    fused_domains = [name for name, entry in plan['domains'].items()
                     if entry.get('attention', 'fp32-reference') != 'fp32-reference']
    if any(name not in ('initial', 'prefill', 'decode') or
           plan['domains'][name]['attention'] != 'fp32-ordered-workers-v1'
           or 'linalg' not in plan['domains'][name] for name in fused_domains):
        raise ValueError('Attention strategy requires a captured compatible domain')
    strategies = {plan['domains'][name]['attention'] for name in fused_domains}
    if len(strategies) > 1:
        raise ValueError('Select one attention runtime strategy per session')
    attention_strategy = next(iter(strategies), 'fp32-reference')
    decode_kernel = plan.get('triton_decode')
    if decode_kernel is not None:
        if (attention_strategy != 'fp32-ordered-workers-v1' or
                'decode' not in fused_domains or not isinstance(decode_kernel, dict) or
                set(decode_kernel) != {'object', 'object_sha256', 'manifest_sha256'}):
            raise ValueError('triton_decode requires a fused decode domain and object/manifest hashes')
        decode_object = Path(decode_kernel['object']).resolve(strict=True)
        decode_manifest = decode_object.with_name('manifest.json')
        if (sha(decode_object) != decode_kernel['object_sha256'] or
                sha(decode_manifest) != decode_kernel['manifest_sha256']):
            raise ValueError('Triton decode object/manifest changed')
        decode_kernel = dict(decode_kernel, object=str(decode_object))
        plan['triton_decode'] = decode_kernel
    prefill_kernels = plan.get('triton_prefill')
    prefill_inputs = {}
    if prefill_kernels is not None:
        if (decode_kernel is None or not {'initial', 'prefill'}.issubset(fused_domains) or
                not isinstance(prefill_kernels, dict) or
                set(prefill_kernels) != {'pack', 'qk', 'softmax', 'pv'}):
            raise ValueError('triton_prefill requires fused initial/prefill, strict decode and all four stages')
        for kind, entry in prefill_kernels.items():
            if not isinstance(entry, dict) or set(entry) != {'object', 'object_sha256', 'manifest_sha256'}:
                raise ValueError('Invalid strict prefill kernel record: ' + kind)
            obj = Path(entry['object']).resolve(strict=True)
            manifest = obj.with_name('manifest.json')
            receipt = json.loads(manifest.read_text())
            if (sha(obj) != entry['object_sha256'] or sha(manifest) != entry['manifest_sha256'] or
                    receipt.get('object_sha256') != entry['object_sha256'] or
                    receipt.get('profile') != 'strict-prefill-' + kind or
                    receipt.get('lwp', {}).get('enabled') is not False):
                raise ValueError('Strict prefill kernel identity/profile differs: ' + kind)
            prefill_inputs[kind] = obj
            prefill_kernels[kind] = dict(entry, object=str(obj))
    hmx_prefill = plan.get('triton_hmx_prefill')
    if hmx_prefill is not None:
        if (decode_kernel is None or int8_kernels is None or prefill_kernels is not None or
                not {'initial', 'prefill'}.issubset(fused_domains) or
                not isinstance(hmx_prefill, dict) or
                set(hmx_prefill) != {'object', 'object_sha256', 'manifest_sha256'}):
            raise ValueError('HMX prefill requires fused initial/prefill, strict decode, INT8 HMX '
                             'and no strict prefill stages')
        hmx_object = Path(hmx_prefill['object']).resolve(strict=True)
        hmx_manifest = hmx_object.with_name('manifest.json')
        receipt = json.loads(hmx_manifest.read_text())
        if (sha(hmx_object) != hmx_prefill['object_sha256'] or
                sha(hmx_manifest) != hmx_prefill['manifest_sha256'] or
                receipt.get('object_sha256') != hmx_prefill['object_sha256'] or
                receipt.get('profile') != 'experimental-prefill-fp16' or
                receipt.get('raw_buffer_access') != 'dsp-load-store' or
                receipt.get('lwp', {}).get('enabled') is not False or
                any(receipt.get(key) is not True for key in
                    ('reuse_hmx_workspace', 'prepare_hmx_matmul', 'external_allocator'))):
            raise ValueError('HMX prefill kernel identity, preparation or allocator contract differs')
        hmx_prefill = dict(hmx_prefill, object=str(hmx_object))
        plan['triton_hmx_prefill'] = hmx_prefill
    hmx_decode = plan.get('triton_hmx_decode')
    if hmx_decode is not None:
        if (hmx_prefill is None or not isinstance(hmx_decode, dict) or
                set(hmx_decode) != {'object', 'object_sha256', 'manifest_sha256', 'min_kv'} or
                type(hmx_decode['min_kv']) is not int or not 1 <= hmx_decode['min_kv'] <= 1024):
            raise ValueError('HMX decode requires HMX prefill, kernel hashes and min_kv in 1..1024')
        shared_receipt = json.loads(hmx_manifest.read_text())
        if shared_receipt.get('prepared_hmx_slot_bytes') != 192 * 1024:
            raise ValueError('HMX decode shared runtime requires 192KiB preparation slots; rebuild prefill')
        hmx_decode_object = Path(hmx_decode['object']).resolve(strict=True)
        manifest = hmx_decode_object.with_name('manifest.json')
        receipt = json.loads(manifest.read_text())
        if (sha(hmx_decode_object) != hmx_decode['object_sha256'] or
                sha(manifest) != hmx_decode['manifest_sha256'] or
                receipt.get('object_sha256') != hmx_decode['object_sha256'] or
                receipt.get('profile') != 'experimental-decode-fp16' or
                receipt.get('entry_point') != 'attention_forward_q1_hmx_g2' or
                receipt.get('kv_heads_per_program') != 2 or
                receipt.get('prepared_hmx_slot_bytes') != 192 * 1024 or
                receipt.get('shared_runtime') != 'experimental-prefill-fp16' or
                receipt.get('contract', {}).get('query') != [1, 1] or
                receipt.get('raw_buffer_access') != 'dsp-load-store' or
                receipt.get('lwp', {}).get('enabled') is not False or
                any(receipt.get(key) is not True for key in
                    ('reuse_hmx_workspace', 'prepare_hmx_matmul', 'external_allocator'))):
            raise ValueError('HMX decode kernel identity, query or shared-runtime contract differs')
        hmx_decode = dict(hmx_decode, object=str(hmx_decode_object))
        plan['triton_hmx_decode'] = hmx_decode
    silu_kernel = plan.get('triton_silu')
    if silu_kernel is not None:
        if (not parallel_silu or not isinstance(silu_kernel, dict) or
                set(silu_kernel) != {'object', 'object_sha256', 'manifest_sha256'}):
            raise ValueError('triton_silu requires parallel_silu and object/manifest hashes')
        silu_object = Path(silu_kernel['object']).resolve(strict=True)
        silu_manifest = silu_object.with_name('manifest.json')
        receipt = json.loads(silu_manifest.read_text())
        if (sha(silu_object) != silu_kernel['object_sha256'] or
                sha(silu_manifest) != silu_kernel['manifest_sha256'] or
                receipt.get('profile') not in ('strict-silu', 'experimental-silu-fp32', 'llama-silu-fp32') or
                receipt.get('object_sha256') != silu_kernel['object_sha256'] or
                receipt.get('lwp', {}).get('enabled') is not False):
            raise ValueError('SiLU kernel identity/profile differs')
        silu_profile = receipt['profile']
        silu_kernel = dict(silu_kernel, object=str(silu_object))
        plan['triton_silu'] = silu_kernel
    # Attention kernels read the resident K/V in the storage they were compiled
    # for: FP16 kernels with FP16 storage only, and FP16 storage with FP16 kernels.
    attention_storage = [json.loads(Path(k['object']).with_name('manifest.json').read_text())
                         .get('contract', {}).get('storage')
                         for k in (hmx_prefill, decode_kernel, hmx_decode) if k is not None]
    if features.get('kv_storage') == 'fp16':
        if hmx_prefill is None or decode_kernel is None or any(s != 'fp16' for s in attention_storage):
            raise ValueError('FP16 KV storage requires HMX prefill and decode kernels compiled with --kv-fp16')
    elif 'fp16' in attention_storage:
        raise ValueError('Attention kernels compiled with --kv-fp16 require fused_graph kv_storage fp16')
    domain_defaults = dict(enableRowMinMax=row_minmax, scheduleFMAChains=False,
                           scheduleDequantLoops=False, parallelDequantLoops=False,
                           scheduleReductionLoops=False,
                           outlineHVXMemCpy=False)
    domain_options = {}
    for name, entry in plan['domains'].items():
        options = entry.get('compiler_options', {})
        if (not isinstance(options, dict) or set(options) - set(domain_defaults) or
                any(type(value) is not bool for value in options.values())):
            raise ValueError('Unsupported per-domain compiler options: ' + name)
        domain_options[name] = dict(domain_defaults, **options)
        if (domain_options[name]['parallelDequantLoops'] and
                not domain_options[name]['scheduleDequantLoops']):
            raise ValueError('Parallel dequant requires scheduled dequant loops: ' + name)
    parallel_dequant = [name for name in DOMAINS
                        if domain_options[name]['parallelDequantLoops']]
    # These runtimes split rows with __hexagon_parallel_for_v1. Without the
    # linked worker adapter they would bind to the compiler's serial fallback.
    pooled = [key for key in ('parallel_rope', 'head_norm_rope', 'parallel_trig', 'causal_mask', 'kv_storage',
                              'q8_0_projections')
              if features.get(key)]
    if pooled and not parallel_dequant:
        raise ValueError('fused_graph ' + ', '.join(pooled) + ' require the row-parallel worker adapter '
                         '(parallelDequantLoops in at least one domain)')
    needs_llama = bool(fused_domains) or integer_threads == 4 or parallel_silu or bool(parallel_dequant)
    if needs_llama != bool(args.llama_source):
        raise ValueError('Use --llama-source when selecting attention, integer, SiLU or dequant workers')
    # Reject mixed compiler products before replaying weight-sized preparation.
    def check_compiler(name, bundle):
        if (bundle['extension_sha256'] != compiler_build['products']['compiler/profiles/dynamic-kv/libtriton.so']
                or bundle.get('codegen_policy') != dict(scope='isolated_process',
                    llvm_option='hexagon-insert', enabled=False)):
            raise ValueError('Resident object compiler identity/policy differs: ' + name)
        compiled = bundle.get('effective_options', bundle.get('options', {}))
        expected = (True, 'True') if domain_options[name]['enableRowMinMax'] else (False, 'False')
        if compiled.get('enableRowMinMax', False) not in expected:
            raise ValueError('Resident row-minmax option differs: ' + name +
                             '; rebuild this domain with the selected setting')
        for key, enabled in domain_options[name].items():
            if compiled.get(key, False) not in (enabled, str(enabled)):
                raise ValueError('Resident compiler option differs: ' + name + '/' + key +
                                 '; rebuild this domain with the selected setting')
    bundles = {} if compile_resident else {
        name: load_bundle(Path(plan['domains'][name]['objects'])) for name in DOMAINS}
    for name, (bundle, _) in bundles.items():
        check_compiler(name, bundle)
    header = generate(plan)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tools, sdk = Path(config['hexagon_tools']), Path(config['hexagon_sdk'])
    runtime = native_root()/'runtime'
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(config.get('library_path', [])))
    record = dict(status='linking', commands=[], domains={}, numerical_acceptance=False,
                  spec_sha256=sha(args.resident_spec), builder_sha256=sha(Path(__file__)),
                  configuration_sha256=sha(args.config), native_sources=native_identity(),
                  compiler_build_sha256=sha(build/'build.json'))
    compiler_options = ROOT/'models/qwen3/config/compiler-options.json'
    record['compiler_overrides'] = {'enableRowMinMax': str(row_minmax)}
    record['domain_compiler_overrides'] = domain_options
    if compile_resident:
        record['compiler_options_sha256'] = sha(compiler_options)

    def run(stage, command):
        argv = [str(arg) for arg in command]
        record['commands'].append(dict(stage=stage, argv=argv))
        with (output/(stage+'.log')).open('w') as log:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        attention_archive = []
        if needs_llama:
            attention_build = output/'attention'
            run('attention-configure', [config.get('cmake', 'cmake'), '-G', 'Ninja',
                '-S', runtime/'attention', '-B', attention_build,
                '-DLLAMA_SOURCE='+str(args.llama_source.resolve()),
                '-DHEXAGON_SDK='+str(sdk), '-DHEXAGON_TOOLS='+str(tools),
                *(['-DTRITON_DECODE_OBJECT='+str(decode_object)] if decode_kernel else []),
                *['-DTRITON_STRICT_PREFILL_'+kind.upper()+'_OBJECT='+str(obj)
                  for kind, obj in prefill_inputs.items()],
                *(['-DTRITON_PREFILL_OBJECT='+str(hmx_object),
                   '-DTRITON_PREFILL_POOL_BYTES=1048576', '-DSHARE_HMX_RUNTIME=ON']
                  if hmx_prefill else []),
                *(['-DTRITON_HMX_DECODE_OBJECT='+str(hmx_decode_object),
                   '-DTRITON_HMX_DECODE_MIN_KV='+str(hmx_decode['min_kv'])]
                  if hmx_decode else []),
                *(['-DTRITON_SILU_OBJECT='+str(silu_object)] if silu_kernel else []),
                *(['-DKV_FP16=ON'] if features.get('kv_storage') == 'fp16' else []),
                *(['-DKV_DMA=ON'] if features.get('kv_dma') else [])])
            target, archive_name = {
                'fp32-reference': ('qwen-workers-static', 'libqwen_workers.a'),
                'fp32-ordered-workers-v1': ('qwen-ordered-fp32-attention-static', 'libqwen_ordered_fp32_attention.a'),
            }[attention_strategy]
            if decode_kernel:
                target, archive_name = ('qwen-triton-strict-attention-static',
                                        'libqwen_triton_strict_attention.a')
            if hmx_prefill:
                target, archive_name = ('qwen-triton-experimental-attention-static',
                                        'libqwen_triton_experimental_attention.a')
            run('attention-build', [config.get('cmake', 'cmake'), '--build', attention_build,
                                   '--target', target, '--parallel', '4'])
            archive = attention_build/archive_name
            attention_archive = [archive]
            if parallel_silu:
                silu_target, silu_name = (('qwen-triton-silu-static', 'libqwen_triton_silu.a')
                    if silu_kernel else ('qwen-ordered-silu-static', 'libqwen_ordered_silu.a'))
                run('silu-build', [config.get('cmake', 'cmake'), '--build', attention_build,
                                  '--target', silu_target, '--parallel', '4'])
                silu_archive = attention_build/silu_name
                attention_archive.append(silu_archive)
                record['parallel_silu'] = dict(domains=list(DOMAINS),
                                               archive_sha256=sha(silu_archive))
                if silu_kernel:
                    record['parallel_silu']['triton_silu'] = silu_kernel
            shutil.copyfile(args.llama_source/'LICENSE', output/'LICENSE.llama.cpp')
            revision = subprocess.check_output(['git', '-C', str(args.llama_source),
                'rev-parse', 'HEAD'], text=True).strip()
            record['llama_runtime'] = dict(source_revision=revision,
                archive_sha256=sha(archive), integer_threads=integer_threads)
            if fused_domains:
                record['attention'] = dict(implementation=attention_strategy,
                    domains=fused_domains, archive_sha256=sha(archive),
                    power_policy='benchmark-max-vote' if benchmark_max_performance else 'caller',
                    source_revision=revision, model_quality_accepted=False)
                if decode_kernel:
                    record['attention']['triton_decode'] = decode_kernel
                    decode_profile = json.loads(decode_manifest.read_text()).get('profile')
                    record['attention']['decode_precision'] = (
                        'FP32 storage; HVX QFloat QK/PV, tree QK reduction; not bitwise reference'
                        if decode_profile == 'qfloat-decode' else 'ordered FP32')
                if prefill_kernels is not None:
                    record['attention']['triton_prefill'] = prefill_kernels
                if hmx_prefill:
                    record['attention'].update(
                        implementation=('triton-hmx-prefill-hvx-qfloat-decode'
                                        if decode_profile == 'qfloat-decode' else
                                        'triton-hmx-prefill-strict-fp32-decode'),
                        triton_hmx_prefill=hmx_prefill,
                        prefill_precision='FP16 dot operands/results; FP32 online softmax state',
                        kv_storage='FP32', numerical_contract='bounded-error; not bitwise reference',
                        shared_hmx_owner=True)
                if hmx_decode:
                    record['attention'].update(
                        implementation='triton-hmx-prefill-hybrid-decode',
                        triton_hmx_decode=hmx_decode,
                        hmx_decode_precision='FP16 dot operands/results and probabilities; FP32 online state',
                        decode_dispatch='HMX for effective KV >= min_kv; supplied HVX kernel otherwise')
        record['preparation'] = {}
        for name in DOMAINS:
            entry = plan['domains'][name]
            if 'linalg' not in entry:
                continue
            prepared = prepare_resident(name, entry, build, output/(name+'-prepared'), env,
                                        integer_weights, silu_effects, activation_reuse,
                                        parallel_silu, vectorize_activation, batch_arm_rope, features)
            record['preparation'][name] = prepared
            if compile_resident:
                objects = output/(name+'-objects')
                run(name+'-compile', [config.get('python', sys.executable),
                    '-m', 'models.qwen3.runtime.arm.compile_qwen3_artifact',
                    '--input', prepared['output']['path'], '--output-bundle', objects,
                    '--libtriton', build/'compiler/profiles/dynamic-kv/libtriton.so',
                    '--backend-source', source, '--options-from', compiler_options,
                    *[item for key, enabled in domain_options[name].items()
                      for item in ('--option', key+'='+str(enabled))],
                    '--geninsert-policy', 'process', '--progress'])
                bundles[name] = load_bundle(objects)
                check_compiler(name, bundles[name][0])
                entry['objects'] = str(objects)
            prior = bundles[name][0]
            if prepared['output']['sha256'] != prior['input_sha256']:
                previous = Path(prior['input'])
                if not previous.is_file() or sha(previous) != prior['input_sha256']:
                    raise ValueError('Original compiled IR required for exact comparison: ' + name)
                identities = [complete_ir_identity(path, build/'policy/qwen-packed-i8-opt', env,
                    output/(name+'-'+label+'-full-IR.log')) for label, path in
                    [('original', previous), ('prepared', Path(prepared['output']['path']))]]
                if identities[0] != identities[1]:
                    raise ValueError('Fresh prepared computation or weights differ: ' + name +
                                     '; inspect preparation before compiling a new candidate')
                prepared['reuse'] = dict(original_bytecode_sha256=prior['input_sha256'],
                    full_IR_without_debug_locations=identities[0], constants_elided=False)
            original = Path(prior['input'])
            if original.is_file() and sha(original) == prior['input_sha256']:
                # The retained compiled input is identical in computation and
                # constants. Do not accumulate another weight-sized IR/archive.
                prepared['removed_redundant_intermediates'] = []
                for path in (Path(prepared['output']['path']),
                             Path(prepared['interchange']['archive_path'])):
                    if path.resolve() == original.resolve():
                        continue  # Fresh compilation owns this input; retain it for reproducible relinking.
                    prepared['removed_redundant_intermediates'].append(dict(
                        path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
                    path.unlink()
            write(output/(name+'-prepared/preparation.json'), prepared)
        if compile_resident:
            write(output/'resident-plan.json', plan)
        reference = constant_symbols(bundles['prefill'][1][1:], tools/'bin/llvm-readobj', env)
        targets = {}
        for name, value in sorted(reference.items()):
            targets.setdefault((value['bytes'], value['sha256']), name)
        record['shared_constants'] = reference
        mains = []
        for name in DOMAINS:
            entry = plan['domains'][name]
            bundle, objects = bundles[name]
            actual = constant_symbols(objects[1:], tools/'bin/llvm-readobj', env)
            aliases = {}
            for symbol, value in actual.items():
                key = (value['bytes'], value['sha256'])
                if key not in targets:
                    raise ValueError('No byte-identical shared constant: ' + name + '/' + symbol)
                destination = symbol if reference.get(symbol) == value else targets[key]
                if destination != symbol:
                    aliases[symbol] = destination
            names = subprocess.check_output([str(tools/'bin/llvm-nm'), '--defined-only',
                '--extern-only', str(objects[0])], env=env, text=True)
            names = {row.split()[-1] for row in names.splitlines() if row.split()}
            exports = {entry['function'], '_mlir_ciface_' + entry['function']}
            if not exports <= names:
                raise ValueError('Captured function is absent from main object: ' + name)
            # This weak compiler fallback is overridden by the application's
            # synchronous worker adapter, shared across all compiled domains.
            runtime_abi = {'__hexagon_parallel_for_v1'}
            helpers = {symbol: 'qwen_' + name + '_' + symbol
                       for symbol in names - exports - runtime_abi}
            mapping = output/(name+'-symbols.txt')
            allocation_symbols = {}
            if temporary_buffer_bytes:
                options = bundle.get('effective_options', bundle.get('options', {}))
                if any(options.get(key) not in (False, 'False') for key in
                       ('enableMultiThreading', 'enableSCFThreading')):
                    raise ValueError('Temporary buffer reuse requires serial graph allocation: ' + name)
                allocation_symbols = {'malloc': 'qwen_graph_malloc', 'free': 'qwen_graph_free'}
            mapping.write_text(''.join(a+' '+b+'\n' for a, b in
                sorted({**helpers, **aliases, **allocation_symbols}.items())))
            main = output/(name+'.o')
            run('symbols-'+name, [tools/'bin/hexagon-llvm-objcopy',
                '--no-record-command-line', '--redefine-syms='+str(mapping), objects[0], main])
            mains.append(main)
            record['domains'][name] = dict(objects=bundle['objects'], constant_aliases=aliases,
                renamed_helpers=helpers, allocation_symbols=allocation_symbols, original_constants=actual,
                bundle_sha256=sha(Path(entry['objects'])/'bundle.json'),
                compiled_input_sha256=bundle['input_sha256'],
                compiler_sha256=bundle['extension_sha256'])
        constants = []
        shared_modules = len(bundles['prefill'][1]) - 1
        if (shared_modules != 4 if q8_0 is None else not 1 <= shared_modules <= 4):
            raise ValueError('Expected four shared constant modules' if q8_0 is None else
                             'Expected one to four shared constant modules without INT8 weights')
        q8_0_record = None
        if q8_0 is not None:
            from .q8_0.builder import weights_and_library
            q8_0_record = weights_and_library(
                {name: output/(name+'-prepared/q8_0-dump') for name in DOMAINS}, Path(q8_0['hf_checkpoint']),
                plan['checkpoint_sha256'], args.llama_source.resolve()/'gguf-py', output, tools, config['lld'],
                config.get('python', sys.executable), ROOT, env, log=lambda line: None)
        flags = ['-O3', '-mv73', '-mhvx', '--ld-path='+config['lld'],
                 '-Wl,-z,separate-loadable-segments', '-shared', '-fPIC', '-G0']
        for i, obj in enumerate(bundles['prefill'][1][1:], 1):
            name = f'libqwen_dynamic_kv_prefill_const_{i}.so'
            raw = output/(name+'.unsplit')
            run('constant-'+str(i), [tools/'bin/hexagon-clang++', *flags,
                '-mlong-calls', '-nostdlib++', obj, '-Wl,-soname,'+name, '-o', raw])
            run('constant-loads-'+str(i), [config.get('python', sys.executable),
                '-m', 'hexagon_runtime.split_constant_load_segments', '--input', raw,
                '--output', output/name, '--proof', output/(name+'.proof.json'),
                '--expected-sha256', sha(raw)])
            raw.unlink()  # Reproducible intermediate; proof retains its hash and link command.
            constants.append(name)
        (output/'qwen_resident_bindings.h').write_text(header)
        worker_flags = []
        if integer_threads == 4:
            worker_flags = ['-DHEXAGON_PREPACK_LLAMA_WORKERS',
                '-mhmx', '-include', 'hmx_hexagon_protos.h',
                '-I'+str(args.llama_source.resolve()/'ggml/src/ggml-hexagon/htp'),
                '-I'+str(sdk/'incs'), '-I'+str(sdk/'incs/stddef'),
                '-I'+str(sdk/'rtos/qurt/computev73/include/qurt')]
        resource_scope_flags = (['-DQWEN_PREPACKED_FORWARD_SCOPE']
                                if integer_threads == 4 else [])
        integer_link = [output/'packed-int8.o']
        integer_libraries = []
        if int8_kernels is not None:
            int8_build = output/'int8'
            run('int8-configure', [config.get('cmake', 'cmake'), '-G', 'Ninja',
                '-S', runtime/'int8', '-B', int8_build,
                '-DLLAMA_SOURCE='+str(args.llama_source.resolve()),
                '-DHEXAGON_SDK='+str(sdk), '-DHEXAGON_TOOLS='+str(tools),
                '-DHEXKL_DIR='+str(Path(config['hexkl']).resolve()),
                *(['-DSHARE_HMX_RUNTIME=ON'] if hmx_prefill else []),
                '-DTRITON_PACK_OBJECT='+str(int8_inputs['pack'][0]),
                '-DTRITON_ACCUMULATE_OBJECT='+str(int8_inputs['accumulate'][0]),
                *(['-DTRITON_DEQUANT_OBJECT='+str(graph_inputs['dequant-dequant'][0]),
                   '-DTRITON_DEQUANT_RESIDUAL_OBJECT='+str(graph_inputs['dequant-residual'][0]),
                   '-DTRITON_DEQUANT_PRODUCT_OBJECT='+str(graph_inputs['dequant-product'][0])]
                  if features.get('int8_dequant') else [])])
            run('int8-build', [config.get('cmake', 'cmake'), '--build', int8_build,
                              '--target', 'qwen-int8-prefill-static', '--parallel', '4'])
            integer_link = [int8_build/'libqwen_int8_prefill.a']
            integer_libraries = [Path(config['hexkl'])/'lib/hexagon_toolv19_v73/libhexkl_micro.a']
            record['triton_int8_prefill'] = dict(kernels=int8_kernels,
                archive_sha256=sha(integer_link[0]), hexkl_sha256=sha(integer_libraries[0]))
        else:
            run('integer-runtime', [tools/'bin/hexagon-clang', '-O3', '-mv73', '-mhvx', *worker_flags,
                '-fPIC', '-G0', '-c', runtime/'profile'/('prepacked_int8_matmul_runtime.c' if integer_weights != 'plain'
                    else 'packed_int8_multi_matmul_runtime.c'),
                '-o', output/'packed-int8.o'])
        includes = [output, runtime/'dsp-include', runtime, ROOT/'hexagon_runtime/native',
                    sdk/'incs', sdk/'incs/stddef', sdk/'rtos/qurt/computev73/include/qurt',
                    sdk/'rtos/qurt/computev73/include/posix']
        trig = build/'runtime/trig_runtime.o'
        record['trig_runtime_sha256'] = sha(trig)
        temporary_runtime = ([ROOT/'hexagon_runtime/native/temporary_buffer_pool.cpp']
                             if temporary_buffer_bytes else [])
        temporary_flags = (['-DQWEN_TEMPORARY_BUFFER_BYTES='+str(temporary_buffer_bytes)]
                           if temporary_buffer_bytes else [])
        # Helpers expose FP32 values, not libm errno. This admits direct strict
        # FMA instructions instead of repeated fmaf calls; no fast-math flags.
        # HexKL contains SDK LTO bitcode with proprietary HMX intrinsics.
        # Upstream LLD accepts it but leaves llvm.hexagon.M8.* unresolved.
        # Keep the source-built linker for ordinary objects/constants; use the
        # matching SDK linker for the model when consuming that SDK bitcode.
        model_flags = flags
        if int8_kernels is not None:
            sdk_linker = sdk_hmx_linker['path']
            record['sdk_hmx_linker'] = sdk_hmx_linker
            model_flags = [('--ld-path='+str(sdk_linker)) if flag.startswith('--ld-path=')
                           else flag for flag in flags if flag != "-Wl,-z,separate-loadable-segments"]
        benchmark_flags = (['-DQWEN_BENCHMARK_MAX_PERFORMANCE',
                            ROOT/'hexagon_runtime/native/benchmark_performance.cpp']
                           if benchmark_max_performance else [])
        parallel_runtime = []
        if parallel_dequant:
            parallel_runtime = ['-DQWEN_PARALLEL_KV_EXPORT',
                ROOT/'hexagon_runtime/native/parallel_for.cpp',
                '-I'+str(args.llama_source.resolve()/'ggml/src/ggml-hexagon/htp')]
            record['parallel_dequant'] = dict(domains=parallel_dequant, workers=4,
                runtime_sha256=sha(ROOT/'hexagon_runtime/native/parallel_for.cpp'),
                abi='__hexagon_parallel_for_v1')
            record['parallel_kv_export'] = dict(min_query=16, workers=4,
                shares_runtime='parallel_dequant')
        # Fused-graph runtimes and kernels, appended after the model output as
        # in the accepted candidates.
        graph_link = []
        graph_flags = ['-O3', '-fPIC', '-G0', '-mv73', '-mhvx', '-mhvx-length=128B', '-std=c++17',
                       '-fno-exceptions', '-fno-rtti', '-I'+str(sdk/'incs'), '-I'+str(sdk/'incs/stddef')]
        for key, source_name, extra in (('parallel_rope', 'parallel_rope.cpp', []),
                                        ('head_norm_rope', 'parallel_rope_norm.cpp', ['-Wall']),
                                        ('embedding_rows', 'embedding_rows.cpp', ['-Wall']),
                                        ('parallel_trig', 'parallel_trig.cpp', ['-Wall']),
                                        ('causal_mask', 'parallel_mask.cpp', ['-Wall'])):
            if not features.get(key):
                continue
            obj = output/(Path(source_name).stem+'.o')
            run('graph-'+key, [tools/'bin/hexagon-clang++', *graph_flags, *extra, '-c',
                               runtime/'graph'/source_name, '-o', obj])
            graph_link.append(obj)
            if key == 'parallel_rope':
                kernel = output/'rope_kernel.o'
                run('graph-rope-kernel', [tools/'bin/hexagon-llvm-objcopy',
                    *['--localize-symbol='+s for s in ('memrefCopy', '__hexagon_mlir_expf', '_Assert',
                                                       'aligned_alloc', 'roundevenf')],
                    graph_inputs['rope'][0], kernel])
                graph_link.append(kernel)
            if key == 'head_norm_rope':
                kernel = output/'rope_norm_kernel.o'
                run('graph-rope-norm-kernel', [tools/'bin/hexagon-llvm-objcopy',
                    *['--localize-symbol='+s for s in ('memrefCopy', '__hexagon_mlir_expf', '_Assert',
                                                       'aligned_alloc', 'roundevenf')],
                    graph_inputs['rope-norm'][0], kernel])
                graph_link.append(kernel)
            if key == 'causal_mask':
                kernel = output/'mask_kernel.o'
                run('graph-mask-kernel', [tools/'bin/hexagon-llvm-objcopy', '--keep-global-symbol=qwen_causal_mask',
                                          graph_inputs['mask'][0], kernel])
                graph_link.append(kernel)
        kv_flags = []
        if features.get('kv_storage') == 'fp16':
            kv_flags = ['-DQWEN_KV_FP16']
            obj = output/'kv_fp16.o'
            run('kv-fp16', [tools/'bin/hexagon-clang++', '-O3', '-mv73', '-mhvx', '-mhvx-length=128B', '-mhvx-ieee-fp',
                            '-DPYTORCH_HEXAGON_KV_FP16_HVX', '-fPIC', '-G0', '-mlong-calls', '-std=c++17',
                            '-fno-math-errno', '-c', ROOT/'hexagon_runtime/native/kv_fp16.cpp', '-o', obj])
            graph_link.append(obj)
            # qwen_kv_append_f16_v1, called by the qwen-fp16-kv-storage appends.
            append = output/'kv_append.o'
            run('graph-kv-append', [tools/'bin/hexagon-clang++', *graph_flags, '-Wall',
                                    '-I'+str(ROOT/'hexagon_runtime/native'), '-c', runtime/'graph'/'kv_append.cpp', '-o', append])
            graph_link.append(append)
        if q8_0_record is not None:
            from .q8_0.builder import runtime_object
            q8_0_record['runtime'] = runtime_object(tools, includes, Path(config['hexkl'])/'include', output, env, run)
            library = Path(q8_0_record['library']['library'])
            shutil.copyfile(library, output/library.name)
            graph_link += [q8_0_record['runtime']['object'], q8_0_record['library']['table'],
                           '-L'+str(output), '-l'+library.name[3:-3]]
        # Mode 4 (forward without logits) relies on the Q8_0 runtime's LM head.
        no_logits_flags = ['-DQWEN_RESIDENT_NO_LOGITS'] if q8_0_record is not None else []
        # One attention K/V VTCM lease per forward. Only for Q8_0 sessions: their
        # domains call no INT8 matmul, so the INT8 prefill DMA scope never takes
        # its own 3 MB lease (both would not fit next to the HMX pool).
        if q8_0_record is not None and features.get('kv_dma'):
            no_logits_flags.append('-DQWEN_ATTENTION_FORWARD_SCOPE')
        run('link', [tools/'bin/hexagon-clang++', *model_flags, '-mlong-calls', '-nostdlib++',
            '-std=c++17', *kv_flags, *no_logits_flags, '-fno-math-errno', *temporary_flags, *temporary_runtime, *benchmark_flags, *parallel_runtime,
            *resource_scope_flags,
            *['-I'+str(path) for path in includes], runtime/'resident_session.cpp',
            *mains, *integer_link, *attention_archive, ROOT/'hexagon_runtime/native/kv_cache.cpp',
            runtime/'aligned_alloc_shim.cpp', trig, '-L'+str(output),
            *['-l'+name[3:-3] for name in constants],
            '-L'+str(sdk/'libs/qhl_hvx/prebuilt/hexagon_toolv87_v73'), '-lqhmath_hvx',
            tools/'target/hexagon/lib/v73/G0/pic/libc++.a',
            tools/'target/hexagon/lib/v73/G0/pic/libc++abi.a', *integer_libraries, '-o', output/'model.so',
            *graph_link])
        # Unsupported SDK intrinsics must fail here, not at the DSP loader.
        unresolved = subprocess.check_output(
            [str(tools/'bin/llvm-nm'), '-u', str(output/'model.so')], env=env, text=True)
        intrinsics = [line.split()[-1] for line in unresolved.splitlines()
                      if line.split() and line.split()[-1].startswith('llvm.')]
        if intrinsics:
            raise ValueError('Unlowered LLVM intrinsics in model: ' + ', '.join(intrinsics))
        for name in DOMAINS:
            entry = plan['domains'][name]
            tag = 'single0' if name == 'initial_single' else name
            graph = Path(entry['graph'])
            if sha(graph) != entry['graph_sha256']:
                raise ValueError('Captured graph changed: ' + name)
            shutil.copyfile(graph, output/(tag+'-graph.txt'))
            write(output/(tag+'-mapping.json'), entry['mapping'])
        for name in ('single1-graph.txt', 'single1-mapping.json', 'decompositions.py'):
            item = plan['runtime_files'][name]
            if sha(Path(item['path'])) != item['sha256']:
                raise ValueError('Runtime graph/decomposition changed: ' + name)
            shutil.copyfile(item['path'], output/name)
        if q8_0_record is not None:
            constants.append(Path(q8_0_record['library']['library']).name)
        files = ['model.so', 'decompositions.py', *constants]
        if needs_llama:
            files.append('LICENSE.llama.cpp')
        files += [tag+suffix for tag in ('single0', 'initial', 'prefill', 'decode', 'single1')
                  for suffix in ('-graph.txt', '-mapping.json')]
        manifest = dict(schema='qwen3-resident-session-v1', constants=constants,
            checkpoint_sha256=plan['checkpoint_sha256'],
            files={name: dict(bytes=(output/name).stat().st_size, sha256=sha(output/name)) for name in files})
        manifest['resident_forward_modes'] = [1, 3, 4] if q8_0_record is not None else [1, 3]
        # Building a library is not a model-quality qualification, even when
        # its outputs preserve an earlier implementation's arithmetic.
        manifest['model_quality_accepted'] = False
        manifest['integer_weights'] = integer_weights
        manifest['integer_threads'] = integer_threads
        manifest['integer_resource_scope'] = 'forward' if resource_scope_flags else 'operator'
        if int8_kernels is not None:
            manifest['triton_int8_prefill'] = record['triton_int8_prefill']
        manifest['benchmark_max_performance'] = benchmark_max_performance
        manifest['silu_effects'] = silu_effects
        manifest['activation_reuse'] = activation_reuse
        manifest['parallel_silu'] = parallel_silu
        if parallel_dequant:
            manifest['parallel_dequant'] = record['parallel_dequant']
        if silu_kernel:
            manifest['triton_silu'] = silu_kernel
            manifest['silu_precision'] = (
                'llama.cpp Hexagon SDK QFloat SiLU; not bitwise PyTorch reference'
                if silu_profile == 'llama-silu-fp32' else
                'bounded FP32 approximation with strict exceptional vectors; not bitwise reference'
                if silu_profile == 'experimental-silu-fp32' else 'strict FP32 reference')
        manifest['batch_arm_rope'] = batch_arm_rope
        manifest['row_minmax'] = row_minmax
        manifest['domain_compiler_options'] = domain_options
        manifest['vectorize_activation'] = vectorize_activation
        if features:
            manifest['fused_graph'] = {key: value for key, value in features.items() if key != 'q8_0_projections'}
            record['fused_graph'] = features
        if q8_0_record is not None:
            record['q8_0_projections'] = q8_0_record
            manifest['q8_0_projections'] = dict(
                library=Path(q8_0_record['library']['library']).name,
                projections=q8_0_record['library']['count'], shared_constant_libraries=shared_modules,
                weights=dict(source=q8_0_record['weights']['source'], hf_sha256=q8_0_record['weights']['hf_sha256'],
                             weights_json_sha256=q8_0_record['weights']['sha256'],
                             library_sha256=q8_0_record['library']['library_sha256'],
                             table_sha256=q8_0_record['library']['table_sha256'],
                             runtime_sha256=q8_0_record['runtime']['object_sha256']),
                numerics=dict(prefill='Q8_0 weights dequantized to FP16, FP16 activations, HMX FP16 accumulation',
                              single_row='Q8_0 x Q8_0 per 32 values (llama.cpp CPU vec_dot), FP32 accumulation',
                              lm_head='Q8_0 x Q8_0 per 32 values (llama.cpp CPU vec_dot), FP32 accumulation'))
        manifest['temporary_buffer_bytes'] = temporary_buffer_bytes
        if needs_llama:
            manifest['llama_runtime'] = record['llama_runtime']
        if fused_domains:
            manifest['attention'] = record['attention']
        write(output/'session.json', manifest)
        load_session(output, {'files': {'hf/model.safetensors': {'sha256': plan['checkpoint_sha256']}}})
        verify_products(build, source, args.config)
        if decode_kernel and (sha(decode_object) != decode_kernel['object_sha256'] or
                              sha(decode_manifest) != decode_kernel['manifest_sha256']):
            raise ValueError('Triton decode inputs changed during linking')
        for label, (obj, manifest_path, entry) in graph_inputs.items():
            if sha(obj) != entry['object_sha256'] or sha(manifest_path) != entry['manifest_sha256']:
                raise ValueError('Fused-graph kernel inputs changed during linking: ' + label)
        if hmx_prefill and (sha(hmx_object) != hmx_prefill['object_sha256'] or
                            sha(hmx_manifest) != hmx_prefill['manifest_sha256']):
            raise ValueError('HMX prefill inputs changed during linking')
        for kind, (obj, manifest_path) in int8_inputs.items():
            if (sha(obj) != int8_kernels[kind]['object_sha256'] or
                    sha(manifest_path) != int8_kernels[kind]['manifest_sha256']):
                raise ValueError('INT8 kernel inputs changed during linking: ' + kind)
        if (native_identity() != record['native_sources']
                or sha(Path(__file__)) != record['builder_sha256']
                or sha(args.resident_spec) != record['spec_sha256']
                or sha(args.config) != record['configuration_sha256']
                or (compile_resident and sha(compiler_options) != record['compiler_options_sha256'])):
            raise ValueError('Link inputs changed during execution; rerun with stable source')
        record.update(status='resident_session_linked_NOT_DSP_qualified',
                      session_sha256=sha(output/'session.json'))
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        write(output/'link.json', record)


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
    command = [str(python), str(ROOT/'models/qwen3/runtime/arm/prepare_captured_qwen_hexagon.py'),
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
    for name in ('config', 'build', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('producer', 'context', 'hf', 'catalog'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--producer-sha256')
    parser.add_argument('--resident-spec', type=Path,
                        help='Link four existing dynamic-cache object bundles; no model recompilation')
    parser.add_argument('--compile-resident', action='store_true',
                        help='Compile all four captured Linalg domains before resident linking')
    parser.add_argument('--llama-source', type=Path,
                        help='Pinned llama.cpp checkout for selected fused attention or integer workers')
    parser.add_argument('--trusted-local-producer', action='store_true')
    parser.add_argument('--compiler-source', type=Path)
    parser.add_argument('--frontend-only', action='store_true')
    args = parser.parse_args()
    if args.resident_spec:
        if any((args.producer, args.context, args.hf, args.catalog, args.producer_sha256,
                args.frontend_only, args.trusted_local_producer)):
            parser.error('Resident linking cannot be combined with legacy capture-build options')
        return link_resident(args)
    if args.compile_resident:
        parser.error('--compile-resident requires --resident-spec')
    if args.llama_source:
        parser.error('--llama-source requires an explicit resident plan')
    if not all((args.producer, args.context, args.hf, args.catalog,
                args.producer_sha256, args.compiler_source)):
        parser.error('Model build requires producer, producer-sha256, context, hf, catalog and compiler-source')
    if not args.trusted_local_producer:
        parser.error('Only explicitly trusted local graph captures may be restored')
    source, build, output = args.compiler_source.resolve(), args.build.resolve(), args.output.resolve()
    from .build import verify_products
    compiler_build = verify_products(build, source, args.config)
    sys.path[:0] = [str(source), str(ROOT/'models/qwen3/runtime/arm')]
    config = json.loads(args.config.read_text())
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(source),
        str(ROOT/'models/qwen3/runtime/arm')]), LD_LIBRARY_PATH=os.pathsep.join(config.get('library_path', [])))
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
        run('compile', [str(python), str(ROOT/'models/qwen3/runtime/arm/compile_qwen3_artifact.py'),
            '--input', str(output/'prepared/hexagon-linalg.mlirbc'), '--output-bundle', str(output/'objects'),
            '--libtriton', str(build/'compiler/profiles'/profile/'libtriton.so'),
            '--backend-source', str(source), '--options-from', str(ROOT/'models/qwen3/config/compiler-options.json'),
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
