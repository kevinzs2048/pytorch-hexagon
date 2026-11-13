"""Exercise build/relink bookkeeping with stub compiler/SDK commands, not DSP math."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.qwen3 import build, build_model
from models.qwen3.runtime.arm import prepare_qwen_resident_bindings as bindings
from models.qwen3.runtime.arm.qwen3_object_bundle import write_bundle


def test_recorded_options_match_compiler_schema_and_allow_row_override():
    import importlib.util
    from models.qwen3.runtime.arm.compile_qwen3_artifact import resolve_options

    path = build_model.ROOT/'third_party/hexagon-mlir/qcom_hexagon_backend/backend/hexagon_options.py'
    spec = importlib.util.spec_from_file_location('resident_test_options', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = module.HexagonOptions().__dict__
    options, _ = resolve_options(defaults,
        build_model.ROOT/'models/qwen3/config/compiler-options.json',
        ['enableRowMinMax=True'])
    assert options['enableRowMinMax'] == 'True'
    assert options['addFastMath'] == options['enableConversionToFp16'] == 'False'
    for key in ('dispatchHMX', 'reuseHMXWorkspace', 'prepareHMXMatmul',
                'externalKernelAllocator', 'scheduleFMAChains', 'scheduleDequantLoops',
                'scheduleReductionLoops', 'parallelDequantLoops'):
        assert options[key] == 'False'
    assert options['stackBufferBudget'] == '0'


@pytest.fixture
def resident_build(tmp_path, monkeypatch):
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    config = write(tmp_path/'toolchain.json', json.dumps(dict(
        hexagon_tools=str(tmp_path/'sdk-tools'), hexagon_sdk=str(tmp_path/'sdk'),
        lld='stub-linker')))
    write(tmp_path/'build/build.json', '{}')
    write(tmp_path/'build/runtime/trig_runtime.o', 'stub trig object')
    plan = dict(checkpoint_sha256='checkpoint', temporary_buffer_bytes=1024,
                domains={}, runtime_files={})
    for name in bindings.DOMAINS:
        graph = write(tmp_path/(name+'-graph.txt'), 'test graph')
        plan['domains'][name] = dict(linalg='test capture', function='forward_'+name,
            graph=str(graph), graph_sha256=build_model.sha(graph), mapping={})
    for name in ('single1-graph.txt', 'single1-mapping.json', 'decompositions.py'):
        file = write(tmp_path/name, '{}')
        plan['runtime_files'][name] = dict(path=str(file), sha256=build_model.sha(file))
    spec = write(tmp_path/'plan.json', json.dumps(plan))
    args = SimpleNamespace(config=config, compiler_source=tmp_path/'compiler',
        build=tmp_path/'build', resident_spec=spec, compile_resident=True,
        llama_source=None, output=tmp_path/'session')
    state = SimpleNamespace(args=args, plan=plan, commands=[], override={})
    monkeypatch.setattr(build, 'verify_products', lambda *a: {
        'products': {'compiler/profiles/dynamic-kv/libtriton.so': 'compiler'}})
    monkeypatch.setattr(build, 'native_identity', lambda: {'test-native': 'stable'})
    monkeypatch.setattr(bindings, 'generate', lambda spec: 'test bindings')
    monkeypatch.setattr(build_model, 'constant_symbols', lambda *a: {
        'weight': {'bytes': 8, 'sha256': 'same-weight'}})

    def prepare(name, entry, build, output, environment, *options):
        source = write(output/'input.mlir', 'test prepared graph '+name)
        archive = write(output/'capture.bin', 'test archive')
        return dict(output={'path': str(source), 'sha256': build_model.sha(source)},
                    interchange={'archive_path': str(archive)})
    monkeypatch.setattr(build_model, 'prepare_resident', prepare)

    def run(argv, **kwargs):
        state.commands.append(argv)
        if '--output-bundle' in argv:
            directory = Path(argv[argv.index('--output-bundle')+1])
            name = directory.name.removesuffix('-objects')
            source = Path(argv[argv.index('--input')+1])
            option = (argv[argv.index('--option')+1] if '--option' in argv
                      else 'enableRowMinMax=False')
            key, value = option.split('=', 1)
            assert key == 'enableRowMinMax'
            effective = dict(enableMultiThreading='False', enableSCFThreading='False',
                             enableRowMinMax=value)
            effective.update(dict(argv[i+1].split('=', 1)
                                  for i, value in enumerate(argv) if value == '--option'))
            effective.update(state.override.get(name, {}))
            write_bundle(directory, [b'\x7fELF'+name.encode()]+[b'\x7fELFconstant']*4,
                dict(extension_sha256='compiler', codegen_policy=dict(
                    scope='isolated_process', llvm_option='hexagon-insert', enabled=False),
                    input=str(source), input_sha256=build_model.sha(source),
                    effective_options=effective))
        elif any(x.startswith('--redefine-syms=') for x in argv):
            Path(argv[-1]).write_bytes(b'stub relocated object')
        elif '--output' in argv:
            Path(argv[argv.index('--output')+1]).write_bytes(b'stub split library')
        elif '-o' in argv:
            Path(argv[argv.index('-o')+1]).write_bytes(b'stub SDK product')
        return SimpleNamespace(returncode=0)

    def symbols(argv, **kwargs):
        name = Path(argv[-1]).parent.name.removesuffix('-objects')
        return 'T forward_'+name+'\nT _mlir_ciface_forward_'+name+'\n'
    monkeypatch.setattr(build_model.subprocess, 'run', run)
    monkeypatch.setattr(build_model.subprocess, 'check_output', symbols)
    return state


@pytest.mark.parametrize('enabled', [False, True])
def test_compile_with_pool_finishes_and_relink_preserves_all_domain_options(resident_build, enabled):
    state = resident_build
    state.plan['benchmark_max_performance'] = enabled
    if enabled:
        state.plan['row_minmax'] = True
    state.args.resident_spec.write_text(json.dumps(state.plan))
    build_model.link_resident(state.args)
    output = state.args.output
    record = json.loads((output/'link.json').read_text())
    assert record['status'] == 'resident_session_linked_NOT_DSP_qualified'
    assert record['compiler_options_sha256'] == build_model.sha(
        build_model.ROOT/'models/qwen3/config/compiler-options.json')
    compiled = [c for c in state.commands if '--output-bundle' in c]
    assert len(compiled) == 4
    assert all(c[c.index('--option')+1] == 'enableRowMinMax='+str(enabled) for c in compiled)
    manifest = json.loads((output/'session.json').read_text())
    assert manifest['row_minmax'] is enabled and manifest['temporary_buffer_bytes'] == 1024
    assert manifest['benchmark_max_performance'] is enabled
    assert manifest['model_quality_accepted'] is False
    link = next(c for c in state.commands if c[-1] == str(output/'model.so'))
    assert ('-DQWEN_BENCHMARK_MAX_PERFORMANCE' in link) is enabled
    assert '-DQWEN_PARALLEL_KV_EXPORT' not in link
    for domain in record['domains'].values():
        assert domain['allocation_symbols'] == {'malloc': 'qwen_graph_malloc', 'free': 'qwen_graph_free'}
    if not enabled:
        # Legacy receipts predate this option; omitted means the disabled default.
        for name in bindings.DOMAINS:
            path = output/(name+'-objects/bundle.json')
            bundle = json.loads(path.read_text())
            del bundle['effective_options']['enableRowMinMax']
            path.write_text(json.dumps(bundle))
    # Load the emitted plan and sealed bundles through the actual relink route.
    state.args.resident_spec = output/'resident-plan.json'
    state.args.compile_resident = False
    state.args.output = output.with_name('relinked')
    state.commands.clear()
    build_model.link_resident(state.args)
    assert not any('--output-bundle' in c for c in state.commands)
    assert json.loads((state.args.output/'session.json').read_text())['row_minmax'] is enabled
    # Changing a plan must not silently relabel existing compiled objects.
    plan = json.loads(state.args.resident_spec.read_text())
    plan['row_minmax'] = not enabled
    state.args.resident_spec.write_text(json.dumps(plan))
    state.args.output = output.with_name('mismatched')
    with pytest.raises(ValueError, match='row-minmax option differs: initial_single'):
        build_model.link_resident(state.args)
    assert not state.args.output.exists()


def test_rejects_one_domain_that_did_not_apply_requested_option(resident_build):
    state = resident_build
    state.plan['row_minmax'] = True
    state.args.resident_spec.write_text(json.dumps(state.plan))
    state.override['prefill'] = {'enableRowMinMax': 'False'}
    with pytest.raises(ValueError, match='row-minmax option differs: prefill'):
        build_model.link_resident(state.args)
    assert not (state.args.output/'model.so').exists()


@pytest.mark.parametrize('bad', ['True', 1, None])
@pytest.mark.parametrize('option', ['row_minmax', 'benchmark_max_performance'])
def test_rejects_non_boolean_setting_before_preparation(resident_build, bad, option):
    state = resident_build
    state.plan[option] = bad
    state.args.resident_spec.write_text(json.dumps(state.plan))
    with pytest.raises(ValueError, match=option+' must be a boolean'):
        build_model.link_resident(state.args)
    assert state.commands == []


@pytest.mark.parametrize("prefill,silu", [(False, None), (True, None),
    (True, "strict-silu"), (True, "experimental-silu-fp32"), (True, "llama-silu-fp32")])
def test_triton_decode_selection_survives_relink(resident_build, monkeypatch, prefill, silu):
    state = resident_build
    root = state.args.output.parent
    kernel = root/'kernel.o'
    kernel.write_bytes(b'test decode object')
    receipt = root/'manifest.json'
    receipt.write_text('{}')
    state.plan['triton_decode'] = dict(object=str(kernel),
        object_sha256=build_model.sha(kernel), manifest_sha256=build_model.sha(receipt))
    state.plan['domains']['decode']['attention'] = 'fp32-ordered-workers-v1'
    if prefill:
        state.plan['triton_prefill'] = {}
        for domain in ('initial', 'prefill'):
            state.plan['domains'][domain]['attention'] = 'fp32-ordered-workers-v1'
        for stage in ('pack', 'qk', 'softmax', 'pv'):
            directory = root/stage
            directory.mkdir()
            obj = directory/'kernel.o'
            obj.write_bytes(b'test stage '+stage.encode())
            manifest = directory/'manifest.json'
            manifest.write_text(json.dumps(dict(profile='strict-prefill-'+stage,
                object_sha256=build_model.sha(obj), lwp={'enabled': False})))
            state.plan['triton_prefill'][stage] = dict(object=str(obj),
                object_sha256=build_model.sha(obj), manifest_sha256=build_model.sha(manifest))

    if silu:
        state.plan['parallel_silu'] = True
        folder = root/'silu'
        folder.mkdir()
        obj = folder/'kernel.o'
        obj.write_bytes(b'test silu')
        manifest = folder/'manifest.json'
        manifest.write_text(json.dumps(dict(profile=silu,
            object_sha256=build_model.sha(obj), lwp={'enabled': False})))
        state.plan['triton_silu'] = dict(object=str(obj), object_sha256=build_model.sha(obj),
            manifest_sha256=build_model.sha(manifest))
    state.args.llama_source = root/'llama'
    state.args.llama_source.mkdir()
    (state.args.llama_source/'LICENSE').write_text('test license')
    state.args.resident_spec.write_text(json.dumps(state.plan))
    original = build_model.subprocess.run

    def run(argv, **kwargs):
        if '--target' in argv:
            target = argv[argv.index('--target')+1]
            assert target in ('qwen-triton-strict-attention-static', 'qwen-triton-silu-static')
            directory = Path(argv[argv.index('--build')+1])
            directory.mkdir(exist_ok=True)
            archive = 'libqwen_triton_silu.a' if target == 'qwen-triton-silu-static' else 'libqwen_triton_strict_attention.a'
            (directory/archive).write_bytes(b'test archive')
        return original(argv, **kwargs)
    monkeypatch.setattr(build_model.subprocess, 'run', run)
    build_model.link_resident(state.args)
    assert any('-DTRITON_DECODE_OBJECT='+str(kernel) in c for c in state.commands)
    output = state.args.output
    state.args.resident_spec = output/'resident-plan.json'
    state.args.compile_resident = False
    state.args.output = root/'relinked'
    build_model.link_resident(state.args)
    session = json.loads((state.args.output/'session.json').read_text())
    assert session['attention']['triton_decode'] == state.plan['triton_decode']
    if silu:
        assert session['triton_silu'] == state.plan['triton_silu']
        assert ('not bitwise' in session['silu_precision']) == (silu != 'strict-silu')
        assert any('-DTRITON_SILU_OBJECT='+str(root/'silu/kernel.o') in c for c in state.commands)
    if prefill:
        assert session['attention']['triton_prefill'] == state.plan['triton_prefill']
        assert any('-DTRITON_STRICT_PREFILL_QK_OBJECT='+str(root/'qk/kernel.o') in c
                   for c in state.commands)
        (root/'qk/kernel.o').write_bytes(b'changed prefill')
        state.args.output = root/'changed-prefill'
        with pytest.raises(ValueError, match='Strict prefill kernel identity/profile differs: qk'):
            build_model.link_resident(state.args)
        assert not state.args.output.exists()

    # The plan must not silently pick up a different kernel at the same path.
    kernel.write_bytes(b'changed object')
    state.args.output = root/'changed'
    with pytest.raises(ValueError, match='Triton decode object/manifest changed'):
        build_model.link_resident(state.args)
    assert not state.args.output.exists()


@pytest.mark.parametrize('option', ['scheduleDequantLoops', 'enableRowMinMax'])
def test_domain_scheduling_options_survive_relink_and_reject_stale_objects(resident_build, option):
    state = resident_build
    state.plan['domains']['decode']['compiler_options'] = {option: True}
    state.args.resident_spec.write_text(json.dumps(state.plan))
    build_model.link_resident(state.args)
    output = state.args.output
    bundle = json.loads((output/'decode-objects/bundle.json').read_text())
    assert bundle['effective_options'][option] == 'True'
    state.args.resident_spec = output/'resident-plan.json'
    state.args.compile_resident = False
    state.args.output = output.with_name('scheduled-relink')
    build_model.link_resident(state.args)
    plan = json.loads(state.args.resident_spec.read_text())
    plan['domains']['decode']['compiler_options'][option] = False
    state.args.resident_spec.write_text(json.dumps(plan))
    state.args.output = output.with_name('stale-schedule')
    with pytest.raises(ValueError, match='Resident (compiler option|row-minmax option) differs: decode'):
        build_model.link_resident(state.args)
    assert not state.args.output.exists()


@pytest.mark.parametrize('options', [{'addFastMath': True}, {'scheduleFMAChains': 'True'}])
def test_rejects_unsupported_domain_scheduling_options(resident_build, options):
    state = resident_build
    state.plan['domains']['decode']['compiler_options'] = options
    state.args.resident_spec.write_text(json.dumps(state.plan))
    with pytest.raises(ValueError, match='Unsupported per-domain compiler options: decode'):
        build_model.link_resident(state.args)
    assert state.commands == []


@pytest.mark.parametrize('hmx_attention', [False, True, 'decode'])
def test_int8_prefill_relink_preserves_kernels_and_rejects_replacement(
        resident_build, monkeypatch, hmx_attention):
    state = resident_build
    root = state.args.output.parent
    state.plan.update(integer_weights='hvx-i8-prepacked-v1', integer_threads=4,
                      triton_int8_prefill={})
    for kind in ('pack', 'accumulate'):
        folder = root/kind
        folder.mkdir()
        kernel = folder/'kernel.o'
        kernel.write_bytes(b'test integer object '+kind.encode())
        receipt = folder/'manifest.json'
        receipt.write_text(json.dumps(dict(profile='strict-int8-'+kind,
            object_sha256=build_model.sha(kernel), lwp={'enabled': False})))
        state.plan['triton_int8_prefill'][kind] = dict(object=str(kernel),
            object_sha256=build_model.sha(kernel), manifest_sha256=build_model.sha(receipt))
    state.args.llama_source = root/'llama'
    state.args.llama_source.mkdir()
    (state.args.llama_source/'LICENSE').write_text('test license')
    hexkl = root/'hexkl/lib/hexagon_toolv19_v73/libhexkl_micro.a'
    hexkl.parent.mkdir(parents=True)
    hexkl.write_bytes(b'test HexKL')
    config = json.loads(state.args.config.read_text())
    sdk_linker = Path(config['hexagon_tools'])/'bin/hexagon-link'
    sdk_linker.parent.mkdir(parents=True)
    sdk_linker.write_bytes(b'test SDK linker')
    config['hexkl'] = str(root/'hexkl')
    state.args.config.write_text(json.dumps(config))
    if hmx_attention:
        for domain in ('initial', 'prefill', 'decode'):
            state.plan['domains'][domain]['attention'] = 'fp32-ordered-workers-v1'
        for key, profile in [('triton_decode', 'strict-decode'),
                             ('triton_hmx_prefill', 'experimental-prefill-fp16')]:
            folder = root/key
            folder.mkdir()
            obj = folder/'kernel.o'
            obj.write_bytes(profile.encode())
            receipt = folder/'manifest.json'
            receipt.write_text(json.dumps(dict(profile=profile,
                object_sha256=build_model.sha(obj), lwp={'enabled': False},
                raw_buffer_access='dsp-load-store', prepared_hmx_slot_bytes=196608, reuse_hmx_workspace=True,
                prepare_hmx_matmul=True, external_allocator=True)))
            state.plan[key] = dict(object=str(obj), object_sha256=build_model.sha(obj),
                                   manifest_sha256=build_model.sha(receipt))
    if hmx_attention == 'decode':
        folder = root/'hmx-decode'
        folder.mkdir()
        obj = folder/'kernel.o'
        obj.write_bytes(b'q1 HMX kernel')
        receipt = folder/'manifest.json'
        receipt.write_text(json.dumps(dict(profile='experimental-decode-fp16',
            object_sha256=build_model.sha(obj), lwp={'enabled': False},
            entry_point='attention_forward_q1_hmx_g2', kv_heads_per_program=2, prepared_hmx_slot_bytes=196608, shared_runtime='experimental-prefill-fp16',
            contract={'query': [1, 1]}, raw_buffer_access='dsp-load-store',
            reuse_hmx_workspace=True, prepare_hmx_matmul=True, external_allocator=True)))
        state.plan['triton_hmx_decode'] = dict(object=str(obj),
            object_sha256=build_model.sha(obj), manifest_sha256=build_model.sha(receipt), min_kv=256)
    state.args.resident_spec.write_text(json.dumps(state.plan))
    original = build_model.subprocess.run

    def run(argv, **kwargs):
        if '--target' in argv:
            directory = Path(argv[argv.index('--build')+1])
            directory.mkdir(exist_ok=True)
            target = argv[argv.index('--target')+1]
            archive = {'qwen-workers-static': 'libqwen_workers.a',
                       'qwen-triton-experimental-attention-static': 'libqwen_triton_experimental_attention.a',
                       'qwen-int8-prefill-static': 'libqwen_int8_prefill.a'}[target]
            (directory/archive).write_bytes(b'test archive')
        return original(argv, **kwargs)
    monkeypatch.setattr(build_model.subprocess, 'run', run)
    build_model.link_resident(state.args)
    output = state.args.output
    link = next(c for c in state.commands if c[-1] == str(output/'model.so'))
    assert '--ld-path='+str(sdk_linker) in link
    assert '-DQWEN_PREPACKED_FORWARD_SCOPE' in link
    constant_link = next(c for c in state.commands if any(str(a).endswith('.unsplit') for a in c))
    assert '--ld-path='+config['lld'] in constant_link
    assert str(output/'int8/libqwen_int8_prefill.a') in link
    assert str(output/'packed-int8.o') not in link
    assert link.index(str(output/'int8/libqwen_int8_prefill.a')) < link.index(str(hexkl))
    state.args.resident_spec = output/'resident-plan.json'
    state.args.compile_resident = False
    state.args.output = root/'relinked-int8'
    build_model.link_resident(state.args)
    session = json.loads((state.args.output/'session.json').read_text())
    assert session['triton_int8_prefill']['kernels'] == state.plan['triton_int8_prefill']
    assert session['integer_resource_scope'] == 'forward'
    if hmx_attention:
        attention = session['attention']
        assert attention['implementation'] == ('triton-hmx-prefill-hybrid-decode'
            if hmx_attention == 'decode' else 'triton-hmx-prefill-strict-fp32-decode')
        if hmx_attention == 'decode':
            assert attention['triton_hmx_decode'] == state.plan['triton_hmx_decode']
            assert any('-DTRITON_HMX_DECODE_MIN_KV=256' in c for c in state.commands)
        assert attention['shared_hmx_owner'] and attention['kv_storage'] == 'FP32'
        assert attention['model_quality_accepted'] is False
        assert sum('-DSHARE_HMX_RUNTIME=ON' in c for c in state.commands) == 4
        path = root/'triton_hmx_prefill/manifest.json'
        original_receipt = path.read_text()
        broken = json.loads(original_receipt)
        broken['external_allocator'] = False
        path.write_text(json.dumps(broken))
        plan_path = state.args.resident_spec
        plan = json.loads(plan_path.read_text())
        original_plan = plan_path.read_text()
        plan['triton_hmx_prefill']['manifest_sha256'] = build_model.sha(path)
        plan_path.write_text(json.dumps(plan))
        state.args.output = root/'bad-hmx-contract'
        with pytest.raises(ValueError, match='HMX prefill kernel identity'):
            build_model.link_resident(state.args)
        assert not state.args.output.exists()
        path.write_text(original_receipt)
        plan_path.write_text(original_plan)
    if hmx_attention == 'decode':
        # An old one-head object cannot silently link to a two-head launcher.
        path = root/'hmx-decode/manifest.json'
        original_receipt = path.read_text()
        broken = json.loads(original_receipt)
        broken['entry_point'] = 'attention_forward_q1_hmx'
        path.write_text(json.dumps(broken))
        plan_path = state.args.resident_spec
        original_plan = plan_path.read_text()
        plan = json.loads(original_plan)
        plan['triton_hmx_decode']['manifest_sha256'] = build_model.sha(path)
        plan_path.write_text(json.dumps(plan))
        state.args.output = root/'old-hmx-launcher-contract'
        with pytest.raises(ValueError, match='HMX decode kernel identity'):
            build_model.link_resident(state.args)
        assert not state.args.output.exists()
        path.write_text(original_receipt)
        plan_path.write_text(original_plan)
        # Decode borrows the prefill object's runtime. Matching entry names
        # alone cannot make its old 128 KiB slots hold a grouped QK tile.
        path = root/'triton_hmx_prefill/manifest.json'
        original_receipt = path.read_text()
        broken = json.loads(original_receipt)
        broken['prepared_hmx_slot_bytes'] = 128 * 1024
        path.write_text(json.dumps(broken))
        plan = json.loads(original_plan)
        plan['triton_hmx_prefill']['manifest_sha256'] = build_model.sha(path)
        plan_path.write_text(json.dumps(plan))
        state.args.output = root/'old-hmx-workspace-contract'
        with pytest.raises(ValueError, match='shared runtime requires 192KiB'):
            build_model.link_resident(state.args)
        assert not state.args.output.exists()
        path.write_text(original_receipt)
        plan_path.write_text(original_plan)
    (root/'pack/kernel.o').write_bytes(b'changed kernel')
    state.args.output = root/'changed-int8'
    with pytest.raises(ValueError, match='INT8 kernel identity/profile differs'):
        build_model.link_resident(state.args)
    assert not state.args.output.exists()


def test_link_rejects_unlowered_intrinsics_before_session_receipt(resident_build, monkeypatch):
    state = resident_build
    state.args.resident_spec.write_text(json.dumps(state.plan))
    original = build_model.subprocess.check_output

    def symbols(argv, **kwargs):
        if '-u' in argv:
            return '         U llvm.hexagon.M8.mxclracc\n'
        return original(argv, **kwargs)

    monkeypatch.setattr(build_model.subprocess, 'check_output', symbols)
    with pytest.raises(ValueError, match='Unlowered LLVM intrinsics'):
        build_model.link_resident(state.args)
    assert not (state.args.output/'session.json').exists()


def test_parallel_dequant_requires_scheduled_helpers(resident_build):
    state = resident_build
    state.plan['domains']['prefill']['compiler_options'] = {'parallelDequantLoops': True}
    state.args.resident_spec.write_text(json.dumps(state.plan))
    with pytest.raises(ValueError, match='Parallel dequant requires scheduled'):
        build_model.link_resident(state.args)
    assert not state.commands


def test_parallel_dequant_requires_worker_runtime(resident_build):
    state = resident_build
    state.plan['domains']['prefill']['compiler_options'] = {
        'parallelDequantLoops': True, 'scheduleDequantLoops': True}
    state.args.resident_spec.write_text(json.dumps(state.plan))
    with pytest.raises(ValueError, match='Use --llama-source'):
        build_model.link_resident(state.args)
    assert not state.commands


def test_kv_export_reuses_linked_parallel_runtime(resident_build, monkeypatch):
    state = resident_build
    state.plan['domains']['prefill']['compiler_options'] = {
        'parallelDequantLoops': True, 'scheduleDequantLoops': True}
    state.args.llama_source = state.args.output.parent/'llama'
    state.args.llama_source.mkdir()
    (state.args.llama_source/'LICENSE').write_text('test license')
    state.args.resident_spec.write_text(json.dumps(state.plan))
    original = build_model.subprocess.run

    def run(argv, **kwargs):
        if '--target' in argv and 'qwen-workers-static' in argv:
            directory = Path(argv[argv.index('--build')+1])
            directory.mkdir(exist_ok=True)
            (directory/'libqwen_workers.a').write_bytes(b'test worker archive')
        return original(argv, **kwargs)

    monkeypatch.setattr(build_model.subprocess, 'run', run)
    build_model.link_resident(state.args)
    link = next(c for c in state.commands if c[-1] == str(state.args.output/'model.so'))
    assert '-DQWEN_PARALLEL_KV_EXPORT' in link
    runtime = str(build_model.ROOT/'hexagon_runtime/native/parallel_for.cpp')
    assert link.count(runtime) == 1
    record = json.loads((state.args.output/'link.json').read_text())
    assert record['parallel_kv_export']['shares_runtime'] == 'parallel_dequant'
