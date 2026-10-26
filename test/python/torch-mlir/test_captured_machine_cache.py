"""Orchestration tests with fake compiler processes, not compiler/DSP acceptance.

Real torch-mlir lowering and board cold/warm validation are separate tests.
"""
from qwen3.compiler_dependency import compiler_root
import json
from pathlib import Path
import struct
import types
import pytest

from hexagon_compiled_artifact_cache import canonical, digest_file
from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache


def fixture_cache(tmp_path, monkeypatch, *, packed=False,effects=False,copy=False,multi=False,dynamic_copy=False,multi_effects=False,fixed512=False,fixed512_q1=False,reuse=False,reuse_outcome='hit',prepared_reuse=False,prepared_outcome='hit'):
    def blob(name, data=b'test-only'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return dict(path=str(path), sha256=digest_file(path))

    def record(name, value):
        return blob(name, canonical(value))

    compiler = blob('compiler')
    wrapper = blob('wrapper.cpp')
    blob('qwen3_sleef_exp_candidate.h')
    linker = blob('linker')
    reference = record('reference.json', dict(wrapper=wrapper,
        tensor_inputs=[], scalar_inputs=[]))
    constants = record('constants.json', dict(inputs=dict(lld=linker)))
    manifest = record('manifest.json', dict(link_command=compiler['path']+' -o /not-an-input.so'))
    inputs = {name: compiler for name in ('attention_optimizer', 'sums_optimizer',
                                         'knorm_optimizer')}
    inputs.update(compiler=compiler, reference=reference, constants=constants,
                  manifest=manifest, options_result=record('options.json', {}),
                  expected_objects_result=record('expected.json', {}))
    for source_name in ('hexagon_options.py', 'torch_mlir_hexagon_launcher.py', 'hexagon_launcher_base.py',
                        'utils.py', 'hexagon_executor.py', 'hexagon_link_options.py', 'compiler.py'):
        blob('triton/triton/backends/qcom_hexagon_backend/'+source_name)
    if packed:
        inputs.update(packed_i8_optimizer=blob('integer-optimizer'),
            packed_i8_runtime=blob('integer-runtime.o'),packed_i8_nm=blob('integer-nm'))
    if effects:
        inputs.update(silu_effects_optimizer=blob('effect-optimizer'),silu_effects_guard=blob('effect-guard.h'))
    if copy:
        owned=compiler_root()/'tools/qwen3/runtime/profile/guarded_memref_copy.cpp'
        inputs.update(guarded_copy_source=blob('guarded-copy.cpp',owned.read_bytes()),
            guarded_copy_runtime=blob('guarded-copy.o',b'\x7fELF\x01\x01'+bytes(10)+struct.pack('<HH',1,164)))
    if multi:
        inputs.update(packed_multi_i8_optimizer=blob('multi-integer-optimizer'),
            packed_multi_i8_runtime=blob('multi-integer-runtime.o',b'\x7fELF\x01\x01'+bytes(10)+struct.pack('<HH',1,164)),
            packed_multi_i8_nm=blob('multi-integer-nm'))
    if multi_effects:
        inputs['multi_silu_effects_optimizer']=blob('multi-effect-optimizer')
    if fixed512:
        owned=compiler_root()/'tools/qwen3/runtime/profile/fixed512_memcpy.cpp'
        inputs.update(fixed512_copy_source=blob('fixed512_memcpy.cpp',owned.read_bytes()),
            fixed512_copy_runtime=blob('fixed512.o',b'\x7fELF\x01\x01'+bytes(10)+struct.pack('<HH',1,164)),
            fixed512_copy_nm=blob('fixed512-nm'))
    if reuse:
        owned=Path(__file__).resolve().parents[3]/'tools/qwen3/reuse_accepted_compiler_objects.py'
        inputs.update(compiler_stage_reuse_source=dict(path=str(owned),sha256=digest_file(owned)),
            compiler_stage_reuse_cold=record('accepted-cold/acceptance.json',{}),
            compiler_stage_reuse_config=record('config.json',{}))
    if prepared_reuse:
        owned=Path(__file__).resolve().parents[3]/'tools/qwen3/reuse_accepted_prepared_linalg.py'
        inputs['prepared_stage_reuse_source']=dict(path=str(owned),sha256=digest_file(owned))
    recipe = record('recipe.json', dict(schema=1, inputs=inputs,
        packed_i8_query1=packed,silu_effects_query1=effects,guarded_copy_query1=copy,triton_python=str(tmp_path/'triton'),
        packed_multi_i8_query2_to32=multi,guarded_copy_query2_to32=dynamic_copy,
        **({'packed_multi_i8_contract':'original_Qwen_query2_to32_integer_ring_matmul_shared_rows_v1'} if multi else {}),
        **({'guarded_copy_dynamic_contract':'original_Qwen_query2_to32_memrefCopy_exact_integer_product_guards_v1'} if dynamic_copy else {}),
        **({'guarded_copy_contract':'original_Qwen_query1_memrefCopy_exact_integer_product_guards_v1'} if copy else {}),
        **({'silu_effects_query2_to32':True,'silu_effects_dynamic_contract':'original_Qwen_query2_to32_silu_readnone_nounwind_willreturn_v1'} if multi_effects else {}),
        **({'fixed512_copy_query2_to32':True,'fixed512_copy_contract':'original_Qwen_query2_to32_memcpy_exact512_inline_or_SDK_v1'} if fixed512 else {}),
        **({'fixed512_copy_query1':True,'fixed512_copy_query1_contract':'original_Qwen_query1_memcpy_exact512_inline_or_SDK_v1'} if fixed512_q1 else {}),
        **({'reuse_exact_compiler_objects':True} if reuse else {}),
        **({'reuse_exact_prepared_linalg':True} if prepared_reuse else {}),
        **({'silu_effects_contract':'original_Qwen_query1_silu_readnone_nounwind_willreturn_v1'} if effects else {})))
    frontend_root = tmp_path/'frontend'
    frontend_root.mkdir()
    build = frontend_root/('a'*64+'-build')
    build.mkdir()
    ir = blob(str(build/'captured-linalg.mlirbc'), b'ML\xefR-test-only')
    frontend = dict(status='captured_AOT_Linalg_cached_NOT_machine_code',
        key='a'*64, build=build.name,
        request=dict(graph=dict(sha256='b'*64), parameters_sha256='c'*64,
                     frontend_sha256='d'*64, function_name='Test', retained_input_indices=[0]),
        artifacts=[dict(path='captured-linalg.mlirbc', sha256=ir['sha256'],
                        bytes=(build/'captured-linalg.mlirbc').stat().st_size)])
    (frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    machine_root = tmp_path/'machine'
    machine_root.mkdir()
    calls = []
    arguments=[]

    def fake_run(argv, **kwargs):
        if argv[1]=='--defined-only':
            if Path(argv[2]).name.startswith('fixed512'):
                return types.SimpleNamespace(returncode=0,stdout='00000000 T __wrap_memcpy\n')
            return types.SimpleNamespace(returncode=0,stdout='00000000 T _mlir_ciface___qwen_i8_multi_matmul\n')
        if argv[1]=='-u':
            if Path(argv[2]).name.startswith('fixed512'):
                return types.SimpleNamespace(returncode=0,stdout=' U __real_memcpy\n')
            return types.SimpleNamespace(returncode=0,stdout=' U _mlir_ciface___qwen_i8_matmul\n U _mlir_ciface___qwen_i8_multi_matmul\n')
        script = Path(argv[1]).name
        calls.append(script)
        arguments.append(argv)
        output = Path(argv[argv.index('--output')+1])
        output.mkdir()
        if script in ('prepare_captured_qwen_hexagon.py','reuse_accepted_prepared_linalg.py'):
            if script=='reuse_accepted_prepared_linalg.py':
                if prepared_outcome in ('miss','corrupt'):
                    record(str(output/'reuse-result.json'),dict(status='prepared_stage_cache_miss' if prepared_outcome=='miss' else 'prepared_stage_reuse_rejected',DSP_calls=0))
                    return types.SimpleNamespace(returncode=10 if prepared_outcome=='miss' else 1)
                record(str(output/'reuse-result.json'),dict(status='exact_accepted_prepared_Linalg_copied_NOT_pipeline_or_DSP_acceptance',
                    DSP_calls=0,new_prepare_optimizer_invocations=0,query_bounds=[2,32],
                    current_recipe={'sha256':argv[argv.index('--recipe-sha256')+1]}))
            graph = blob(str(output/'hexagon-linalg.mlirbc'), b'ML\xefR-prepared-test-only')
            blob(str(output/'adapted-source.mlir.gz'))
            result=dict(output=graph)
            if '--packed-i8-optimizer' in argv or (script=='reuse_accepted_prepared_linalg.py' and packed and not multi):
                result['packed_integer_policy']=dict(coverage=197)
                blob(str(output/'packed-optimizer.log'))
            if '--packed-multi-i8-optimizer' in argv or (script=='reuse_accepted_prepared_linalg.py' and multi):
                result['packed_multi_integer_policy']=dict(coverage=197)
                blob(str(output/'packed-multi-optimizer.log'))
            if '--silu-effects-optimizer' in argv or (script=='reuse_accepted_prepared_linalg.py' and effects and not multi_effects):
                result['silu_effects']=dict(coverage=28)
                blob(str(output/'silu-effects-optimizer.log'))
            if '--multi-silu-effects-optimizer' in argv or (script=='reuse_accepted_prepared_linalg.py' and multi_effects):
                result['multi_silu_effects']=dict(coverage=28)
                blob(str(output/'multi-silu-effects-optimizer.log'))
            record(str(output/'result.json'),result)
        elif script == 'reuse_accepted_compiler_objects.py':
            if reuse_outcome=='corrupt':
                record(str(output/'reuse-result.json'),dict(status='compiler_stage_reuse_rejected'))
                return types.SimpleNamespace(returncode=1)
            if reuse_outcome=='miss':
                record(str(output/'reuse-result.json'),dict(status='compiler_stage_cache_miss',
                    new_machine_code_compilations=0,DSP_calls=0))
                return types.SimpleNamespace(returncode=10)
            code=blob(str(output/'dynamic-hexagon-0.o'),b'ELF-test-only')
            blob(str(output/'compiler-diagnostics.txt'),b'')
            record(str(output/'result.json'),dict(machine_code_compilations=1,objects=[code]))
            record(str(output/'producer-result.json'),dict(historical_machine_code_compilations=1))
            record(str(output/'reuse-result.json'),dict(new_machine_code_compilations=0,
                model_inference_acceptance=False,DSP_calls=0))
        elif script == 'probe_hexagon_dynamic_linalg.py':
            code=blob(str(output/'dynamic-hexagon-0.o'), b'ELF-test-only')
            blob(str(output/'compiler-diagnostics.txt'), b'')
            record(str(output/'result.json'), dict(machine_code_compilations=1,objects=[code]))
        elif script == 'prepare_captured_qwen_q1_caller.py':
            record(str(output/'result.json'), dict(test_only=True))
            blob(str(output/'qwen_captured_q1_wrapper.cpp'))
            blob(str(output/'qwen3_sleef_exp_candidate.h'))
            if '--silu-effects-guard' in argv:
                blob(str(output/'silu_effects_guard.h'))
        elif script in ('relink_captured_hexagon_object.py', 'link_qwen_static_query_entry.py'):
            library = blob(str(output/'model.so'), b'ELF-library-test-only')
            record(str(output/'result.json'), dict(library=library))
        else:
            raise AssertionError(script)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr('hexagon_qwen_captured_machine_cache.subprocess.run', fake_run)
    cache = CapturedQwenMachineCache(machine_root, recipe['path'], recipe['sha256'], frontend_root)
    cache.testing_call_arguments=arguments
    return cache, frontend, calls


@pytest.mark.parametrize('outcome',['hit','miss','corrupt'])
def test_exact_prepare_reuse_skips_only_unchanged_stage_and_rejects_corruption(tmp_path,monkeypatch,outcome):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,
        multi=True,dynamic_copy=True,multi_effects=True,fixed512=True,reuse=True,
        prepared_reuse=True,prepared_outcome=outcome)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract=dict(variables={'query':[2,32]},inputs=[{}]*62,outputs=[{}]*57)
    if outcome=='corrupt':
        with pytest.raises(ValueError,match='preparation authentication failed'):
            cache.get_or_compile(frontend,runtime_contract=contract)
        assert 'prepare_captured_qwen_hexagon.py' not in calls
        assert 'reuse_accepted_compiler_objects.py' not in calls
        assert 'link_qwen_static_query_entry.py' not in calls
        assert not list(cache.root.glob('*.json'))
        return
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    assert calls.count('reuse_accepted_prepared_linalg.py')==1
    assert ('prepare_captured_qwen_hexagon.py' in calls) is (outcome=='miss')
    assert result['model_object_codegen_invocations']==0
    assert calls.count('prepare_captured_qwen_q1_caller.py')==1
    assert calls.count('link_qwen_static_query_entry.py')==1
    assert cache.get_or_compile(frontend,runtime_contract=contract)['cache_hit']
    assert calls.count('reuse_accepted_prepared_linalg.py')==1
    folder=Path(result['directory'])
    proof=folder/('prepared/reuse-result.json' if outcome=='hit' else 'prepared-reuse-miss/reuse-result.json')
    assert proof.is_file()


@pytest.mark.parametrize('outcome',['hit','miss','corrupt'])
def test_exact_stage_reuse_miss_uses_normal_compiler_but_corruption_never_does(tmp_path,monkeypatch,outcome):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,
        multi=True,dynamic_copy=True,multi_effects=True,fixed512=True,reuse=True,reuse_outcome=outcome)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract=dict(variables={'query':[2,32]},inputs=[{}]*62,outputs=[{}]*57)
    if outcome=='corrupt':
        import subprocess
        with pytest.raises(subprocess.CalledProcessError):
            cache.get_or_compile(frontend,runtime_contract=contract)
        assert 'probe_hexagon_dynamic_linalg.py' not in calls
        assert 'link_qwen_static_query_entry.py' not in calls
        assert not (cache.root/next(cache.root.glob('*.lock')).with_suffix('.json').name).exists()
        return
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    assert ('probe_hexagon_dynamic_linalg.py' in calls) is (outcome=='miss')
    assert result['compiler_invocations']==1
    assert result['model_object_codegen_invocations']==(1 if outcome=='miss' else 0)
    assert calls.count('link_qwen_static_query_entry.py')==1
    hit=cache.get_or_compile(frontend,runtime_contract=contract)
    assert hit['cache_hit'] and hit['model_object_codegen_invocations']==0
    assert calls.count('link_qwen_static_query_entry.py')==1
    folder=Path(result['directory'])
    retained=folder/('objects-reuse-miss/reuse-result.json' if outcome=='miss' else 'objects/reuse-result.json')
    retained.write_bytes(b'changed provenance')
    with pytest.raises(ValueError,match='Cached compiler bytes drifted'):
        cache.get_or_compile(frontend,runtime_contract=contract)


def test_stage_reuse_preserves_normal_compilation_for_other_admitted_bounds(tmp_path,monkeypatch):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,
        multi=True,dynamic_copy=True,multi_effects=True,fixed512=True,reuse=True)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract=dict(variables={'query':[3,13]},inputs=[{}]*62,outputs=[{}]*57)
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    assert 'reuse_accepted_compiler_objects.py' not in calls
    assert 'probe_hexagon_dynamic_linalg.py' in calls
    assert result['model_object_codegen_invocations']==1
    assert 'objects/producer-result.json' not in [r['path'] for r in result['artifacts']]


def key_recipe(tmp_path,monkeypatch):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,
        multi=True,dynamic_copy=True,multi_effects=True,fixed512=True,reuse=True,prepared_reuse=True)
    from hexagon_q1_integer_tiles import CONTRACT
    monkeypatch.setattr('hexagon_q1_integer_tiles.validate_runtime',lambda *args:None)
    path=tmp_path/'recipe.json';recipe=json.loads(path.read_text())
    recipe.update(integer_tiles_query1=True,integer_tiles_contract=CONTRACT,
        empty_tensor_cse_query1=True,empty_tensor_cse_contract='original_Qwen_query1_CSE_after_empty_tensor_v1',
        gqa_value_view_query1=True,gqa_value_view_contract='original_Qwen_query1_identity_V_shape_view_v1',
        gqa_value_forward_query1=True,gqa_value_forward_contract='original_Qwen_query1_component_slices_through_proved_V_repeat_v1',
        gqa_key_compact_query1=True,gqa_key_compact_contract='original_Qwen_query1_eight_head_contiguous_K_SCF_copy_v1')
    for name in ('integer_tiles_source','integer_tiles_header','integer_tiles_original_header',
            *('query1_'+kind+'_'+item for kind in ('cse','gqa_value_view','gqa_value_forward','gqa_key_compact')
              for item in ('compiler','sdk_receipt','objects_receipt'))):
        blob=tmp_path/name;blob.write_bytes(name.encode())
        recipe['inputs'][name]=dict(path=str(blob),sha256=digest_file(blob))
    path.write_bytes(canonical(recipe));return cache,frontend,calls,path,recipe


@pytest.mark.parametrize('bounds',[[1,1],[2,32]])
def test_K_recipe_selects_its_own_q1_compiler_and_original_dynamic_compiler(tmp_path,monkeypatch,bounds):
    old,frontend,calls,path,recipe=key_recipe(tmp_path,monkeypatch)
    cache=CapturedQwenMachineCache(old.root,path,digest_file(path),old.frontend_root)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    captured=[]
    class InspectedRequest(Exception):pass
    def inspect(request,callback):
        captured.append(request);raise InspectedRequest
    monkeypatch.setattr(cache.cache,'get_or_compile',inspect)
    with pytest.raises(InspectedRequest):
        cache.get_or_compile(frontend,runtime_contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57))
    selected='query1_gqa_key_compact_compiler' if bounds==[1,1] else 'compiler'
    assert captured[0]['compiler_sha256']==recipe['inputs'][selected]['sha256']
    assert calls==[]


@pytest.mark.parametrize('change',['not-bool','missing-parent','wrong-contract','missing-compiler','source-drift'])
def test_K_recipe_rejects_incoherent_scope_before_any_compiler(tmp_path,monkeypatch,change):
    cache,_,calls,path,recipe=key_recipe(tmp_path,monkeypatch)
    if change=='not-bool':recipe['gqa_key_compact_query1']=1
    elif change=='missing-parent':recipe['gqa_value_forward_query1']=False
    elif change=='wrong-contract':recipe['gqa_key_compact_contract']='toy-rank4-copy'
    elif change=='missing-compiler':del recipe['inputs']['query1_gqa_key_compact_compiler']
    else:Path(recipe['inputs']['query1_gqa_key_compact_compiler']['path']).write_bytes(b'drift')
    path.write_bytes(canonical(recipe))
    with pytest.raises(ValueError):CapturedQwenMachineCache(cache.root,path,digest_file(path),cache.frontend_root)
    assert calls==[]


@pytest.mark.parametrize('bounds',[[1,1],[2,32],[3,13]])
def test_combined_recipe_uses_actual_domain_and_authenticates_each_runtime(tmp_path,monkeypatch,bounds):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,multi=True,dynamic_copy=True)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57)
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    prepare=cache.testing_call_arguments[0]
    assert ('--packed-i8-optimizer' in prepare) is (bounds==[1,1])
    assert ('--silu-effects-optimizer' in prepare) is (bounds==[1,1])
    assert ('--packed-multi-i8-optimizer' in prepare) is (bounds==[2,32])
    link=next(a for a in cache.testing_call_arguments if Path(a[1]).name=='link_qwen_static_query_entry.py')
    assert ('--guarded-copy-runtime' in link) is (bounds in ([1,1],[2,32]))
    assert ('--guarded-copy-dynamic' in link) is (bounds==[2,32])
    folder=Path(result['directory'])
    if bounds in ([1,1],[2,32]):
        command=json.loads((folder/'packed-manifest.json').read_text())['link_command']
        assert ('packed-multi-i8-runtime.o' in command) is (bounds==[2,32])
        assert ('/integer-runtime.o' in command) is (bounds==[1,1])
    assert cache.get_or_compile(frontend,runtime_contract=contract)['cache_hit'] and len(calls)==4
    if bounds==[2,32]:
        (folder/'runtime/packed-multi-i8-runtime.o').write_bytes(b'drift')
        with pytest.raises(ValueError,match='Cached compiler bytes drifted'):
            cache.get_or_compile(frontend,runtime_contract=contract)


@pytest.mark.parametrize('change',['integer-contract','integer-not-bool','integer-architecture','missing-runtime','copy-contract','copy-not-bool'])
def test_multi_recipe_rejects_incoherent_or_corrupt_dependencies_before_compile(tmp_path,monkeypatch,change):
    cache,_,calls=fixture_cache(tmp_path,monkeypatch,packed=True,copy=True,multi=True,dynamic_copy=True)
    path=tmp_path/'recipe.json';recipe=json.loads(path.read_text())
    if change=='integer-contract':recipe['packed_multi_i8_contract']='different'
    elif change=='integer-not-bool':recipe['packed_multi_i8_query2_to32']=1
    elif change=='missing-runtime':del recipe['inputs']['packed_multi_i8_runtime']
    elif change=='copy-contract':recipe['guarded_copy_dynamic_contract']='different'
    elif change=='copy-not-bool':recipe['guarded_copy_query2_to32']=1
    else:
        row=recipe['inputs']['packed_multi_i8_runtime'];Path(row['path']).write_bytes(b'wrong-machine')
        row['sha256']=digest_file(row['path'])
    path.write_bytes(canonical(recipe))
    with pytest.raises(ValueError):
        CapturedQwenMachineCache(cache.root,path,digest_file(path),cache.frontend_root)
    assert not calls


@pytest.mark.parametrize('bounds',[[1,1],[2,32]])
@pytest.mark.parametrize('effects',[False,True])
def test_copy_recipe_only_links_actual_query1_and_authenticates_owned_object(tmp_path,monkeypatch,bounds,effects):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=effects,copy=True)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57)
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    selected=bounds==[1,1]
    link=next(a for a in cache.testing_call_arguments if Path(a[1]).name=='link_qwen_static_query_entry.py')
    assert ('--guarded-copy-runtime' in link) is selected
    assert ('--guarded-copy-source' in link) is selected
    assert '--guarded-copy-runtime' not in cache.testing_call_arguments[0]
    folder=Path(result['directory']);runtime=folder/'runtime/guarded-copy-runtime.o'
    assert runtime.exists() is selected
    assert cache.get_or_compile(frontend,runtime_contract=contract)['cache_hit'] and len(calls)==4
    if selected:
        runtime.write_bytes(b'CORRUPTED')
        with pytest.raises(ValueError,match='Cached compiler bytes drifted'):
            cache.get_or_compile(frontend,runtime_contract=contract)


def test_copy_route_rejects_missing_actual_shape_before_any_compile(tmp_path,monkeypatch):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,copy=True)
    cache.silu_effects_query1=False;cache.packed_i8_query1=False
    with pytest.raises(ValueError,match='Guarded byte-copy query1 route requires'):
        cache.get_or_compile(frontend)
    assert calls==[]


@pytest.mark.parametrize('change',['wrong-source','wrong-architecture','missing-runtime','wrong-contract','not-bool'])
def test_copy_recipe_rejects_drift_before_compilation(tmp_path,monkeypatch,change):
    cache,_,calls=fixture_cache(tmp_path,monkeypatch,packed=True,copy=True)
    path=tmp_path/'recipe.json';recipe=json.loads(path.read_text())
    if change in ('wrong-source','wrong-architecture'):
        row=recipe['inputs']['guarded_copy_source' if change=='wrong-source' else 'guarded_copy_runtime']
        Path(row['path']).write_bytes(b'changed')
        row['sha256']=digest_file(row['path'])
    elif change=='missing-runtime':del recipe['inputs']['guarded_copy_runtime']
    elif change=='wrong-contract':recipe['guarded_copy_contract']='different_copy'
    else:recipe['guarded_copy_query1']=1
    path.write_bytes(canonical(recipe))
    with pytest.raises(ValueError):
        CapturedQwenMachineCache(cache.root,path,digest_file(path),cache.frontend_root)
    assert calls==[]


@pytest.mark.parametrize('bounds',[[1,1],[2,32]])
def test_effect_recipe_pairs_query1_policy_guard_and_cached_bytes(tmp_path,monkeypatch,bounds):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57)
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    selected=bounds==[1,1]
    assert ('--silu-effects-optimizer' in cache.testing_call_arguments[0]) is selected
    caller=next(argv for argv in cache.testing_call_arguments if Path(argv[1]).name=='prepare_captured_qwen_q1_caller.py')
    assert ('--silu-effects-guard' in caller) is selected
    folder=Path(result['directory'])
    assert (folder/'prepared/silu-effects-optimizer.log').exists() is selected
    assert (folder/'caller/silu_effects_guard.h').exists() is selected
    assert cache.get_or_compile(frontend,runtime_contract=contract)['cache_hit'] and len(calls)==4
    if selected:
        (folder/'caller/silu_effects_guard.h').write_bytes(b'changed runtime guard')
        with pytest.raises(ValueError,match='Cached compiler bytes drifted'):
            cache.get_or_compile(frontend,runtime_contract=contract)


def test_effect_route_rejects_missing_live_shape_contract(tmp_path,monkeypatch):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,effects=True)
    with pytest.raises(ValueError,match='SiLU effect query1 route requires'):
        cache.get_or_compile(frontend)
    assert calls==[]


def test_effect_guard_corruption_rejected_before_compile(tmp_path,monkeypatch):
    cache,_,_=fixture_cache(tmp_path,monkeypatch,effects=True)
    (tmp_path/'effect-guard.h').write_bytes(b'changed')
    with pytest.raises(ValueError,match='dependency changed'):
        CapturedQwenMachineCache(cache.root,tmp_path/'recipe.json',digest_file(tmp_path/'recipe.json'),cache.frontend_root)


def test_effect_recipe_requires_explicit_original_math_contract(tmp_path,monkeypatch):
    cache,_,_=fixture_cache(tmp_path,monkeypatch,effects=True)
    path=tmp_path/'recipe.json';recipe=json.loads(path.read_text());recipe['silu_effects_contract']='foreign_math'
    path.write_bytes(canonical(recipe))
    with pytest.raises(ValueError,match='Explicit original guarded SiLU'):
        CapturedQwenMachineCache(cache.root,path,digest_file(path),cache.frontend_root)


@pytest.mark.parametrize('bounds',[[1,1],[2,32]])
def test_opt_in_integer_recipe_routes_only_real_query1_contract(tmp_path,monkeypatch,bounds):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    original_manifest=(tmp_path/'manifest.json').read_bytes()
    contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57)
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    packed=bounds==[1,1]
    assert ('--packed-i8-optimizer' in cache.testing_call_arguments[0]) is packed
    folder=Path(result['directory'])
    assert (folder/'packed-import-proof.json').exists() is packed
    assert (folder/'packed-manifest.json').exists() is packed
    if packed:
        modified=json.loads((folder/'packed-manifest.json').read_text())
        assert str(tmp_path/'integer-runtime.o') in modified['link_command']
    assert (tmp_path/'manifest.json').read_bytes()==original_manifest
    assert cache.get_or_compile(frontend,runtime_contract=contract)['cache_hit']
    assert len(calls)==4


def test_integer_route_rejects_absent_shape_contract_before_compile(tmp_path,monkeypatch):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True)
    with pytest.raises(ValueError,match='actual captured shape contract'):
        cache.get_or_compile(frontend)
    assert calls==[]


def test_integer_runtime_corruption_refused_at_recipe_admission(tmp_path,monkeypatch):
    cache,_,_=fixture_cache(tmp_path,monkeypatch,packed=True)
    (tmp_path/'integer-runtime.o').write_bytes(b'changed')
    with pytest.raises(ValueError,match='dependency changed'):
        CapturedQwenMachineCache(cache.root,tmp_path/'recipe.json',digest_file(tmp_path/'recipe.json'),cache.frontend_root)


def test_one_cold_build_then_authenticated_reuse(tmp_path, monkeypatch):
    cache, frontend, calls = fixture_cache(tmp_path, monkeypatch)
    first = cache.get_or_compile(frontend)
    assert first['machine_code_compilations'] == 1 and not first['cache_hit']
    assert calls == ['prepare_captured_qwen_hexagon.py', 'probe_hexagon_dynamic_linalg.py',
                     'relink_captured_hexagon_object.py']
    second = cache.get_or_compile(frontend)
    assert second['key'] == first['key'] and second['cache_hit']
    assert second['machine_code_compilations'] == 0 and len(calls) == 3


def test_library_corruption_refused_without_rebuild(tmp_path, monkeypatch):
    cache, frontend, calls = fixture_cache(tmp_path, monkeypatch)
    first = cache.get_or_compile(frontend)
    (Path(first['directory'])/'linked/model.so').write_bytes(b'CORRUPT')
    with pytest.raises(ValueError, match='Cached compiler bytes drifted'):
        cache.get_or_compile(frontend)
    assert len(calls) == 3


def test_live_capture_entry_mismatch_stops_before_compiler(tmp_path, monkeypatch):
    cache, frontend, calls = fixture_cache(tmp_path, monkeypatch)
    frontend['request']['graph']['sha256'] = 'e'*64
    with pytest.raises(ValueError, match='live capture binding differ'):
        cache.get_or_compile(frontend)
    assert calls == []


@pytest.mark.parametrize('arity,bounds', [(62, [1, 1]), (65, [2, 32]), (68, [2, 32])])
def test_actual_arity_builds_own_caller_and_caches_it(tmp_path, monkeypatch, arity, bounds):
    cache, frontend, calls = fixture_cache(tmp_path, monkeypatch)
    frontend['request']['retained_input_indices'] = list(range(arity))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract = dict(variables={'query': bounds}, inputs=[{}]*arity, outputs=[{}]*57)
    result = cache.get_or_compile(frontend, runtime_contract=contract)
    assert calls == ['prepare_captured_qwen_hexagon.py', 'probe_hexagon_dynamic_linalg.py',
                     'prepare_captured_qwen_q1_caller.py', 'link_qwen_static_query_entry.py']
    second = cache.get_or_compile(frontend, runtime_contract=contract)
    assert second['cache_hit'] and second['key'] == result['key'] and len(calls) == 4
    (Path(result['directory'])/'caller/qwen_captured_q1_wrapper.cpp').write_bytes(b'CHANGED')
    with pytest.raises(ValueError, match='Cached compiler bytes drifted'):
        cache.get_or_compile(frontend, runtime_contract=contract)


def test_reject_specialized_contract_on_other_capture(tmp_path, monkeypatch):
    cache, frontend, calls = fixture_cache(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='actual bounded captured contract'):
        cache.get_or_compile(frontend, runtime_contract=dict(
            variables={'query': [1, 1]}, inputs=[{}]*62, outputs=[{}]*57))
    assert calls == []


def test_wrapper_generator_change_invalidates_machine_cache(tmp_path, monkeypatch):
    cache, frontend, calls = fixture_cache(tmp_path, monkeypatch)
    first = cache.get_or_compile(frontend)
    (tmp_path/'triton/triton/backends/qcom_hexagon_backend/hexagon_launcher_base.py').write_bytes(b'NEW GENERATOR')
    recipe = tmp_path/'recipe.json'
    rebuilt_cache = CapturedQwenMachineCache(cache.root, recipe, digest_file(recipe), cache.frontend_root)
    changed = rebuilt_cache.get_or_compile(frontend)
    assert changed['key'] != first['key'] and changed['machine_code_compilations'] == 1
    assert len(calls) == 6


@pytest.mark.parametrize('bounds',[[1,1],[2,32],[3,13]])
def test_multi_effects_preserve_query1_and_pair_actual_dynamic_policy_guard_and_cache(tmp_path,monkeypatch,bounds):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,multi=True,dynamic_copy=True,multi_effects=True)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57)
    result=cache.get_or_compile(frontend,runtime_contract=contract)
    prepare=cache.testing_call_arguments[0]
    assert ('--silu-effects-optimizer' in prepare) is (bounds==[1,1])
    assert ('--multi-silu-effects-optimizer' in prepare) is (bounds==[2,32])
    assert ('--packed-multi-i8-optimizer' in prepare) is (bounds==[2,32])
    caller=next(argv for argv in cache.testing_call_arguments if Path(argv[1]).name=='prepare_captured_qwen_q1_caller.py')
    assert ('--silu-effects-guard' in caller) is (bounds in [[1,1],[2,32]])
    assert ('--silu-effects-dynamic' in caller) is (bounds==[2,32])
    folder=Path(result['directory'])
    assert (folder/'prepared/multi-silu-effects-optimizer.log').exists() is (bounds==[2,32])
    assert cache.get_or_compile(frontend,runtime_contract=contract)['cache_hit'] and len(calls)==4
    if bounds==[2,32]:
        (folder/'prepared/multi-silu-effects-optimizer.log').write_bytes(b'changed coverage')
        with pytest.raises(ValueError,match='Cached compiler bytes drifted'):
            cache.get_or_compile(frontend,runtime_contract=contract)


@pytest.mark.parametrize('change',['wrong-contract','not-bool','missing-optimizer','optimizer-drift','missing-q1-effects','missing-multi-integer','missing-dynamic-copy'])
def test_multi_effect_recipe_rejects_incomplete_or_corrupt_contract_before_compilation(tmp_path,monkeypatch,change):
    cache,_,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,multi=True,dynamic_copy=True,multi_effects=True)
    path=tmp_path/'recipe.json';recipe=json.loads(path.read_text())
    if change=='wrong-contract':recipe['silu_effects_dynamic_contract']='foreign'
    elif change=='not-bool':recipe['silu_effects_query2_to32']=1
    elif change=='missing-optimizer':del recipe['inputs']['multi_silu_effects_optimizer']
    elif change=='optimizer-drift':Path(recipe['inputs']['multi_silu_effects_optimizer']['path']).write_bytes(b'changed')
    else:recipe[{'missing-q1-effects':'silu_effects_query1','missing-multi-integer':'packed_multi_i8_query2_to32','missing-dynamic-copy':'guarded_copy_query2_to32'}[change]]=False
    path.write_bytes(canonical(recipe))
    with pytest.raises(ValueError):CapturedQwenMachineCache(cache.root,path,digest_file(path),cache.frontend_root)
    assert not calls
