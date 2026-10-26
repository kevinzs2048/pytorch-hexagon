"""Compiler binding tests only; actual SDK/DSP/model acceptance is separate."""
import json
from pathlib import Path

import pytest

from hexagon_compiled_artifact_cache import canonical, digest_file
from hexagon_fixed512_memcpy import add_link_runtime
from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
from test_captured_machine_cache import fixture_cache


@pytest.mark.parametrize('bounds', [[1,1],[2,32],[3,13]])
def test_explicit_query1_copy_uses_original_owned_wrapper_and_unchanged_multi_domain(tmp_path,monkeypatch,bounds):
    cache,frontend,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,
        copy=True,multi=True,dynamic_copy=True,multi_effects=True,fixed512=True,fixed512_q1=True)
    frontend['request']['retained_input_indices']=list(range(62))
    (cache.frontend_root/(frontend['key']+'.json')).write_bytes(canonical(frontend))
    result=cache.get_or_compile(frontend,runtime_contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57))
    link=next(argv for argv in cache.testing_call_arguments if Path(argv[1]).name=='link_qwen_static_query_entry.py')
    assert ('--fixed512-copy-runtime' in link) is (bounds in ([1,1],[2,32]))
    assert ('--fixed512-copy-query1' in link) is (bounds==[1,1])
    assert cache.get_or_compile(frontend,runtime_contract=dict(variables={'query':bounds},inputs=[{}]*62,outputs=[{}]*57))['cache_hit']


@pytest.mark.parametrize('key,value',[('fixed512_copy_query1',1),('fixed512_copy_query1_contract','wrong'),('fixed512_copy_query2_to32',False)])
def test_query1_copy_requires_explicit_original_contract_before_compile(tmp_path,monkeypatch,key,value):
    cache,_,calls=fixture_cache(tmp_path,monkeypatch,packed=True,effects=True,copy=True,
        multi=True,dynamic_copy=True,multi_effects=True,fixed512=True,fixed512_q1=True)
    path=tmp_path/'recipe.json';recipe=json.loads(path.read_text());recipe[key]=value;path.write_bytes(canonical(recipe))
    with pytest.raises(ValueError):CapturedQwenMachineCache(cache.root,path,digest_file(path),cache.frontend_root)
    assert calls==[]


@pytest.mark.parametrize('bounds', [[1, 1], [2, 32], [3, 13]])
def test_only_original_dynamic_domain_links_owned_copy_and_authenticates_cache(tmp_path, monkeypatch, bounds):
    cache, frontend, calls = fixture_cache(tmp_path, monkeypatch, packed=True, effects=True,
        copy=True, multi=True, dynamic_copy=True, multi_effects=True, fixed512=True)
    frontend['request']['retained_input_indices'] = list(range(62))
    (cache.frontend_root / (frontend['key'] + '.json')).write_bytes(canonical(frontend))
    contract = dict(variables={'query': bounds}, inputs=[{}] * 62, outputs=[{}] * 57)
    result = cache.get_or_compile(frontend, runtime_contract=contract)
    link = next(a for a in cache.testing_call_arguments if Path(a[1]).name == 'link_qwen_static_query_entry.py')
    selected = bounds == [2, 32]
    for flag in ('--fixed512-copy-runtime', '--fixed512-copy-source', '--fixed512-copy-nm'):
        assert (flag in link) is selected
        assert flag not in cache.testing_call_arguments[0]
    runtime = Path(result['directory']) / 'runtime/fixed512-memcpy.o'
    assert runtime.exists() is selected
    assert cache.get_or_compile(frontend, runtime_contract=contract)['cache_hit'] and len(calls) == 4
    if selected:
        runtime.write_bytes(b'CORRUPT')
        with pytest.raises(ValueError, match='Cached compiler bytes drifted'):
            cache.get_or_compile(frontend, runtime_contract=contract)


def test_original_twenty_dependency_recipe_keeps_new_flag_off(tmp_path, monkeypatch):
    cache, _, calls = fixture_cache(tmp_path, monkeypatch, packed=True, effects=True,
        copy=True, multi=True, dynamic_copy=True, multi_effects=True)
    assert cache.fixed512_copy_query2_to32 is False
    assert 'hexagon_fixed512_memcpy.py' not in cache.sources and calls == []


@pytest.mark.parametrize('change', ['not-bool', 'wrong-contract', 'no-multi-silu', 'missing-source',
                                  'wrong-source', 'wrong-machine', 'recursive-sdk-fallback'])
def test_incoherent_recipe_or_runtime_is_rejected_before_any_model_compile(tmp_path, monkeypatch, change):
    cache, _, calls = fixture_cache(tmp_path, monkeypatch, packed=True, effects=True,
        copy=True, multi=True, dynamic_copy=True, multi_effects=True, fixed512=True)
    path = tmp_path / 'recipe.json'
    recipe = json.loads(path.read_text())
    if change == 'not-bool':
        recipe['fixed512_copy_query2_to32'] = 1
    elif change == 'wrong-contract':
        recipe['fixed512_copy_contract'] = 'other'
    elif change == 'no-multi-silu':
        recipe['silu_effects_query2_to32'] = False
    elif change == 'missing-source':
        del recipe['inputs']['fixed512_copy_source']
    elif change in ('wrong-source', 'wrong-machine'):
        row = recipe['inputs']['fixed512_copy_source' if change == 'wrong-source' else 'fixed512_copy_runtime']
        Path(row['path']).write_bytes(b'changed')
        row['sha256'] = digest_file(row['path'])
    else:
        import hexagon_fixed512_memcpy
        from types import SimpleNamespace
        original = hexagon_fixed512_memcpy.subprocess.run
        def recursive_nm(argv, **kwargs):
            if Path(argv[-1]).name.startswith('fixed512'):
                return SimpleNamespace(stdout='00000000 T __wrap_memcpy\n' if '--defined-only' in argv else ' U memcpy\n')
            return original(argv, **kwargs)
        monkeypatch.setattr('hexagon_fixed512_memcpy.subprocess.run', recursive_nm)
    path.write_bytes(canonical(recipe))
    with pytest.raises(ValueError, match='nonrecursive SDK fallback' if change == 'recursive-sdk-fallback' else None):
        CapturedQwenMachineCache(cache.root, path, digest_file(path), cache.frontend_root)
    assert calls == []


def test_memcpy_link_preserves_existing_guarded_copy_and_original_arguments():
    original = ['compiler', 'caller.cpp', 'model.o', 'guarded.o', '-Wl,--wrap=memrefCopy', '-o', 'output.so']
    result = add_link_runtime(original, '/owned/fixed512.o')
    assert result == original[:-2] + ['/owned/fixed512.o', '-Wl,--wrap=memcpy'] + original[-2:]
    q1 = add_link_runtime(original, '/owned/fixed512.o', query1=True)
    assert q1 == original[:1] + ['/owned/fixed512.o', '-Wl,--wrap=memcpy'] + original[1:]
    assert original[-2:] == ['-o', 'output.so']
    for invalid in (original + ['extra'], result,
                    [word for word in original if word != '-Wl,--wrap=memrefCopy']):
        with pytest.raises(ValueError):
            add_link_runtime(invalid, '/owned/fixed512.o')
