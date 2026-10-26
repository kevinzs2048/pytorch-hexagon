"""Runtime admission cannot turn a missing build tool into relaxed model identity."""
import copy
import json
from pathlib import Path

import pytest
import build
from build import entry
from runtime_dependencies import runtime_lock, verify_runtime_dependencies


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    def file(name, data=b'pinned'):
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
        return p
    compiler=file('compiler.so'); hf=file('hf/model.safetensors'); lld=file('ld.lld')
    constant=file('constants/weights.so')
    (tmp_path/'model-constants.lock.json').write_text(json.dumps(dict(libraries=[dict(
        name=constant.name,bytes=constant.stat().st_size,sha256=entry(constant)['sha256'])])))
    monkeypatch.setattr(build,'HERE',tmp_path)
    config=dict(compiler=str(compiler),hf=str(hf.parent),lowering_opt=str(tmp_path/'missing-old-opt'),
                lld=str(lld),constants_dir=str(constant.parent))
    inputs={k:entry(file(k)) for k in ('attention_optimizer','sums_optimizer','knorm_optimizer')}
    inputs['compiler']=entry(compiler)
    recipe=file('recipe.json',json.dumps(dict(schema=1,inputs=inputs)).encode())
    ref=entry(recipe)
    worker=['python','worker.py','--machine-recipe',str(recipe),'--machine-recipe-sha256',ref['sha256']]
    lock=dict(files=[dict(root=k,relative=r,bytes=p.stat().st_size,sha256=entry(p)['sha256'])
                    for k,r,p in [('compiler','',compiler),('hf',hf.name,hf)]]+
              [dict(root='lowering_opt',relative='',bytes=100,sha256='f'*64)])
    return config,lock,ref,worker


def test_runtime_preserves_original_build_failure(fixture):
    config,lock,ref,worker=fixture
    with pytest.raises(ValueError,match='lowering_opt'):
        build.verify_dependencies(config,lock)
    checked,proof=verify_runtime_dependencies(config,lock,ref,worker)
    assert {r['root'] for r in checked}=={'compiler','hf'}
    assert proof['excluded_build_only_files']==lock['files'][-1:]
    assert len(lock['files'])==3


@pytest.mark.parametrize('target',['compiler','hf','constants'])
def test_corrupt_execution_bytes_still_rejected(fixture,target):
    config,lock,ref,worker=fixture
    path={'compiler':Path(config['compiler']),'hf':Path(config['hf'])/'model.safetensors',
          'constants':Path(config['constants_dir'])/'weights.so'}[target]
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_runtime_dependencies(config,lock,ref,worker)


def test_changed_recipe_or_worker_not_admitted(fixture):
    config,lock,ref,worker=fixture
    wrong=worker.copy();wrong[-1]='0'*64
    with pytest.raises(ValueError):runtime_lock(config,lock,ref,wrong)
    Path(ref['path']).write_bytes(b'{}')
    with pytest.raises(ValueError):runtime_lock(config,lock,ref,worker)


def test_actual_legacy_recipe_cannot_exclude_its_tool(fixture):
    config,lock,ref,worker=fixture
    recipe=Path(ref['path']);r=json.loads(recipe.read_text())
    legacy=Path(config['lowering_opt']);legacy.write_bytes(b'pinned')
    r['inputs']['attention_optimizer']=entry(legacy)
    recipe.write_text(json.dumps(r));ref=entry(recipe);worker[-1]=ref['sha256']
    with pytest.raises(ValueError,match='actually requires'):
        runtime_lock(config,lock,ref,worker)


def test_extra_or_unknown_exclusions_rejected(fixture):
    config,lock,ref,worker=fixture
    changed=copy.deepcopy(lock);changed['files'].append(changed['files'][-1])
    with pytest.raises(ValueError):runtime_lock(config,changed,ref,worker)
