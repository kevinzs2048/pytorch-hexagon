"""Actual framework miss/hit must preserve alias and input mutation semantics.

These small CPU graphs verify the cache protocol only, not Qwen numerical or
DSP performance acceptance. The production adapter never calls graph.forward.
"""
import hashlib
from pathlib import Path
import platform
import json
import subprocess
import sys

import pytest
import torch

from qwen3.runtime.support.native_aot_output import native_cache_scope,namespace_digest,NativeAOTOutput
from qwen3.runtime.support import native_aot_output


def manifest():
    return dict(adapter_sha256=hashlib.sha256(Path(native_aot_output.__file__).read_bytes()).hexdigest(),
        native_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_revision='2'*40,recipe_sha256='3'*64,
        runtime_catalog_sha256='4'*64,model_parameter_sha256='5'*64,module_constants_sha256='6'*64,
        torch_version=torch.__version__,torch_git=torch.version.git_version,
        python_version=platform.python_version(),architecture=platform.machine())


def function(x):
    x.add_(1)
    return x.view(-1),x.sin()


def backend(graph,inputs):
    import torch._dynamo.backends.common as common
    from torch._functorch.aot_autograd import make_boxed_func
    def compiler(aten,examples):
        # CPU execution is exclusive to this toy framework-protocol test.
        return make_boxed_func(aten.forward)
    def no_backward(*unused):raise RuntimeError('Inference test only')
    return common.aot_autograd(fw_compiler=compiler,inference_compiler=compiler,
        bw_compiler=no_backward)(graph,inputs)


def run_once(root,events,n):
    torch._dynamo.reset()
    x=torch.arange(n*8,dtype=torch.float32).reshape(n,8)
    if n!=1:torch._dynamo.mark_dynamic(x,0,min=2,max=32)
    with native_cache_scope(manifest(),root,events),torch.no_grad():
        outputs=torch.compile(function,backend=backend,dynamic=True,fullgraph=True)(x)
    expected=torch.arange(n*8,dtype=torch.float32).reshape(n,8)+1
    torch.testing.assert_close(x,expected,rtol=0,atol=0)
    torch.testing.assert_close(outputs[0],expected.view(-1),rtol=0,atol=0)
    torch.testing.assert_close(outputs[1],expected.sin(),rtol=0,atol=0)
    assert outputs[0].untyped_storage()._cdata==x.untyped_storage()._cdata
    return outputs


def test_real_cache_hit_rebinds_current_scope_with_dynamic_alias_and_mutation(tmp_path):
    events=[]
    run_once(tmp_path,events,3)
    assert [e['stage'] for e in events]==['original_native_compiler_on_AOT_miss']
    run_once(tmp_path,events,7)
    assert [e['stage'] for e in events]==[
        'original_native_compiler_on_AOT_miss','original_native_compiler_on_AOT_hit_rebind']


@pytest.mark.parametrize('key',['native_source_sha256','recipe_sha256','model_parameter_sha256',
    'module_constants_sha256','numerical_revision'])
def test_backend_and_model_changes_get_different_namespaces(key):
    original=manifest();changed=dict(original);changed[key]='a'*len(changed[key])
    assert namespace_digest(original)!=namespace_digest(changed)


def test_wrong_adapter_source_rejected():
    changed=manifest();changed['adapter_sha256']='0'*64
    with pytest.raises(ValueError):namespace_digest(changed)


def test_live_artifact_cannot_load_without_matching_compiler_scope():
    artifact=object.__new__(NativeAOTOutput);artifact._namespace='unmatched';artifact._call=None
    with pytest.raises(ValueError):artifact.post_compile([],None,{})


def test_actual_quantized_subclass_leaves_rebound_on_cache_hit(tmp_path):
    from torchao.quantization import quantize_,Int8DynamicActivationInt8WeightConfig
    events=[]
    for seed,n in [(42,3),(43,7)]:
        torch._dynamo.reset();torch.manual_seed(seed)
        model=torch.nn.Linear(32,32,bias=False).eval().requires_grad_(False)
        quantize_(model,Int8DynamicActivationInt8WeightConfig(set_inductor_config=False))
        assert type(model.weight).__name__=='LinearActivationQuantizedTensor'
        x=torch.randn(n,32);torch._dynamo.mark_dynamic(x,0,min=2,max=32)
        with torch.no_grad():expected=model(x)
        with native_cache_scope(manifest(),tmp_path,events),torch.no_grad():
            actual=torch.compile(model,backend=backend,dynamic=True,fullgraph=True)(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert [e['stage'] for e in events]==[
        'original_native_compiler_on_AOT_miss','original_native_compiler_on_AOT_hit_rebind']
    assert events[-1]['flattened_input_count']>events[-1]['dynamo_input_count']


def test_two_fresh_processes_share_disk_artifact_without_live_callbacks(tmp_path):
    program=('import json,sys;from pathlib import Path;from test_native_aot_output import run_once;'
             'events=[];run_once(Path(sys.argv[1]),events,int(sys.argv[2]));print(json.dumps(events))')
    all_events=[]
    for n in (3,7):
        result=subprocess.run([sys.executable,'-c',program,str(tmp_path),str(n)],
            cwd=Path(__file__).parent,capture_output=True,text=True,check=True,timeout=90)
        all_events.extend(json.loads(result.stdout))
    assert [event['stage'] for event in all_events]==[
        'original_native_compiler_on_AOT_miss','original_native_compiler_on_AOT_hit_rebind']


def test_singleton_and_bounded_dynamic_graphs_remain_separate(tmp_path):
    events=[]
    for n in (2,1,32,1):run_once(tmp_path,events,n)
    assert [event['stage'] for event in events]==[
        'original_native_compiler_on_AOT_miss','original_native_compiler_on_AOT_miss',
        'original_native_compiler_on_AOT_hit_rebind','original_native_compiler_on_AOT_hit_rebind']
    assert events[0]['graph']!=events[1]['graph']
    assert events[0]['graph']==events[2]['graph'] and events[1]['graph']==events[3]['graph']
