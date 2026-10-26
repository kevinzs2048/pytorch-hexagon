"""Experimental bundled AOT artifact; always rebind the original native compiler.

Framework cache entries own all runtime wrappers and guards. This OutputCode
stores the AOT graph and its input descriptors, never a model, DSP owner or live
compiler closure. Loading invokes the current compiler's original full binding
and validation path. It does not execute the serialized graph on the CPU.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import time

import torch
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
from torch._inductor.output_code import OutputCode


_binding = ContextVar('native_aot_current_binding', default=None)


def namespace_digest(manifest):
    required={'adapter_sha256','native_source_sha256','numerical_revision',
              'recipe_sha256','runtime_catalog_sha256','model_parameter_sha256',
              'module_constants_sha256','torch_version','torch_git','python_version','architecture'}
    if set(manifest)!=required or manifest['torch_version']!=torch.__version__:
        raise ValueError('Complete explicit backend/model/ABI/framework namespace required')
    for key,value in manifest.items():
        if not isinstance(value,str) or not value:
            raise ValueError('Nonempty namespace strings required')
        if key.endswith('_sha256') and (len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Full SHA256 namespace pins required')
    if manifest['adapter_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Current adapter executable differs from namespace')
    if manifest['torch_git']!=torch.version.git_version:
        raise ValueError('Current framework Git revision differs from namespace')
    if manifest['python_version']!=platform.python_version() or manifest['architecture']!=platform.machine():
        raise ValueError('Current interpreter and architecture differ from cache namespace')
    if len(manifest['numerical_revision'])!=40 or any(c not in '0123456789abcdef' for c in manifest['numerical_revision']):
        raise ValueError('Full numerical source revision required')
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def current_binding(expected):
    value=_binding.get()
    if value is None or value['namespace']!=expected:
        raise ValueError('Fresh matching native compiler scope required; no stale callable')
    return value


@contextmanager
def compiler_scope(namespace,compiler,identity,events):
    token=_binding.set(dict(namespace=namespace,compiler=compiler,identity=identity,events=events))
    try:yield
    finally:_binding.reset(token)


class NativeAOTOutput(OutputCode):
    def __init__(self,graph,examples):
        super().__init__()
        value=_binding.get()
        if value is None:raise ValueError('Native compiler binding scope required')
        tracing=torch._guards.TracingContext.get()
        metadata=tracing.fw_metadata
        if metadata is None or len(metadata.input_info)!=len(examples):
            raise ValueError('Authoritative complete flattened AOT input metadata required')
        self._namespace=value['namespace']
        self._graph=graph
        self._input_info=copy.deepcopy(metadata.input_info)
        self._identity=value['identity'](graph)
        self._serialized=None
        self._serialized_sha256=None
        self._boxed_call=True
        self._call=value['compiler'](graph,examples)
        if getattr(self._call,'_boxed_call',False) is not True:
            raise ValueError('Original compiler must retain the boxed input convention')
        value['events'].append(dict(stage='original_native_compiler_on_AOT_miss',graph=self._identity,
                                    completed_ns=time.perf_counter_ns()))

    def __call__(self,inputs):
        if self._call is None:
            raise RuntimeError('Serialized artifact requires fresh native rebind before execution')
        return self._call(inputs)

    def prepare_for_serialization(self):
        from torch.fx._graph_pickler import GraphPickler,Options
        from capture_qwen_symbolic_graph_carrier import allowed_carrier_operator
        current_binding(self._namespace)
        if self._graph is not None:
            # Default filter excludes framework stacks with process pointers.
            # Authoritative val/desc metadata stays; changed identity is rejected.
            data=GraphPickler.dumps((self._graph,self._input_info),
                                   Options(ops_filter=allowed_carrier_operator))
            self._serialized=data
            self._serialized_sha256=hashlib.sha256(data).hexdigest()
        if self._serialized is None:raise ValueError('Complete AOT artifact required')
        self._graph=None
        self._input_info=None
        self._call=None

    def post_compile(self,example_inputs,constants,graph_kwargs):
        if self._call is not None:return
        from torch._guards import detect_fake_mode
        from torch.fx._graph_pickler import GraphPickler
        value=current_binding(self._namespace)
        if self._serialized is None or hashlib.sha256(self._serialized).hexdigest()!=self._serialized_sha256:
            raise ValueError('Serialized native graph bytes changed')
        fake_mode=detect_fake_mode(example_inputs)
        if fake_mode is None:raise ValueError('Current framework fake mode required for rebind')
        graph,input_info=GraphPickler.loads(self._serialized,fake_mode)
        if not isinstance(graph,torch.fx.GraphModule):
            raise ValueError('Restored AOT GraphModule required')
        graph.recompile()
        if value['identity'](graph)!=self._identity:
            raise ValueError('Complete symbolic graph identity changed during reload')
        nodes=[n for n in graph.graph.nodes if n.op=='placeholder']
        if len(nodes)!=len(input_info) or any('val' not in node.meta or 'desc' not in node.meta for node in nodes):
            raise ValueError('Every original AOT input needs authoritative value and provenance descriptor')
        # Cache load receives unflattened Dynamo examples, not compiler inputs.
        # Original compiler resolves each saved desc against its fresh Dynamo
        # arguments, including subclass leaves, and checks all frozen bytes.
        examples=[node.meta['val'] for node in nodes]
        tracing=torch._guards.TracingContext.get();previous=tracing.fw_metadata
        try:
            tracing.fw_metadata=SimpleNamespace(input_info=input_info)
            callback=value['compiler'](graph,examples)
        finally:
            tracing.fw_metadata=previous
        if getattr(callback,'_boxed_call',False) is not True:
            raise ValueError('Reloaded native compiler lost boxed input convention')
        self._graph=graph;self._input_info=input_info;self._call=callback
        value['events'].append(dict(stage='original_native_compiler_on_AOT_hit_rebind',
            graph=self._identity,flattened_input_count=len(examples),dynamo_input_count=len(example_inputs),
            current_model_rebound=True,completed_ns=time.perf_counter_ns()))

    def set_triton_bundle(self,triton_bundle):
        if triton_bundle is not None:
            raise ValueError('Native Hexagon backend has no Triton bundle')


@contextmanager
def native_cache_scope(manifest,cache_root,events):
    """Process-local opt-in; retain actual guard evaluation and normal miss path."""
    import torch._dynamo.backends.common as common
    import torch._functorch.config as config
    namespace=namespace_digest(manifest);cache_root=Path(cache_root)
    if not cache_root.is_absolute() or cache_root.is_symlink():
        raise ValueError('Explicit absolute native AOT cache root required')
    cache_root.mkdir(parents=True,exist_ok=True)
    scoped=cache_root/namespace;scoped.mkdir(exist_ok=True)
    if scoped.is_symlink():raise ValueError('Regular namespaced cache directory required')
    identity_file=scoped/'native-backend-identity.json'
    payload=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()+b'\n'
    try:
        with identity_file.open('xb') as stream:stream.write(payload)
    except FileExistsError:
        if identity_file.is_symlink() or identity_file.read_bytes()!=payload:
            raise ValueError('Native backend cache namespace identity changed')
    if torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards or config.bypass_autograd_cache_key:
        raise ValueError('Actual framework dynamic guards and deterministic keys required')
    from hexagon_captured_linalg_cache import captured_identity
    original=common.aot_autograd;previous=os.environ.get('TORCHINDUCTOR_CACHE_DIR')
    def factory(**kwargs):
        compiler=kwargs.get('fw_compiler')
        if (not callable(compiler) or kwargs.get('inference_compiler') is not compiler
                or getattr(compiler,'__name__',None)!='compiler'):
            return original(**kwargs)
        code=getattr(compiler,'__code__',None)
        if code is None or not Path(code.co_filename).is_file():
            return original(**kwargs)
        if hashlib.sha256(Path(code.co_filename).read_bytes()).hexdigest()!=manifest['native_source_sha256']:
            # Preserve lazy framework backends created through the same factory.
            return original(**kwargs)
        changed=dict(kwargs)
        wrapped=SerializableAOTDispatchCompiler(NativeAOTOutput,NativeAOTOutput)
        changed.update(fw_compiler=wrapped,inference_compiler=wrapped)
        backend=original(**changed)
        def invoke(graph,inputs):
            with compiler_scope(namespace,compiler,captured_identity,events):
                return backend(graph,inputs)
        return invoke
    try:
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str(scoped)
        common.aot_autograd=factory
        with config.patch(bundled_autograd_cache=True,enable_autograd_cache=True):
            yield dict(namespace=namespace,cache_directory=str(scoped))
    finally:
        common.aot_autograd=original
        if previous is None:os.environ.pop('TORCHINDUCTOR_CACHE_DIR',None)
        else:os.environ['TORCHINDUCTOR_CACHE_DIR']=previous
