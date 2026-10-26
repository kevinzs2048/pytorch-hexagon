"""Read-only ARM environment/model setup for native graph capture, no runtime.

The checkpoint, source closure, quantization and native cache are the same as
the accepted ARM driver. No DSP session or substitute forward is introduced.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def admit_arm_environment(root,digest,runtime_check):
    sources=Path(runtime_check)/'sources'
    sys.path[:0]=[str(Path(root)/'runtime'),str(sources),str(sources/'runtime')]
    from hexagon_arm_bundle import sha,member
    from hexagon_arm_prefill_bundle import ArmPrefillBundle
    bundle=ArmPrefillBundle(root,digest)
    pm=bundle.read('provenance/python-manifest.json')
    preflight=json.loads((Path(runtime_check)/'result.json').read_text())
    if (preflight['status']!='ARM_native_INT8_graph_capture_passed_not_model_or_DSP'
            or preflight['source_manifest_sha256']!=bundle.files['provenance/python-manifest.json']):
        raise ValueError('Original pinned ARM framework preflight required')
    for row in pm['files']:
        if sha(member(sources,row['path']))!=row['sha256']:raise ValueError('Original ARM framework source drift')
    return bundle,pm


def load_arm_model(bundle,pm):
    import torch
    from transformers import Qwen3ForCausalLM
    from torchao.quantization import quantize_,Int8DynamicActivationInt8WeightConfig
    from hexagon_dynamo_parameters import digest
    context=bundle.read('context.json')
    if (any(importlib.metadata.version(name)!=context['versions'][name] for name in ('torch','torchao','transformers'))
            or torch.version.git_version!=pm['torch_git']):
        raise ValueError('Pinned native ARM framework build differs')
    model=Qwen3ForCausalLM.from_pretrained(str(bundle.base.root/'hf'),local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
    expected=dict(context['model_config']);actual=json.loads(json.dumps(model.config.to_dict()))
    expected.pop('_name_or_path',None);actual.pop('_name_or_path',None)
    if actual!=expected:raise ValueError('Native model numerical configuration changed')
    h=hashlib.sha256()
    for name,value in model.named_parameters():h.update((name+':'+digest(value)+'\n').encode())
    if h.hexdigest()!=context['historical_parameter_sha256']:raise ValueError('Native checkpoint parameter bytes changed')
    quantize_(model,Int8DynamicActivationInt8WeightConfig(set_inductor_config=False))
    return model,context
