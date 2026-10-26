"""Pinned native model/configuration helpers, not a generation implementation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from hexagon_generate_reuse import sha


def tensor_bytes(tensor):
    import torch
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def load_model(context_path):
    import torch
    from transformers import Qwen3ForCausalLM, AutoTokenizer
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    context = json.loads(Path(context_path).read_text())
    if {k: importlib.metadata.version(k) for k in context['versions']} != context['versions']:
        raise ValueError('Pinned framework versions differ')
    checkpoint = context['checkpoint']
    if sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Checkpoint bytes differ')
    torch.set_num_threads(8)
    directory = str(Path(checkpoint['path']).parent)
    model = Qwen3ForCausalLM.from_pretrained(directory, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').eval().requires_grad_(False)
    if json.loads(json.dumps(model.config.to_dict())) != context['model_config']:
        raise ValueError('Native model configuration differs')
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update((name + ':' + hashlib.sha256(tensor_bytes(value)).hexdigest() + '\n').encode())
    if digest.hexdigest() != context['historical_parameter_sha256']:
        raise ValueError('Original named model weights differ')
    quantize_(model, Int8DynamicActivationInt8WeightConfig(set_inductor_config=False))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return model, tokenizer, context


def native_cache(model):
    import torch
    from transformers.cache_utils import StaticCache
    cache = StaticCache(config=model.config, max_cache_len=64)
    cache.early_initialization(1, 8, 128, torch.float32, torch.device('cpu'))
    return cache


def validate_request(token_ids, max_new_tokens):
    if (type(max_new_tokens) is not int or max_new_tokens <= 0 or not token_ids
            or any(type(t) is not int or not 0 <= t < 151936 for t in token_ids)
            or len(token_ids) + max_new_tokens > 64):
        raise ValueError('Batch-one unpadded request must fit the 64-token native cache')


def generation_options(backend=None, *, max_new_tokens=8):
    from transformers import CompileConfig
    result = dict(do_sample=False, max_new_tokens=max_new_tokens, prefill_chunk_size=1,
                  return_dict_in_generate=True, output_logits=True)
    if backend is None:
        result['disable_compile'] = True
    else:
        cfg = CompileConfig(backend=backend, fullgraph=True, dynamic=False, mode='default')
        # Explicit version-pinned Transformers switch: host tensor locations do
        # not identify the device used by our custom compiled callable.
        cfg._compile_all_devices = True
        result['compile_config'] = cfg
    return result
