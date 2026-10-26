"""Prepare capacity1024 through native Transformers APIs, without cache policy.

Separate opt-in profile; the accepted capacity256 helper stays unchanged.
The pinned framework's generate() remains responsible for reset and updates.
"""
import torch


def prepare_native1024_cache(model):
    if model.device.type != 'cpu' or model.dtype != torch.float32:
        raise ValueError('Current experiment requires native CPU-visible FP32 cache')
    old = getattr(model, '_cache', None)
    if old is not None and old.max_cache_len > 1024:
        raise ValueError('Existing larger cache does not match compiled capacity1024')
    config = model.config.get_text_config(decoder=True)
    cache = model._get_cache(cache_implementation='static', batch_size=1,
                            max_cache_len=1024, model_kwargs={})
    cache.early_initialization(1, config.num_key_value_heads, config.head_dim,
                               torch.float32, model.device)
    if cache.max_cache_len != 1024:
        raise ValueError('Native cache differs from compiled capacity1024')
    return cache

