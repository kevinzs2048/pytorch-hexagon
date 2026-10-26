"""Capacity128 test configuration; all cache and generation logic stays native."""
CASES=(('Say hello.',4,15),
    ('Remember this code: BLUE. '+'Wait. '*19+'What is the code? Answer with only the code.',4,67),
    ('Say hello.',4,15))


def native_cache(model):
    import torch
    from transformers.cache_utils import StaticCache
    cache=StaticCache(config=model.config,max_cache_len=128)
    cache.early_initialization(1,8,128,torch.float32,torch.device('cpu'))
    return cache


def validate_request(ids,max_new_tokens):
    if (not ids or type(max_new_tokens) is not int or max_new_tokens<=0
            or len(ids)+max_new_tokens>128
            or any(type(t) is not int or not 0<=t<151936 for t in ids)):
        raise ValueError('Batch1 unpadded request must fit native capacity128')


def validate_case(index,ids):
    if type(index) is not int or not 0<=index<len(CASES):raise ValueError('Explicit capacity128 test case required')
    validate_request(ids,CASES[index][1])
    if len(ids)!=CASES[index][2]:raise ValueError('Pinned tokenizer produced a different prompt length')
