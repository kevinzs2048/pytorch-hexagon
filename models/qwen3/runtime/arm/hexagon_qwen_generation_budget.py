"""Native autoregressive cache demand; the last emitted token is not forwarded."""


def cache_demand(prompt_tokens, max_new_tokens):
    if (type(prompt_tokens) is not int or prompt_tokens < 1
            or type(max_new_tokens) is not int or max_new_tokens < 1):
        raise ValueError('Positive integer prompt and generation lengths required')
    return prompt_tokens+max_new_tokens-1


def validate_cache_demand(prompt_tokens, max_new_tokens, capacity=1024):
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Positive integer KV capacity required')
    demand = cache_demand(prompt_tokens, max_new_tokens)
    if demand > capacity:
        raise ValueError('Native worst-case forwarded tokens exceed KV capacity')
    return demand
