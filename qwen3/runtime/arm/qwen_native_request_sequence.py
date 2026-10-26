"""Validate sequential native generate requests; never implement a token loop.

PyTorch owns generation, stopping and cache.reset(). These checks only bound
the experiment and verify positions emitted by native framework calls.
"""
import json


def additional_prompts(encoded, *, generic, max_new_tokens, capacity=1024, vocab=151936):
    if encoded is None:
        return []
    if not generic:
        raise ValueError('Sequential requests require the explicit generic prompt entry')
    prompts = json.loads(encoded)
    if (type(prompts) is not list or not 1 <= len(prompts) <= 8
            or any(type(ids) is not list or not ids
                   or len(ids)+max_new_tokens > capacity
                   or any(type(token) is not int or not 0 <= token < vocab for token in ids)
                   for ids in prompts)):
        raise ValueError('Bounded nonempty sequential prompts required')
    return prompts


def verify_request(forwards, prompt_ids, generated_ids):
    if not forwards or not generated_ids:
        raise ValueError('Native request did not produce model forwards and generated tokens')
    expected_tokens = prompt_ids+generated_ids[:-1]
    actual_tokens = [token for row in forwards for token in row['input_token_ids']]
    actual_positions = [position for row in forwards for position in row['cache_positions']]
    if (any(len(row['cache_positions']) != row['query']
            or len(row['input_token_ids']) != row['query'] for row in forwards)
            or actual_tokens != expected_tokens
            or actual_positions != list(range(len(expected_tokens)))):
        raise ValueError('Native request tokens or reset positions differ')
