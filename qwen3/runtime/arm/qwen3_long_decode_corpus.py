"""Capacity128 long-answer corpus registered in the framework-cache producer.

These are test inputs and postconditions, not a generation implementation.
No min_new_tokens override: natural EOS and max_new_tokens stay native.
Coverage checks do NOT prove DSP execution or numerical correctness; the
complete-model/packet auditor must still verify every actual output.
"""

LONG_DECODE = 'native-capacity128-long-decode-repeat-v1'
_ENGLISH = 'Explain how rain forms. Give a clear explanation in several sentences.'
_CHINESE = '请用几句话解释雨是怎样形成的，包括蒸发、凝结和降雨三个过程。'
# Entries match Case(text, prompt_tokens, max_new_tokens); no forced minimum.
LONG_DECODE_CASES = (
    (_ENGLISH, 26, 64),
    (_CHINESE, 31, 64),
    ('Say hello.', 15, 16),
    (_ENGLISH, 26, 64),
)


def validate_long_decode_coverage(requests, eos_token_ids, *, sequence_key='DSP_sequences'):
    """Additional coverage requirements, after full independent model audit.

Do not silently accept another short-answer test as a long-decode regression.
Token identity against native CPU and runtime lifecycle are checked elsewhere.
"""
    if sequence_key not in ('DSP_sequences', 'CPU_sequences'):
        raise ValueError('Explicit CPU or DSP sequence field required')
    if not isinstance(requests, list) or len(requests) != 4:
        raise ValueError('Four explicit long-decode requests required')
    if (not isinstance(eos_token_ids, (tuple, list)) or not eos_token_ids
            or any(type(v) is not int or not 0 <= v < 151936 for v in eos_token_ids)):
        raise ValueError('Explicit native EOS token IDs required')
    generated = []
    for index, (request, case) in enumerate(zip(requests, LONG_DECODE_CASES)):
        text, prompt_tokens, limit = case
        ids = request.get('input_ids')
        sequence = request.get(sequence_key)
        if (request.get('text') != text
                or type(request.get('prompt_tokens')) is not int
                or request['prompt_tokens'] != prompt_tokens
                or type(request.get('max_new_tokens')) is not int
                or request['max_new_tokens'] != limit
                or request.get('min_new_tokens') is not None
                or not isinstance(ids, list) or len(ids) != prompt_tokens
                or not isinstance(sequence, list) or len(sequence) != 1
                or not isinstance(sequence[0], list) or sequence[0][:prompt_tokens] != ids
                or any(type(v) is not int or not 0 <= v < 151936 for v in sequence[0])):
            raise ValueError('Long-decode corpus, native stop options or sequence differs')
        tokens = sequence[0][prompt_tokens:]
        if (type(request.get('new_tokens')) is not int
                or request['new_tokens'] != len(tokens) or not 1 <= len(tokens) <= limit):
            raise ValueError('Actual generated token count differs')
        if index in (0, 1, 3) and len(tokens) < 32:
            raise ValueError('Long request must actually generate at least 32 tokens')
        if any(token in eos_token_ids for token in tokens[:-1]):
            raise ValueError('Generation continued past native EOS')
        if len(tokens) < limit and tokens[-1] not in eos_token_ids:
            raise ValueError('Early stop without native EOS')
        if index in (1, 2) and tokens[-1] not in eos_token_ids:
            raise ValueError('Chinese and short requests must exercise native EOS')
        generated.append(tokens)
    if requests[0][sequence_key] != requests[-1][sequence_key]:
        raise ValueError('Repeated long request differs after cache reset')
    return dict(long_decode_coverage=True, sequence_field=sequence_key,
                new_tokens=[len(tokens) for tokens in generated],
                natural_EOS_requests=[i for i, tokens in enumerate(generated)
                                      if tokens[-1] in eos_token_ids],
                repeated_long_request_identical=True,
                forced_min_new_tokens=False)
