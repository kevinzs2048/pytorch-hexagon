"""Short framework-integration cases, not a language quality benchmark."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    text: str
    prompt_tokens: int
    max_new_tokens: int
    min_new_tokens: int | None = None
    required_last_position: int | None = None

    def extra_generation_kwargs(self):
        return {} if self.min_new_tokens is None else {'min_new_tokens': self.min_new_tokens}

NAME='native-prefill4-short-all-tails-repeat-v1'
CASES=(Case('Say hello.',15,16),Case('Hello.',14,16),
       Case('Say hello briefly.',16,16),Case('Say hello.',15,16))


def validate_ids(index,ids):
    if type(index) is not int or not 0<=index<len(CASES):raise ValueError('Explicit corpus request required')
    case=CASES[index]
    if (not isinstance(ids,list) or len(ids)!=case.prompt_tokens
            or any(type(t) is not int or not 0<=t<151936 for t in ids)):
        raise ValueError('Pinned native tokenizer inputs differ')


def validate_coverage(requests,eos_token_ids,*,sequence_key):
    if sequence_key not in ('CPU_sequences','DSP_sequences') or len(requests)!=len(CASES):
        raise ValueError('Complete original short requests required')
    for i,(row,case) in enumerate(zip(requests,CASES)):
        validate_ids(i,row['input_ids'])
        if (row['text']!=case.text or row['prompt_tokens']!=case.prompt_tokens
                or row['max_new_tokens']!=case.max_new_tokens or row.get('min_new_tokens') is not None):
            raise ValueError('Native generation options differ')
        sequences=row[sequence_key]
        if len(sequences)!=1 or sequences[0][:case.prompt_tokens]!=row['input_ids']:
            raise ValueError('Actual sequence prefix differs')
        generated=sequences[0][case.prompt_tokens:]
        if (not 1<=len(generated)<=case.max_new_tokens or row['new_tokens']!=len(generated)
                or any(t in eos_token_ids for t in generated[:-1])
                or len(generated)<case.max_new_tokens and generated[-1] not in eos_token_ids):
            raise ValueError('Native EOS/length stopping differs')
        if row['call_end']-row['call_begin']!=(case.prompt_tokens+3)//4+len(generated)-1:
            raise ValueError('Four-token prefill/single-token decode call count differs')
    if requests[0][sequence_key]!=requests[3][sequence_key]:raise ValueError('Repeated request changed')
    return dict(validation_profile=NAME,prefill_chunk_size=4,prompt_tokens=[r['prompt_tokens'] for r in requests],
        new_tokens=[r['new_tokens'] for r in requests],all_native_stopping=True,
        natural_EOS_requests=[i for i,r in enumerate(requests) if r[sequence_key][0][-1] in eos_token_ids],
        covers_full_chunk_and_tail2_tail3=True,repeated_request_equal=True,answer_quality_assessed=False)
