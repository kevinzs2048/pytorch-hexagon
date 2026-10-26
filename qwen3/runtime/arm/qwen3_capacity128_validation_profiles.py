"""Bounded test corpora, not generation/cache logic or a runtime dispatcher.

The existing smoke corpus keeps its admission unchanged. Boundary cases describe
the separately CPU-checked native 127+2 / 128+1 experiments; listing them here
does not enable or accept them on DSP. All sampling/cache behavior stays native.
"""
from qwen3_long_decode_corpus import LONG_DECODE, LONG_DECODE_CASES
from dataclasses import dataclass
from types import MappingProxyType

from qwen3_capacity128_generate_common import CASES, validate_case as validate_smoke_case

SMOKE = 'native-capacity128-short-long67-repeat-v1'
BOUNDARY = 'native-capacity128-boundary127-128-repeat-v1'


@dataclass(frozen=True)
class Case:
    text: str
    prompt_tokens: int
    max_new_tokens: int
    min_new_tokens: int | None = None
    required_last_position: int | None = None

    def extra_generation_kwargs(self):
        return {} if self.min_new_tokens is None else {'min_new_tokens': self.min_new_tokens}


@dataclass(frozen=True)
class Profile:
    name: str
    cases: tuple[Case, ...]

    @property
    def forward_count_bounds(self):
        # Native generate does not feed its final generated token back through forward.
        return (sum(c.prompt_tokens + max(1, c.min_new_tokens or 0) - 1 for c in self.cases),
                sum(c.prompt_tokens + c.max_new_tokens - 1 for c in self.cases))

    def validate_case(self, index, ids):
        if type(index) is not int or not 0 <= index < len(self.cases):
            raise ValueError('Explicit test case index required')
        if not isinstance(ids, (list, tuple)):
            raise ValueError('Explicit unpadded token sequence required')
        if self.name == SMOKE:
            return validate_smoke_case(index, ids)
        if self.name not in (BOUNDARY, LONG_DECODE):
            raise ValueError('Unregistered validation profile')
        case = self.cases[index]
        if (len(ids) != case.prompt_tokens
                or any(type(t) is not int or not 0 <= t < 151936 for t in ids)
                or len(ids) + case.max_new_tokens - 1 > 128):
            raise ValueError('Pinned boundary request must fit 128 actual forward positions')


_long_suffix = 'Wait. ' * 49 + 'What is the code? Answer with only the code.'
PROFILES = MappingProxyType({
    LONG_DECODE: Profile(LONG_DECODE, tuple(Case(*row) for row in LONG_DECODE_CASES)),
    SMOKE: Profile(SMOKE, tuple(Case(text, length, new) for text, new, length in CASES)),
    BOUNDARY: Profile(BOUNDARY, (
        Case('Say hello.', 15, 2, 2),
        Case('Remember this code: BLUE. ' + _long_suffix, 127, 2, 2, 127),
        Case('Please remember this code: BLUE. ' + _long_suffix, 128, 1, 1, 127),
        Case('Say hello.', 15, 2, 2),
    )),
})


def get_profile(name):
    if type(name) is not str or name not in PROFILES:
        raise ValueError('An explicit supported validation profile is required')
    return PROFILES[name]
