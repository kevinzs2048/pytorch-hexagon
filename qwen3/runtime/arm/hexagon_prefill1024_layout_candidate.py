"""Prospective 1/2/3/4/8-token buffer geometry, NOT runtime admission.

PyTorch retains cache/generation semantics. This only describes packed tensors
for a future thin DSP call boundary. Existing deployed wire version4, native
bundle admission and accepted one/two-token layouts are deliberately unchanged.
New full-model libraries and a separately validated transport are mandatory.
"""
from dataclasses import dataclass

from hexagon_pair256_kv_layout import Pair256KVLayout


@dataclass(frozen=True)
class Prefill1024LayoutCandidate(Pair256KVLayout):
    capacity: int = 1024

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity != 1024:
            raise ValueError('Only prospective capacity1024 geometry')
        if type(self.query_tokens) is not int or self.query_tokens not in (1, 2, 3, 4, 8):
            raise ValueError('Only explicitly planned 1/2/3/4/8-token geometry')

    def prospective_resident_allocation(self):
        """Malloc request sum only, excluding model/loader/RPC/intermediates."""
        maximum = Prefill1024LayoutCandidate(query_tokens=8)
        return maximum.model_input_bytes + 127 + maximum.model_output_bytes + 127
