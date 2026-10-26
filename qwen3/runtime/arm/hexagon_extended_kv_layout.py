"""Prospective 512/1024 native ABI geometry, NOT an admitted DSP runtime.

Reuse existing tensor ordering and last-token logits. This does not enlarge
the deployed capacity256 code, change its packet version, or implement cache
management. A separately compiled model and separately admitted C transport
are mandatory before these sizes may be executed.
"""
from dataclasses import dataclass

from hexagon_pair256_kv_layout import Pair256KVLayout


@dataclass(frozen=True)
class ExtendedKVLayout(Pair256KVLayout):
    capacity: int = 512

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity not in (512, 1024):
            raise ValueError('Only prospective capacity512/1024 ABI geometry')
        if type(self.query_tokens) is not int or self.query_tokens not in (1, 2):
            raise ValueError('Only one/two-token native ABI geometry')

    def prospective_resident_allocation(self):
        """Exact malloc request sum IF current two-buffer design is extended.

        Not DSP peak memory: excludes wrapper, model intermediates, libraries,
        RPC buffers and allocator overhead. Each allocation reserves 127 bytes
        so its data can be aligned to 128 bytes, as in the current C runtime.
        """
        maximum = ExtendedKVLayout(self.capacity, query_tokens=2)
        return maximum.model_input_bytes + 127 + maximum.model_output_bytes + 127
