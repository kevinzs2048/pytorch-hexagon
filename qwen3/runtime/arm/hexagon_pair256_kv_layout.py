"""Candidate one/two-token byte geometry at capacity256; not a wire protocol.

Matches the native pair capture's last-token-only logits. Reuses complete KV
buffer validation and aligned offsets; does not change StaticCache, instantiate
a device session, or enable an existing v1 runtime to accept two-token packets.
"""
from dataclasses import dataclass
from hexagon_resident_kv_layout import ResidentKVLayout


@dataclass(frozen=True)
class Pair256KVLayout(ResidentKVLayout):
    capacity: int = 256
    query_tokens: int = 2

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity != 256:
            raise ValueError('Only explicit capacity256 candidate supported')
        if type(self.query_tokens) is not int or self.query_tokens not in (1, 2):
            raise ValueError('Only one/two-token native ABI supported')

    @property
    def mask_bytes(self):
        return self.query_tokens*self.capacity*4

    @property
    def row_bytes(self):
        return 56*8*self.query_tokens*128*4

    @property
    def reuse_packet_bytes(self):
        # Prospective packet size only: a new versioned wire/runtime is needed.
        return 128+3*self.query_tokens*8+self.mask_bytes

    def signature(self):
        inputs,outputs=super().signature()
        n=self.query_tokens
        inputs[:3]=[dict(dtype='int64',shape=s) for s in ([1,n],[n],[1,n])]
        inputs[5]=dict(dtype='float32',shape=[1,1,n,self.capacity])
        return inputs,outputs

    def validate_position(self,position):
        if type(position) is not int or not 0 <= position <= self.capacity-self.query_tokens:
            raise ValueError('Contiguous token block exceeds native cache capacity')

    def apply_rows(self,before,payload,position):
        self.validate_position(position)
        self.validate_buffers(before)
        if type(payload) is not bytes or len(payload)!=self.row_bytes:
            raise ValueError('Complete tensor/head/token ordered row payload required')
        block_bytes=self.query_tokens*512
        result=[]
        for tensor,old in enumerate(before):
            value=bytearray(old)
            for head in range(8):
                start=(head*self.capacity+position)*512
                source=(tensor*8+head)*block_bytes
                value[start:start+block_bytes]=payload[source:source+block_bytes]
            result.append(bytes(value))
        return result
