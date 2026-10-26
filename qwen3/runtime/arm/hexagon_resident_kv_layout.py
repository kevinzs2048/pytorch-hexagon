"""Explicit capacity64/128/256 packet geometry; not device/model admission.

The deployed capacity64 wire is unchanged. This helper prepares the capacity128
layout and tests it against the compiler's canonical 128-byte-aligned buffer ABI.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResidentKVLayout:
    capacity: int = 64

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity not in (64,128,256):
            raise ValueError('Explicit capacity64, capacity128 or capacity256 geometry required')

    @property
    def kv_bytes(self):return 8*self.capacity*128*4

    @property
    def mask_bytes(self):return self.capacity*4

    @property
    def mask_offset(self):return 384+2*self.kv_bytes

    @property
    def model_input_bytes(self):return 384+56*self.kv_bytes+self.mask_bytes

    @property
    def model_output_bytes(self):return 56*self.kv_bytes+151936*4

    @property
    def full_packet_bytes(self):return 128+self.model_input_bytes

    @property
    def reuse_packet_bytes(self):return 128+3*8+self.mask_bytes

    @property
    def row_bytes(self):return 56*8*128*4

    @property
    def response_bytes(self):return 128+self.row_bytes+151936*4

    def input_kv_offset(self,index):
        if type(index) is not int or not 0<=index<56:raise ValueError('KV tensor index outside 56 tensors')
        return 384+index*self.kv_bytes+(self.mask_bytes if index>=2 else 0)

    def signature(self):
        def kv():return dict(dtype='float32',shape=[1,8,self.capacity,128])
        inputs=[dict(dtype='int64',shape=s) for s in ([1,1],[1],[1,1])]
        inputs += [kv(),kv(),dict(dtype='float32',shape=[1,1,1,self.capacity])]+[kv() for _ in range(54)]
        return inputs,[kv() for _ in range(56)]+[dict(dtype='float32',shape=[1,1,151936])]

    def validate_position(self,position):
        if type(position) is not int or not 0<=position<self.capacity:
            raise ValueError('Integer position outside selected cache capacity')

    def validate_buffers(self,values):
        if (type(values) not in (list,tuple) or len(values)!=56
                or any(type(v) is not bytes or len(v)!=self.kv_bytes for v in values)):
            raise ValueError('Exactly 56 immutable complete KV buffers for selected capacity required')

    def unpack_input_kv(self,raw):
        if type(raw) is not bytes or len(raw)!=self.model_input_bytes:
            raise ValueError('Complete canonical input for selected capacity required')
        return [raw[self.input_kv_offset(i):self.input_kv_offset(i)+self.kv_bytes] for i in range(56)]

    def apply_rows(self,before,payload,position):
        self.validate_position(position);self.validate_buffers(before)
        if type(payload) is not bytes or len(payload)!=self.row_bytes:
            raise ValueError('Complete token-row payload required')
        result=[]
        for tensor,old in enumerate(before):
            value=bytearray(old)
            for head in range(8):
                start=(head*self.capacity+position)*512
                packet=(tensor*8+head)*512
                value[start:start+512]=payload[packet:packet+512]
            result.append(bytes(value))
        return result
