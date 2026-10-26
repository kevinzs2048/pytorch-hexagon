"""Exact per-forward parameter checks against private immutable byte snapshots.

Initial snapshots retain the existing compiler SHA identities. Later checks
compare every logical byte, including .data mutations and NaN/signed-zero bits.
No tensor-version shortcut, sampling, mutable reference, or numerical operation.
"""
import copy
import ctypes
import hashlib


def raw_view(value):
    import torch
    # Matches the existing digest's logical contiguous order; reshape also
    # covers rank-zero module scalars without a floating-point conversion.
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()


class ExactParameterBytePool:
    """Share immutable snapshots across the two actual captured graphs."""

    def __init__(self):
        self._bytes={}
        self._libc=ctypes.CDLL(None)
        self._memcmp=self._libc.memcmp
        self._memcmp.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t]
        self._memcmp.restype=ctypes.c_int

    @property
    def snapshot_bytes(self):
        return sum(len(value) for value in self._bytes.values())

    def bind(self,binding):
        return ExactParameterByteGuard(self,binding)

    def matches(self,key,raw):
        if raw.nbytes!=key[1]:
            raise ValueError('Exact comparison length differs from pinned parameter')
        expected=self._bytes.get(key)
        if expected is None:
            # Hash immutable copied bytes, never a mutable reference. This
            # preserves the original SHA pins even in metadata-only binding.
            expected=raw.tobytes()
            if hashlib.sha256(expected).hexdigest()!=key[0]:return False
            self._bytes[key]=expected
        if len(expected)!=raw.nbytes:
            raise ValueError('Immutable parameter snapshot length changed')
        # c_char_p passes the bytes buffer without a copy; memcmp's explicit
        # length includes embedded NULs. raw owns the live tensor view.
        return self._memcmp(raw.ctypes.data,expected,raw.nbytes)==0


class ExactParameterByteGuard:
    def __init__(self,pool,binding):
        self._pool=pool
        self._records=copy.deepcopy(binding.records)
        self._input_count=binding.input_count
        self._retained=tuple(binding.retained)
        indexes=[row['input_index'] for row in self._records]
        if (type(self._input_count) is not int or len(set(indexes))!=len(indexes)
                or any(type(i) is not int or not 0<=i<self._input_count for i in indexes)
                or list(self._retained)!=[i for i in range(self._input_count) if i not in indexes]
                or any(type(r['bytes']) is not int or r['bytes']<0
                    or len(r['sha256'])!=64 for r in self._records)):
            raise ValueError('Exact parameter guard requires the complete original binding')

    def check(self,inputs):
        from hexagon_dynamo_parameters import spec
        if len(inputs)!=self._input_count:
            raise ValueError('Runtime parameter input arity differs')
        for row in self._records:
            value=inputs[row['input_index']]
            if spec(value)!=row['spec']:
                raise ValueError('Compiled parameter metadata changed; create a fresh compile session')
            raw=raw_view(value)
            if raw.nbytes!=row['bytes'] or not self._pool.matches((row['sha256'],row['bytes']),raw):
                raise ValueError('Compiled parameter changed; create a fresh compile session before DSP execution')
        return [inputs[index] for index in self._retained]
