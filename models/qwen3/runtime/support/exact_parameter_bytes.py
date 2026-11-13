"""Parameter checks against private immutable byte snapshots.

Initial snapshots retain the existing compiler SHA identities. Later checks
compare every logical byte, including .data mutations and NaN/signed-zero bits.
Complete comparisons are the default. An explicit exclusive immutable-model
scope may reuse checks within one request; versions alone never prove ownership.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import ctypes
import hashlib


def raw_view(value):
    import torch
    # Matches the existing digest's logical contiguous order; reshape also
    # covers rank-zero module scalars without a floating-point conversion.
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()


class ExactParameterBytePool:
    """Share immutable snapshots across captured inference graphs."""

    def __init__(self, *, workers=1):
        if type(workers) is not int or workers not in (1, 2, 4):
            raise ValueError('Parameter comparison workers must be 1, 2 or 4')
        self.workers = workers
        self._executor = None
        self._closed = False
        self._bytes={}
        self._owned_request = None
        self._libc=ctypes.CDLL(None)
        self._memcmp=self._libc.memcmp
        self._memcmp.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self._memcmp.restype=ctypes.c_int

    @property
    def snapshot_bytes(self):
        return sum(len(value) for value in self._bytes.values())

    def bind(self,binding):
        return ExactParameterByteGuard(self,binding)

    @contextmanager
    def owned_request(self):
        """Reuse checks only while the caller exclusively owns an immutable model.

        The caller must prevent all parameter writes and external aliases during
        the scope, including writes through .data or NumPy. Version counters are
        only an additional check, not a substitute for that ownership contract.
        Every new scope rechecks complete bytes before admitting each parameter.
        Ordinary callers that cannot guarantee ownership must not enter this scope.
        """
        if self._closed:
            raise RuntimeError('Parameter comparison pool is closed')
        if self._owned_request is not None:
            raise RuntimeError('Owned parameter requests cannot overlap')
        self._owned_request = {}
        try:
            yield
        finally:
            self._owned_request = None

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._closed = True

    def matches(self, key, raw):
        return self.matches_many([(key, raw)])

    def matches_many(self, values):
        if self._closed:
            raise RuntimeError('Parameter comparison pool is closed')
        spans = []
        for key, raw in values:
            if raw.nbytes != key[1]:
                raise ValueError('Exact comparison length differs from pinned parameter')
            expected = self._bytes.get(key)
            if expected is None:
                # Copy and hash before publishing the immutable reference. A
                # comparison below still detects changes during snapshot creation.
                expected = raw.tobytes()
                if hashlib.sha256(expected).hexdigest() != key[0]:
                    return False
                self._bytes[key] = expected
            if len(expected) != raw.nbytes:
                raise ValueError('Immutable parameter snapshot length changed')
            address = ctypes.cast(ctypes.c_char_p(expected), ctypes.c_void_p).value
            spans.append((raw.ctypes.data, address, raw.nbytes))

        def compare(group):
            return all(self._memcmp(a, b, n) == 0 for a, b, n in group)

        # Keep small guards serial. Split large tensors as well as tensor lists,
        # so one large embedding cannot leave the other workers idle. values and
        # _bytes retain both buffers until EVERY submitted comparison completes.
        if self.workers == 1 or sum(n for _, _, n in spans) < 4 * 1024 * 1024:
            return compare(spans)
        groups = [[] for _ in range(self.workers)]
        ordinal = 0
        for a, b, n in spans:
            for offset in range(0, n, 4 * 1024 * 1024):
                groups[ordinal % self.workers].append(
                    (a + offset, b + offset, min(4 * 1024 * 1024, n - offset)))
                ordinal += 1
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.workers,
                                                thread_name_prefix='qwen-parameter-check')
        results = list(self._executor.map(compare, groups))
        return all(results)


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
        if self._pool._closed:
            raise RuntimeError('Parameter comparison pool is closed')
        values = []
        admitted = []
        request = self._pool._owned_request
        for row in self._records:
            value=inputs[row['input_index']]
            if spec(value)!=row['spec']:
                raise ValueError('Compiled parameter metadata changed; create a fresh compile session')
            # Small dynamic module scalars still receive exact checks each time.
            # Retain each parameter tensor to prevent its storage address being reused
            # after destruction. Unversioned inference tensors stay on full checks.
            identity = None
            if (request is not None and row['spec']['shape']
                    and row.get('kind', 'parameter') == 'parameter'):
                try:
                    version = value._version
                except RuntimeError:
                    version = None
                if version is not None:
                    identity = (row['sha256'], row['bytes'], value.data_ptr(),
                                tuple(value.shape), tuple(value.stride()), version)
                    if identity in request:
                        continue
            raw=raw_view(value)
            if raw.nbytes != row['bytes']:
                raise ValueError('Compiled parameter byte length changed')
            values.append(((row['sha256'], row['bytes']), raw))
            if identity is not None:
                admitted.append((identity, value))
        if not self._pool.matches_many(values):
            raise ValueError('Compiled parameter changed; create a fresh compile session before DSP execution')
        if request is not None:
            request.update(admitted)
        return [inputs[index] for index in self._retained]
