"""Bounded ARM-local shared staging. No tensor/cache/generation policy.

FastRPC still transfers complete model buffers. These memfds only remove
Python/AP filesystem staging; they are NOT rpcmem or DSP-resident KV.
"""
import fcntl
from contextlib import contextmanager
import mmap
import os
import threading

from hexagon_runtime.hexagon_buffer_transport import HexagonLibraryBufferStream
from hexagon_runtime.hexagon_rpc_stream import PersistentRPCError


class SharedPacketBuffers:
    def __init__(self, input_bytes, output_bytes):
        if any(type(n) is not int or not 0 < n <= 256*1024**2 for n in (input_bytes, output_bytes)):
            raise ValueError('Bounded shared packet capacities required')
        self.capacities = (input_bytes, output_bytes)
        self.fds, self.maps = [], []
        self.closed = False
        try:
            for label, size in zip(('input', 'output'), self.capacities, strict=True):
                fd = os.memfd_create('hexagon-packet-'+label, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                self.fds.append(fd)
                os.ftruncate(fd, size)
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                            fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                self.maps.append(mmap.mmap(fd, size, access=mmap.ACCESS_WRITE))
        except BaseException:
            self.close()
            raise

    def write_input(self, payload):
        if self.closed or not isinstance(payload, bytes) or not 0 < len(payload) <= self.capacities[0]:
            raise ValueError('Nonempty bounded packet and open shared buffers required')
        self.maps[0][:len(payload)] = payload

    def read_output(self, size):
        if self.closed or type(size) is not int or not 0 < size <= self.capacities[1]:
            raise ValueError('Bounded output and open shared buffers required')
        return self.maps[1][:size]

    def close(self):
        if self.closed:
            return
        for mapping in reversed(self.maps):
            mapping.close()
        for fd in reversed(self.fds):
            os.close(fd)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


class HexagonSharedPacketStream(HexagonLibraryBufferStream):
    def __init__(self, argv, env, directory, *, input_bytes, output_bytes, initial_library, library_path_prefix, timeout=120):
        self._packet_lock = threading.Lock()
        self.packets = SharedPacketBuffers(input_bytes, output_bytes)
        try:
            super().__init__([*argv, *(str(fd) for fd in self.packets.fds)], env, directory,
                input_bytes=input_bytes, output_bytes=output_bytes, initial_library=initial_library,
                library_path_prefix=library_path_prefix, timeout=timeout, pass_fds=tuple(self.packets.fds))
        except BaseException:
            self.packets.close()
            raise

    @contextmanager
    def _exclusive(self):
        if not self._packet_lock.acquire(blocking=False):
            raise PersistentRPCError('Shared packet stream already has an active operation')
        try:
            yield
        finally:
            self._packet_lock.release()

    def start(self):
        with self._exclusive():
            event = super().start()
            if event.get('shared_memory_packets') is not True:
                raise PersistentRPCError('Runtime did not opt into shared-memory packets')
            return event

    def select_library(self, path):
        with self._exclusive():
            return super().select_library(path)

    def invoke_sized(self, *unused):
        raise ValueError('Shared staging requires explicit fresh packet bytes')

    def invoke_packet(self, payload, output_bytes):
        with self._exclusive():
            if (type(output_bytes) is not int or not 0 < output_bytes <= self.output_bytes
                    or not self.ready or self.closed):
                raise ValueError('Ready stream and bounded output required')
            self.packets.write_input(payload)
            event = super().invoke_sized(len(payload), output_bytes)
            # Never read old output when invocation/result/ordinal checks failed.
            return self.packets.read_output(output_bytes), event

    def close(self):
        with self._exclusive():
            try:
                super().close()
            finally:
                self.packets.close()
