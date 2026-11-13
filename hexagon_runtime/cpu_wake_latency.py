# SPDX-License-Identifier: BSD-3-Clause
"""Best-effort hold on the ARM CPU wake-up latency while FastRPC calls run.

Every FastRPC return wakes an ARM core. When the cores are in a deep idle
state the wake-up itself takes as long as the state exit latency (1.45 to
1.76 ms on QCS9075), which is added to every call. Linux keeps a PM QoS
request for as long as the descriptor of the latency device stays open.
The device is usually writable by root only, so the hold may not take
effect; the caller records the outcome.
"""
import os
import struct

DEVICE = "/dev/cpu_dma_latency"


class CpuWakeLatencyHold:
    def __init__(self, microseconds=0, device=DEVICE):
        if type(microseconds) is not int or not 0 <= microseconds <= 1_000_000:
            raise ValueError("Wake-up latency must be 0..1000000 microseconds")
        self.microseconds = microseconds
        self.reason = None
        self._fd = None
        fd = None
        try:
            fd = os.open(device, os.O_WRONLY | os.O_CLOEXEC)
            if os.write(fd, struct.pack("=i", microseconds)) != 4:
                raise OSError("short write")
            self._fd, fd = fd, None
        except OSError as error:
            self.reason = error.strerror or str(error)
        finally:
            if fd is not None:
                os.close(fd)

    @property
    def active(self):
        return self._fd is not None

    def record(self):
        return dict(active=self.active, microseconds=self.microseconds, reason=self.reason)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
