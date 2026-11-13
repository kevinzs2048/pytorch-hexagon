# SPDX-License-Identifier: BSD-3-Clause
import os
import struct

import pytest

from hexagon_runtime.cpu_wake_latency import CpuWakeLatencyHold


def test_hold_writes_the_request_and_keeps_the_descriptor_open(tmp_path):
    device = tmp_path / "latency"
    device.write_bytes(b"")
    with CpuWakeLatencyHold(device=str(device)) as hold:
        assert hold.active and hold.record() == dict(active=True, microseconds=0, reason=None)
        assert device.read_bytes() == struct.pack("=i", 0)
        fd = hold._fd
        os.fstat(fd)
        assert os.get_inheritable(fd) is False
    assert not hold.active
    with pytest.raises(OSError):
        os.fstat(fd)
    hold.close()


def test_missing_or_unwritable_device_is_reported_not_raised(tmp_path):
    hold = CpuWakeLatencyHold(device=str(tmp_path / "absent"))
    assert not hold.active and hold.record()["reason"]
    hold.close()
    if os.geteuid() != 0:
        locked = tmp_path / "locked"
        locked.write_bytes(b"")
        locked.chmod(0o400)
        hold = CpuWakeLatencyHold(device=str(locked))
        assert not hold.active and hold.record()["reason"]


@pytest.mark.parametrize("value", [-1, 1_000_001, 1.5, True, None])
def test_latency_value_is_validated(tmp_path, value):
    with pytest.raises(ValueError):
        CpuWakeLatencyHold(microseconds=value, device=str(tmp_path / "x"))
