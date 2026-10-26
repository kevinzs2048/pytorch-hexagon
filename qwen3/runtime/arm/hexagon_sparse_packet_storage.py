"""Release zero-filled allocation in a closed owned packet, preserving bytes.

This changes disk allocation only, not the model, cache contents, RPC sizes or
network traffic. It must never be used on files another writer may still use.
Both digests are checked through the same locked, non-symlink file descriptor.
"""
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


def probe_sparse_packet_storage(root):
    """Fail before loading a model if the selected filesystem cannot dig holes.

    In particular, an SSHFS path is not equivalent to a local NVMe path for
    fallocate. Only this newly created disposable probe is removed.
    """
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError('Explicit existing packet storage root required')
    payload = b'packet-storage-probe' + bytes(8192)
    with tempfile.NamedTemporaryFile(prefix='.sparse-probe-', dir=root) as stream:
        stream.write(payload)
        stream.flush()
        try:
            return sparsify_verified_packet(Path(stream.name), hashlib.sha256(payload).hexdigest())
        except subprocess.CalledProcessError as exc:
            raise ValueError('Selected evidence filesystem does not support sparse packet allocation; '
                             'omit --sparse-packets and reserve space for complete packets') from exc


def sparsify_verified_packet(path, expected_sha256):
    path = Path(path)
    if not path.is_absolute() or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('Explicit owned packet path and SHA256 required')
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'r+b', buffering=0) as stream:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= 256*1024**2):
            raise ValueError('One bounded, unaliased regular packet file required')
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_sha256:
            raise ValueError('Packet bytes differ before allocation change')
        subprocess.run(['fallocate', '--dig-holes', f'/proc/self/fd/{descriptor}'],
                       pass_fds=(descriptor,), check=True, capture_output=True, timeout=30)
        stream.seek(0)
        after = os.fstat(descriptor)
        if (hashlib.file_digest(stream, 'sha256').hexdigest() != expected_sha256
                or before.st_size != after.st_size
                or before.st_ino != after.st_ino or before.st_dev != after.st_dev):
            raise ValueError('Packet bytes or identity changed during sparse allocation')
        return dict(logical_bytes=after.st_size, sha256=expected_sha256,
                    allocated_bytes_before=before.st_blocks*512,
                    allocated_bytes_after=after.st_blocks*512,
                    full_bytes_reverified=True)
