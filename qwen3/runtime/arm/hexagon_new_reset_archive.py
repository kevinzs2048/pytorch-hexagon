"""Compress only a NEW invocation's verified full-sync communication record.

No model/DSP bytes change. Old archives are never selected by discovery. The
caller must give the exact just-produced file and original packet digest.
"""
import gzip
import hashlib
from pathlib import Path
import shutil

from hexagon_generate_reuse import sha

LIMIT = 32*1024**2
RESERVE = 24*1024**3


class LimitedWriter:
    def __init__(self, stream, maximum):
        self.stream, self.maximum, self.count = stream, maximum, 0

    def write(self, data):
        if self.count + len(data) > self.maximum:
            raise ValueError('New compressed reset record exceeds its explicit budget')
        count = self.stream.write(data)
        self.count += count
        return count


def archive_new_reset(path, *, expected_sha256, expected_bytes, maximum=LIMIT):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or type(expected_bytes) is not int
            or expected_bytes <= 0 or path.stat().st_size != expected_bytes
            or type(maximum) is not int or not 0 < maximum <= LIMIT
            or sha(path) != expected_sha256):
        raise ValueError('Exact NEW regular communication record required')
    if shutil.disk_usage(path.parent).free < RESERVE + maximum + 64*1024**2:
        raise ValueError('Compressed reset record would encroach on24GiB reserve')
    target = path.with_suffix(path.suffix + '.gz')
    # Existing compressed evidence is never overwritten, even after a failure.
    with target.open('xb') as stream:
        bounded = LimitedWriter(stream, maximum)
        with gzip.GzipFile(filename='', mode='wb', fileobj=bounded, mtime=0, compresslevel=1) as archive:
            with path.open('rb') as original:
                while block := original.read(64*1024):
                    archive.write(block)
    digest = hashlib.sha256()
    count = 0
    with gzip.open(target, 'rb') as stream:
        while block := stream.read(4*1024**2):
            count += len(block)
            if count > expected_bytes:
                raise ValueError('Compressed record expands beyond original size')
            digest.update(block)
    if count != expected_bytes or digest.hexdigest() != expected_sha256 or sha(path) != expected_sha256:
        raise ValueError('New record failed complete lossless readback; original retained')
    row = dict(codec='gzip-new-reset-input-v1', member=target.name,
        bytes=expected_bytes, sha256=expected_sha256, stored_bytes=target.stat().st_size,
        stored_sha256=sha(target), maximum_stored_bytes=maximum, lossless_readback=True)
    # Only this explicitly provided NEW transfer copy, after verified readback.
    path.unlink()
    return row
