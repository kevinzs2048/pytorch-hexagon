"""Opt-in exact integer contraction policy for the existing query2..32 graph."""
from pathlib import Path
import struct
import subprocess

from hexagon_compiled_artifact_cache import digest_file

CONTRACT = 'original_Qwen_query2_to32_integer_ring_matmul_shared_rows_v1'
SYMBOL = '_mlir_ciface___qwen_i8_multi_matmul'


def validate_runtime(runtime, nm):
    for row in (runtime, nm):
        if (type(row) is not dict or set(row) != {'path', 'sha256'}
                or not Path(row['path']).is_absolute() or digest_file(row['path']) != row['sha256']):
            raise ValueError('Pinned multi-row integer runtime and inspector required')
    with Path(runtime['path']).open('rb') as stream:
        data = stream.read(20)
    if (len(data) != 20 or data[:6] != b'\x7fELF\x01\x01'
            or struct.unpack_from('<HH', data, 16) != (1, 164)):
        raise ValueError('ELF32 little-endian Hexagon integer runtime required')
    symbols = subprocess.run([nm['path'], '--defined-only', runtime['path']],
        check=True, capture_output=True, text=True, timeout=30).stdout
    names = [line.split()[-1] for line in symbols.splitlines() if line.split()]
    if names.count(SYMBOL) != 1 or any('qwen_multi_probe' in name or 'qwen_multi_packed_probe' in name for name in names):
        raise ValueError('Production integer runtime symbol required without diagnostic counters')
