"""Explicit, query1-only original byte-copy contract for an owned DSP object."""
from pathlib import Path
import struct

from hexagon_compiled_artifact_cache import digest_file

CONTRACT = 'original_Qwen_query1_memrefCopy_exact_integer_product_guards_v1'
DYNAMIC_CONTRACT = 'original_Qwen_query2_to32_memrefCopy_exact_integer_product_guards_v1'
SOURCE_SHA256 = '2e148a5a5d35479944792ce3f4144f2b31f9ca188cfffe70151ab1d57ddef890'


def validate_runtime(runtime, source):
    for row in (runtime, source):
        if (type(row) is not dict or set(row) != {'path', 'sha256'}
                or not Path(row['path']).is_absolute() or digest_file(row['path']) != row['sha256']):
            raise ValueError('Pinned guarded byte-copy object and source required')
    if source['sha256'] != SOURCE_SHA256:
        raise ValueError('Original byte-copy traversal and exact integer guard source required')
    with Path(runtime['path']).open('rb') as stream:
        data = stream.read(20)
    if (len(data) != 20 or data[:6] != b'\x7fELF\x01\x01'
            or struct.unpack_from('<HH', data, 16) != (1, 164)):
        raise ValueError('Owned ELF32 little-endian Hexagon relocatable runtime required')


def add_link_runtime(argv, runtime_path):
    command = list(argv)
    if (command.count('-o') != 1 or command[-2] != '-o'
            or '-Wl,--wrap=memrefCopy' in command or str(runtime_path) in command):
        raise ValueError('One original terminal link output without a copy wrapper required')
    command[len(command)-2:len(command)-2] = [str(runtime_path), '-Wl,--wrap=memrefCopy']
    return command
