"""Opt-in byte-exact memcpy wrapper for the original query2..32 model.

The wrapper interprets no floating-point values. Exactly 512 bytes use the
compiler's guaranteed inline copy; every other size calls the original SDK
memcpy. Descriptor traversal and all DSP32 overflow checks stay unchanged.
"""
from pathlib import Path
import struct
import subprocess

from hexagon_compiled_artifact_cache import digest_file

CONTRACT = 'original_Qwen_query2_to32_memcpy_exact512_inline_or_SDK_v1'
QUERY1_CONTRACT = 'original_Qwen_query1_memcpy_exact512_inline_or_SDK_v1'
SOURCE_SHA256 = 'd004012e21b6d85f42b1a2fb64b6290d2b76c26a5350091aa93252e3ff41587d'


def validate_runtime(runtime, source, nm):
    for row in (runtime, source, nm):
        if (type(row) is not dict or set(row) != {'path', 'sha256'}
                or not Path(row['path']).is_absolute() or digest_file(row['path']) != row['sha256']):
            raise ValueError('Pinned fixed512 byte-copy object, source and inspector required')
    if source['sha256'] != SOURCE_SHA256:
        raise ValueError('Original exact512 inline-or-SDK source required')
    with Path(runtime['path']).open('rb') as stream:
        header = stream.read(20)
    if (len(header) != 20 or header[:6] != b'\x7fELF\x01\x01'
            or struct.unpack_from('<HH', header, 16) != (1, 164)):
        raise ValueError('ELF32 little-endian Hexagon relocatable byte-copy runtime required')
    names = subprocess.run([nm['path'], '--defined-only', runtime['path']],
                           check=True, capture_output=True, text=True, timeout=30).stdout
    symbols = [line.split()[-1] for line in names.splitlines() if line.split()]
    undefined = subprocess.run([nm['path'], '-u', runtime['path']],
                               check=True, capture_output=True, text=True, timeout=30).stdout.split()
    if (symbols != ['__wrap_memcpy'] or undefined != ['U', '__real_memcpy']):
        raise ValueError('Only the production memcpy wrapper and nonrecursive SDK fallback are allowed')


def add_link_runtime(argv, runtime_path, *, query1=False):
    command = list(argv)
    if (command.count('-o') != 1 or command[-2] != '-o'
            or command.count('-Wl,--wrap=memrefCopy') != 1
            or any('wrap=memcpy' in word for word in command) or str(runtime_path) in command):
        raise ValueError('Original guarded terminal model link without another memcpy wrapper required')
    # The accepted q1 early gate places this object before the original caller.
    # LLD object order changes code layout; retain that actual tested identity.
    insertion = 1 if query1 else len(command) - 2
    command[insertion:insertion] = [str(runtime_path), '-Wl,--wrap=memcpy']
    return command
