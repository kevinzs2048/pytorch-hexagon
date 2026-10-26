"""Same query1 memref ABI, signed I8 products and INT32 ring; no FP operations.

Each original four-row pack maps four signed products to the same output lane.
The eight independent accumulators interleave two original column blocks.
INT32 addition modulo 2**32 is associative, including unrestricted initial C.
K/N tails and all strided layouts use the unchanged original implementation.
"""
from pathlib import Path
import struct
import subprocess

from hexagon_compiled_artifact_cache import digest_file

CONTRACT='original_Qwen_query1_i8_tiles256_INT32_ring_v1'
SOURCES=(
    '62e082c20ce5bbe96da1642ce8f59df177e6c9246572892c9b6fe62a2eca9c19',
    'e68e6bb9096c967a0d66114fb1f82f4375465b97e4f8e73a25abdb040d6d6c0e',
    '78f1f5441c3f222c88e51d9cf228537ba133f105348015cc15e0652f854050da')
RUNTIME='a29526b4e48922fa407cecf0cad770d175570d18796f6d7a2a20dc515aed16cd'


def validate_runtime(runtime,nm,source,header,original):
    for row in (runtime,nm,source,header,original):
        if (type(row) is not dict or set(row)!={'path','sha256'} or not Path(row['path']).is_absolute()
                or digest_file(row['path'])!=row['sha256']):
            raise ValueError('Pinned production integer object, inspector and exact sources required')
    if tuple(row['sha256'] for row in (source,header,original))!=SOURCES or runtime['sha256']!=RUNTIME:
        raise ValueError('Only the actual DSP-tested integer production object and original pack are admitted')
    with Path(runtime['path']).open('rb') as f: elf=f.read(20)
    if elf[:6]!=b'\x7fELF\x01\x01' or len(elf)!=20 or struct.unpack_from('<HH',elf,16)!=(1,164):
        raise ValueError('ELF32 little-endian Hexagon relocatable integer runtime required')
    names=subprocess.run([nm['path'],'--defined-only',runtime['path']],check=True,capture_output=True,text=True,timeout=30).stdout
    exports=[line.split()[-1] for line in names.splitlines() if line.split() and line.split()[-2].isupper()]
    if exports!=['_mlir_ciface___qwen_i8_matmul'] or 'qwen_packed_probe_call_count' in names:
        raise ValueError('Original integer ABI and absent production instrumentation required')
