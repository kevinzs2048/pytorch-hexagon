"""Change only the117 tensor transfers of an authenticated two-graph caller."""
from qwen3.compiler_dependency import compiler_root
import re
from pathlib import Path
import struct
import subprocess

from hexagon_compiled_artifact_cache import digest_file

CONTRACT = 'owned_caller_only_exact_HVX_byte_copy_v1'


def validate_runtime(runtime, source, nm):
    for row in (runtime, source, nm):
        if (type(row) is not dict or set(row)!={'path','sha256'}
                or not Path(row['path']).is_absolute() or digest_file(row['path'])!=row['sha256']):
            raise ValueError('Pinned caller copy object, source and inspector required')
    actual_source=compiler_root()/'tools/qwen3/runtime/profile/caller_hvx_byte_copy.c'
    if source['sha256']!=digest_file(actual_source):
        raise ValueError('Only the actual reviewed caller copy implementation is admitted')
    header=Path(runtime['path']).read_bytes()[:20]
    if len(header)!=20 or header[:6]!=b'\x7fELF\x01\x01' or struct.unpack_from('<HH',header,16)!=(1,164):
        raise ValueError('ELF32 little-endian Hexagon caller copy object required')
    defined=subprocess.check_output([nm['path'],'--defined-only',runtime['path']],text=True).split()
    undefined=subprocess.check_output([nm['path'],'-u',runtime['path']],text=True).split()
    if defined!=['00000000','T','qwen_caller_byte_copy'] or undefined!=['U','memcpy']:
        raise ValueError('Only caller byte transfers and existing memcpy fallback allowed')


def add_link_runtime(command, runtime):
    if command.count('-o')!=1 or command[-2]!='-o' or str(runtime) in command:
        raise ValueError('Original terminal model link without another caller copy runtime required')
    result=list(command);result.insert(1,str(runtime));return result


def rewrite_caller(text):
    boundary = 'extern "C" int hexagon_forward_buffers('
    if text.count(boundary) != 1 or 'qwen_caller_byte_copy' in text:
        raise ValueError('One original noninstrumented caller entry required')
    prefix, body = text.split(boundary, 1)
    if 'qwen_memory_profile_' in body:
        raise ValueError('Diagnostic instrumentation is not a production caller')
    input_pattern = r'(?m)^  std::memcpy\(dt(\d+)->aligned, input \+ [^\n]+;\s*$'
    output_pattern = r'(?m)^  std::memcpy\(output \+ out_offset_(\d+), r->r\1\.aligned, [^\n]+;\s*$'
    inputs = [int(match.group(1)) for match in re.finditer(input_pattern, body)]
    outputs = [int(match.group(1)) for match in re.finditer(output_pattern, body)]
    if inputs != list(range(60)) or outputs != list(range(57)):
        raise ValueError('All60 original owned tensor inputs and57 outputs required')
    changed = re.sub(input_pattern, lambda m: m.group(0).replace('std::memcpy', 'qwen_caller_byte_copy'), body)
    changed = re.sub(output_pattern, lambda m: m.group(0).replace('std::memcpy', 'qwen_caller_byte_copy'), changed)
    if changed.replace('qwen_caller_byte_copy', 'std::memcpy') != body:
        raise ValueError('Only tensor transfer callees may change')
    declaration = 'extern "C" void *qwen_caller_byte_copy(void *, const void *, size_t);\n'
    return prefix+declaration+boundary+changed
