"""Change only the117 tensor transfers of an authenticated two-graph caller."""
import re


CONTRACT = 'owned_caller_only_exact_HVX_byte_copy_v1'


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
