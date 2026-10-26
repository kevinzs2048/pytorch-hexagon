"""Sealed opt-in Qwen buffer ABI, separate from generation and numerical policy."""
from qwen3.compiler_dependency import compiler_root
import importlib.util
import json
from pathlib import Path
from hexagon_generate_reuse import sha


def local_buffer_io():
    # Load only this repository's pure ABI utility, never executable code from
    # an evidence record or a remote path.
    path=compiler_root()/'qcom_hexagon_backend/backend/hexagon_buffer_io.py'
    spec=importlib.util.spec_from_file_location('local_hexagon_buffer_abi',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_buffer_binding(compiled):
    binding=compiled.get('buffer_interface')
    graph=compiled['graphs'][0]
    if graph.get('buffer_interface')!=binding or compiled.get('wrapper_relink',{}).get('buffer_interface')!=binding:
        raise ValueError('Run, graph and relink disagree on buffer ABI')
    if binding is None:return None
    if (binding.get('schema')!=1 or binding.get('symbol')!='hexagon_forward_buffers'
            or type(binding.get('validate_input_copy')) is not bool
            or compiled.get('capacity') not in (32,64) or compiled.get('returned_buffer_policy')!='owned-DDR-single-call'
            or binding.get('input_allocation')!='copy-into-original-aligned-Tensor'
            or binding.get('zero_copy') is not False):
        raise ValueError('Unsupported buffer ABI or false memory ownership claim')
    for key in ('layout','generator_source'):
        item=binding[key]
        if sha(item['path'])!=item['sha256']:raise ValueError('Buffer ABI source/layout drift')
    bio=local_buffer_io();layout=json.loads(Path(binding['layout']['path']).read_text())
    bio.validate_layout(layout)
    capacity=compiled['capacity']
    expected=bio.make_layout(
        [dict(dtype='int64',shape=[1,1]),dict(dtype='int64',shape=[1])]+[
            dict(dtype='float32',shape=[1,8,capacity,128])]*56,
        [dict(dtype='float32',shape=[1,1,151936])]+[dict(dtype='float32',shape=[1,8,capacity,128])]*56)
    if layout!=expected:raise ValueError('Qwen buffer dimensions/order differ from established ABI')
    return layout


def buffer_entry_suffix(compiled,wrapper):
    layout=validate_buffer_binding(compiled)
    if layout is None:return ''
    from triton.backends.qcom_hexagon_backend import hexagon_buffer_io as bio
    binding=compiled['buffer_interface']
    if sha(bio.__file__)!=binding['generator_source']['sha256']:
        raise ValueError('Buffer code generator differs from compiled source; refuse silent regeneration')
    code=bio.generate_buffer_entry(wrapper,layout,validate_input_copy=binding['validate_input_copy'])
    import hashlib
    if hashlib.sha256(code.encode()).hexdigest()!=binding['entry_code_sha256']:
        raise ValueError('Generated buffer entry differs from sealed compiled entry')
    return code
