"""Load the pinned compiler buffer ABI without importing Triton.
"""
from qwen3.compiler_dependency import compiler_root
import importlib.util


def local_buffer_io():
    # Load only this repository's pure ABI utility, never executable code from
    # an evidence record or a remote path.
    path=compiler_root()/'qcom_hexagon_backend/backend/hexagon_buffer_io.py'
    spec=importlib.util.spec_from_file_location('local_hexagon_buffer_abi',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
