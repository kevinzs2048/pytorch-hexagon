import hashlib
import sys
import time
import subprocess

import pytest

from apply_unregistered_prefill_policy import annotate_prefix, stream_policy


PREFIX = b'module { func.func @f() attributes {hexagon.fp32_prefill_capacity = 1024 : i64} {return}}'


def test_annotation_is_explicit_and_preserves_original_text():
    result = annotate_prefix(PREFIX)
    assert result.replace(b'hexagon.fp32_prefill_order = "qk-fma-pv-openblas-armv8-k1024", ', b'') == PREFIX


@pytest.mark.parametrize('blob', [b'', PREFIX+PREFIX,
                                PREFIX+b'hexagon.fp32_prefill_order = "other"'])
def test_ambiguous_or_already_annotated_input_rejected(blob):
    with pytest.raises(ValueError):
        annotate_prefix(blob)


def test_stream_hashes_original_and_annotated_bytes(tmp_path):
    source = tmp_path/'source.mlir'
    source.write_bytes(PREFIX+b' '*100000)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    argv = [sys.executable, '-c', 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())']
    proof = stream_policy(argv, source, digest, tmp_path/'received', timeout=5)
    assert proof['original_sha256'] == digest
    assert proof['annotated_stream_sha256'] == hashlib.sha256((tmp_path/'received').read_bytes()).hexdigest()
    assert proof['duplicate_raw_IR_written'] is False


def test_changed_source_bytes_rejected(tmp_path):
    source = tmp_path/'source.mlir'
    source.write_bytes(PREFIX)
    with pytest.raises(ValueError):
        stream_policy([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'], source, '0'*64,
                      tmp_path/'received', timeout=5)


def test_silent_child_has_deadline(tmp_path):
    source = tmp_path/'source.mlir'
    source.write_bytes(PREFIX)
    start = time.monotonic()
    with pytest.raises((ValueError, BrokenPipeError, subprocess.TimeoutExpired)):
        stream_policy([sys.executable, '-c', 'import time; time.sleep(10)'], source,
                      hashlib.sha256(PREFIX).hexdigest(), tmp_path/'received', timeout=0.1)
    assert time.monotonic()-start < 3
