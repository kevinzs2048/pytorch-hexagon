"""Resident session constant inventory: the four shared constant libraries of an
INT8 session, or N shared libraries plus the Q8_0 weight library."""
import hashlib
import json

import pytest

from models.qwen3.resident import Q8_0_LIBRARY, load_session

CHECKPOINT = 'ab' * 32
BASE = {'files': {'hf/model.safetensors': {'sha256': CHECKPOINT}}}


def session(tmp_path, constants, q8=None, modes=None):
    names = ['model.so', 'decompositions.py', *constants]
    names += [t + s for t in ('initial', 'prefill', 'decode', 'single0', 'single1')
              for s in ('-graph.txt', '-mapping.json')]
    files = {}
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
        files[name] = dict(bytes=len(name), sha256=hashlib.sha256(name.encode()).hexdigest())
    manifest = dict(schema='qwen3-resident-session-v1', constants=constants, checkpoint_sha256=CHECKPOINT, files=files)
    if q8 is not None:
        manifest['q8_0_projections'] = q8
    if modes is not None:
        manifest['resident_forward_modes'] = modes
    (tmp_path / 'session.json').write_text(json.dumps(manifest))
    return tmp_path


def shared(n):
    return [f'libqwen_dynamic_kv_prefill_const_{i}.so' for i in range(1, n + 1)]


def q8(n, **extra):
    return dict(dict(library=Q8_0_LIBRARY, projections=197, shared_constant_libraries=n,
                     weights={}, numerics={}), **extra)


def test_int8_session_keeps_exactly_four_shared_libraries(tmp_path):
    assert load_session(session(tmp_path, shared(4)), BASE)['constants'] == shared(4)


def test_int8_session_with_fewer_libraries_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='constant inventory'):
        load_session(session(tmp_path, shared(2)), BASE)


def test_q8_0_session_lists_shared_libraries_then_the_weight_library(tmp_path):
    assert load_session(session(tmp_path, shared(2) + [Q8_0_LIBRARY], q8(2)), BASE)


@pytest.mark.parametrize('constants, record', [
    (shared(2), q8(2)),                                   # weight library missing
    (shared(3) + [Q8_0_LIBRARY], q8(2)),                  # count differs from the record
    ([Q8_0_LIBRARY] + shared(2), q8(2)),                  # weight library not last
    (shared(2) + [Q8_0_LIBRARY], q8(2, extra=1)),         # unknown record field
    (shared(5) + [Q8_0_LIBRARY], q8(5)),                  # more than four shared libraries
    (shared(2) + [Q8_0_LIBRARY], dict(q8(2), library='libother.so')),
])
def test_inconsistent_q8_0_inventory_is_rejected(tmp_path, constants, record):
    with pytest.raises(ValueError, match='Q8_0 projection inventory'):
        load_session(session(tmp_path, constants, record), BASE)


def test_forward_without_logits_is_a_q8_0_session_mode(tmp_path):
    constants = shared(2) + [Q8_0_LIBRARY]
    assert load_session(session(tmp_path, constants, q8(2), modes=[1, 3, 4]), BASE)
    assert load_session(session(tmp_path, constants, q8(2), modes=[1, 3, 4, 5]), BASE)
    for modes, record, libraries in (([1, 3, 4], None, shared(4)), ([1, 4], q8(2), constants),
                                     ([1, 3, 5], q8(2), constants), ([1, 3, 4, 5], None, shared(4)),
                                     ([1, 3, 4, 5, 6], q8(2), constants)):
        with pytest.raises(ValueError, match='forward modes'):
            load_session(session(tmp_path, libraries, record, modes=modes), BASE)


def test_q8_0_shared_libraries_must_be_numbered(tmp_path):
    constants = ['libqwen_dynamic_kv_prefill_const_2.so', 'libqwen_dynamic_kv_prefill_const_1.so', Q8_0_LIBRARY]
    with pytest.raises(ValueError, match='constant inventory'):
        load_session(session(tmp_path, constants, q8(2)), BASE)
