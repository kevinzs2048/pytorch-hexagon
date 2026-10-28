"""Transport admission unit tests; no Qwen or DSP execution is claimed."""
import copy
import json

import pytest

from capture_qwen_symbolic_graph_carrier import sha
from hexagon_live_compile_mailbox import (
    atomic_json, checked_library_path, validate_response, wait_for_compilation)


def reply():
    result = dict(status='carried_ARM_Qwen_graph_bound_to_Hexagon_library_NOT_DSP',
        producer={'sha256': 'a'*64}, DSP_calls=0, CPU_model_forward=False,
        compiled_entry={'graph': {'sha256': 'b'*64}})
    response = dict(schema=1, producer_sha256='a'*64, status='compiled')
    return response, result


def test_response_binds_actual_request():
    response, result = reply()
    assert validate_response(response, 'a'*64, result) == result['compiled_entry']


@pytest.mark.parametrize('fault', ['stale_response', 'stale_result', 'failed', 'status', 'cpu', 'dsp'])
def test_rejects_stale_or_unexecuted_result_claims(fault):
    response, result = reply()
    if fault == 'stale_response':
        response['producer_sha256'] = 'c'*64
    elif fault == 'stale_result':
        result['producer']['sha256'] = 'c'*64
    elif fault == 'failed':
        response['status'] = 'failed'
    elif fault == 'status':
        result['status'] = 'compiling'
    elif fault == 'cpu':
        result['CPU_model_forward'] = True
    else:
        result['DSP_calls'] = 1
    with pytest.raises(ValueError, match='response'):
        validate_response(response, 'a'*64, result)


def test_wait_reads_complete_published_product(tmp_path):
    response, result = reply()
    (tmp_path/'compiled').mkdir()
    path = tmp_path/'compiled'/'result.json'
    atomic_json(path, result)
    response['result_sha256'] = sha(path)
    atomic_json(tmp_path/'response.json', response)
    assert wait_for_compilation(tmp_path, 'a'*64, timeout=1) == (result['compiled_entry'], result)
    path.write_text(json.dumps(dict(result, DSP_calls=1)))
    with pytest.raises(ValueError, match='changed'):
        wait_for_compilation(tmp_path, 'a'*64, timeout=1)


def test_failed_compiler_returns_failure_without_wait_or_fallback(tmp_path):
    atomic_json(tmp_path/'response.json', {'status': 'failed', 'error': 'compile rejected'})
    with pytest.raises(RuntimeError, match='compile rejected'):
        wait_for_compilation(tmp_path, 'a'*64, timeout=1)


def test_timeout_is_not_fallback(tmp_path):
    with pytest.raises(TimeoutError, match='no CPU model fallback'):
        wait_for_compilation(tmp_path, 'a'*64, timeout=0)


def test_library_is_scoped_and_rehashed(tmp_path):
    cache = tmp_path/'cache'
    cache.mkdir()
    path = cache/'code.so'
    path.write_bytes(b'actual machine bytes')
    assert checked_library_path(cache, 'code.so', sha(path)) == path
    with pytest.raises(ValueError, match='bytes changed'):
        checked_library_path(cache, 'code.so', '0'*64)


@pytest.mark.parametrize('relative', ['/tmp/code.so', '../code.so', 'build/../../code.so'])
def test_library_path_escape_rejected_before_read(tmp_path, relative):
    with pytest.raises(ValueError, match='inside selected'):
        checked_library_path(tmp_path, relative, '0'*64)


def test_library_parent_symlink_escape_rejected(tmp_path):
    cache, external = tmp_path/'cache', tmp_path/'external'
    cache.mkdir(); external.mkdir()
    path = external/'code.so'
    path.write_bytes(b'code')
    (cache/'build').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='outside cache'):
        checked_library_path(cache, 'build/code.so', sha(path))


@pytest.mark.parametrize('fault', ['prior_negative', 'prior_bool', 'cpu', 'source',
                                  'architecture', 'torch_git', 'generation'])
def test_live_producer_admission_before_pickle(tmp_path, monkeypatch, fault):
    import torch
    from torch.fx._graph_pickler import GraphPickler
    import hexagon_live_compile_mailbox as live
    from restore_qwen_symbolic_graph_carrier import restore_carrier
    record = dict(status=live.LIVE_STATUS, source_sha256=sha(live.__file__),
        host_machine='aarch64', prior_session_DSP_calls=7, request_graph_executed=False,
        CPU_model_forward=False, context_sha256=live.CONTEXT_SHA,
        generation_sha256=live.GENERATION_SHA, decomposition_sha256=live.POLICY_SHA,
        torch_version=torch.__version__, torch_git=torch.version.git_version)
    if fault == 'prior_negative':
        record['prior_session_DSP_calls'] = -1
    elif fault == 'prior_bool':
        record['prior_session_DSP_calls'] = True
    elif fault == 'cpu':
        record['CPU_model_forward'] = True
    elif fault == 'source':
        record['source_sha256'] = '0'*64
    elif fault == 'architecture':
        record['host_machine'] = 'other'
    elif fault == 'torch_git':
        record['torch_git'] = 'other'
    else:
        record['generation_sha256'] = '0'*64
    path = tmp_path/'result.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr(GraphPickler, 'loads', lambda *args: pytest.fail('Must reject before pickle'))
    with pytest.raises(ValueError, match='pinned'):
        restore_carrier(path, sha(path), trusted_local=True)


def test_worker_lifetime_follows_session_not_predicted_graph_count(tmp_path):
    from audit_arm_dynamic_warm_run import session_closed
    assert session_closed(tmp_path, 0) is False
    atomic_json(tmp_path/'session-closed.json', dict(schema=1, requests_submitted=2,
        status='ARM_native_dynamic_compiler_handoff_generate_completed'))
    assert session_closed(tmp_path, 2) is True
    with pytest.raises(ValueError, match='accounting'):
        session_closed(tmp_path, 1)


@pytest.mark.parametrize('fault', ['failed', 'bool_count', 'schema', 'symlink'])
def test_session_close_does_not_hide_failure_or_missing_transactions(tmp_path, fault):
    from audit_arm_dynamic_warm_run import session_closed
    marker = tmp_path/'session-closed.json'
    record = dict(schema=1, requests_submitted=1,
        status='ARM_native_dynamic_compiler_handoff_generate_completed')
    if fault == 'failed':
        record['status'] = 'failed'
    elif fault == 'bool_count':
        record['requests_submitted'] = True
    elif fault == 'schema':
        record['schema'] = 2
    if fault == 'symlink':
        target = tmp_path/'target.json'
        atomic_json(target, record)
        marker.symlink_to(target)
    else:
        atomic_json(marker, record)
    with pytest.raises(ValueError):
        session_closed(tmp_path, 1)


def test_completion_cannot_conceal_an_unserved_ready_request(tmp_path):
    from audit_arm_dynamic_warm_run import session_closed
    atomic_json(tmp_path/'session-closed.json', dict(schema=1, requests_submitted=0,
        status='ARM_native_dynamic_compiler_handoff_generate_completed'))
    (tmp_path/'capture-00').mkdir()
    atomic_json(tmp_path/'capture-00'/'ready.json', {'schema': 1})
    with pytest.raises(ValueError, match='accounting'):
        session_closed(tmp_path, 0)
