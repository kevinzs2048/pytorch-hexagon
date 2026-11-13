"""Admission and dead-computation checks against an actual prepared Qwen graph.

Set QWEN_FUSED_ATTENTION_OPT and QWEN_FUSED_ATTENTION_IR (prefill, 28 layers).
Weights are elided for these structural tests; never compile the printed fixture.
"""
import os
import re
import subprocess

import pytest

OPT = os.environ.get('QWEN_FUSED_ATTENTION_OPT')
IR = os.environ.get('QWEN_FUSED_ATTENTION_IR')
pytestmark = pytest.mark.skipif(not (OPT and IR), reason='Prepared model and optimizer required')


@pytest.fixture(scope='module')
def source():
    result = subprocess.run([OPT, IR, '--mlir-elide-resource-strings-if-larger=16',
                             '--mlir-elide-elementsattrs-if-larger=16'],
                            capture_output=True, text=True, check=True, timeout=60)
    return result.stdout


def run(source, count=28, implementation='fp32-ordered-workers-v1'):
    return subprocess.run([OPT, '-', '--verify-each', '--mlir-print-op-on-diagnostic=false',
                           '--qwen-fused-attention=expected-count='+str(count)+' implementation='+implementation,
                           '--canonicalize'],
                          input=source, capture_output=True, text=True, timeout=60)


def test_rejects_incomplete_coverage(source):
    result = run(source, 27)
    assert result.returncode != 0 and 'count differs' in result.stderr


def test_ordered_fp32_uses_separate_abi_and_removes_old_computation(source):
    result = run(source, implementation='fp32-ordered-workers-v1')
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('call @qwen_ordered_fp32_attention_v1') == 28
    assert 'call @qwen_fused_attention' not in result.stdout
    assert 'qwen.attention_execution = "fp32-ordered-workers-v1"' in result.stdout
    assert result.stdout.count('memref.copy') == source.count('memref.copy') == 56
    for marker in ('hexagon.fp32_prefill_dot_order', 'hexagon.fp32_effective_prefill_pv_order',
                   'hexagon.fp32_arm_softmax_sum', 'hexagon.resident_key_head_major'):
        assert marker not in result.stdout


def test_rejects_unknown_runtime(source):
    result = run(source, implementation='unknown')
    assert result.returncode != 0 and 'unsupported attention implementation' in result.stderr


@pytest.fixture(scope='module')
def decode_source():
    path = os.environ.get('QWEN_ORDERED_DECODE_IR')
    if not path:
        pytest.skip('Prepared decoder required')
    result = subprocess.run([OPT, path, '--mlir-elide-resource-strings-if-larger=16',
                             '--mlir-elide-elementsattrs-if-larger=16'],
                            capture_output=True, text=True, check=True, timeout=60)
    return result.stdout


def test_ordered_decode_removes_both_qk_branches_and_pv(decode_source):
    result = run(decode_source, implementation='fp32-ordered-workers-v1')
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('call @qwen_ordered_fp32_attention_v1') == 28
    assert 'hexagon.fp32_effective_decode_order' not in result.stdout
    assert 'hexagon.fp32_effective_decode_fma' not in result.stdout
    assert result.stdout.count('memref.copy') == decode_source.count('memref.copy') == 56


def test_ordered_decode_rejects_changed_native_switch(decode_source):
    before_qk = decode_source.split('hexagon.fp32_effective_decode_fma', 1)[0]
    compare = next(line for line in reversed(before_qk.splitlines()) if ' = arith.cmpi ult,' in line)
    threshold = re.search(r', (%[\w]+) : index', compare).group(1)
    changed, count = re.subn(r'('+re.escape(threshold)+r' = arith.constant )4( : index)',
                             r'\g<1>3\2', decode_source, count=1)
    assert count == 1
    assert changed != decode_source
    result = run(changed, implementation='fp32-ordered-workers-v1')
    assert result.returncode != 0 and 'unsupported ordered decode' in result.stderr


def test_ordered_decode_rejects_changed_fma_iterator_order(decode_source):
    lines = decode_source.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if 'hexagon.fp32_effective_decode_fma' in line)
    lines[index] = lines[index].replace('iterator_types = ["parallel", "parallel", "reduction", "parallel"]',
                                       'iterator_types = ["parallel", "parallel", "parallel", "reduction"]', 1)
    changed = ''.join(lines)
    assert changed != decode_source
    result = run(changed, implementation='fp32-ordered-workers-v1')
    assert result.returncode != 0 and 'unsupported ordered decode' in result.stderr


def test_rejects_changed_scaling(source):
    changed, count = re.subn(r'dense<0\.088388\d*> : tensor<f32>',
                            'dense<0.1> : tensor<f32>', source)
    assert count
    result = run(changed)
    assert result.returncode != 0 and 'unsupported fused attention dataflow' in result.stderr


def test_rejects_missing_pv_policy(source):
    changed = source.replace('hexagon.fp32_effective_prefill_pv_order',
                             'test.unrecognized_pv', 1)
    assert changed != source
    result = run(changed)
    assert result.returncode != 0 and 'unsupported fused attention dataflow' in result.stderr


def test_rejects_observable_old_attention(source):
    lines = source.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if 'hexagon.fp32_prefill_dot_order' in line)
    value = lines[start].split('=')[0].strip()
    end = next(i for i in range(start+1, len(lines)) if '} -> tensor<' in lines[i])
    tensor = lines[end].split('->', 1)[1].strip()
    lines.insert(end+1, f'    func.call @test_observe({value}) : ({tensor}) -> ()\n')
    changed = ''.join(lines).replace('module {',
        f'module {{\n  func.func private @test_observe({tensor})', 1)
    result = run(changed)
    assert result.returncode != 0 and 'extra observable results' in result.stderr
