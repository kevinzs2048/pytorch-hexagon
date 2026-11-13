"""Check the changed cache ABI and reject ambiguous storage bindings."""
import os
import shutil
import subprocess

import pytest
from apply_dynamic_bound_qk_policy import FunctionPolicyPatch


def test_streamed_annotation_uses_captured_planes_and_rejects_changed_abi():
    args = ', '.join(f'%arg{i+3}: tensor<1x8x?x128xf32>' for i in range(56))
    source = ('module {\n  func.func @Captured(' + args + ') -> () {\n'
              '    return\n  }\n}\n').encode()
    planes = {i+3: i ^ 1 for i in range(56)}

    def patch(data):
        chunks = []
        stream = FunctionPolicyPatch(chunks.append, function_name='Captured',
            effective_attention='prefill', resident_planes=planes)
        for offset in range(0, len(data), 7):
            stream.write(data[offset:offset+7])
        stream.finish()
        return b''.join(chunks)

    changed = patch(source)
    assert b'%arg3: tensor<1x8x?x128xf32> {hexagon.kv_plane = 1 : i64}' in changed
    assert b'%arg4: tensor<1x8x?x128xf32> {hexagon.kv_plane = 0 : i64}' in changed
    assert b'fp32_effective_prefill_attention = true' in changed
    assert b'fp32_prefill_capacity' not in changed
    with pytest.raises(ValueError, match='types/order differ'):
        patch(source.replace(b'%arg3: tensor<1x8x?x128xf32>', b'%arg3: tensor<1x8x1x128xf32>'))


SOURCE = '''
module {
  func.func @append(%past: tensor<1x8x?x128xf32> {hexagon.kv_plane = 1 : i64},
                    %new: tensor<1x8x1x128xf32>)
      -> (tensor<1x8x?x128xf32>, f32) attributes {hexagon.resident_kv} {
    %c0 = arith.constant 0 : index
    %joined = tensor.concat dim(2) %past, %new
      : (tensor<1x8x?x128xf32>, tensor<1x8x1x128xf32>) -> tensor<1x8x?x128xf32>
    %first = tensor.extract %joined[%c0, %c0, %c0, %c0] : tensor<1x8x?x128xf32>
    return %joined, %first : tensor<1x8x?x128xf32>, f32
  }
}
'''


def run(source, count=1):
    tool = os.environ.get('QWEN_RESIDENT_KV_OPT') or shutil.which('qwen-resident-kv-opt')
    if not tool:
        pytest.skip('build QWEN_BUILD_RESIDENT_KV and set QWEN_RESIDENT_KV_OPT')
    return subprocess.run([tool, '-', '--verify-each',
                           '--qwen-resident-kv=expected-count='+str(count)],
                          input=source, text=True, capture_output=True, timeout=30)


def test_cache_return_is_a_delta_and_mapping_preserves_plane_order():
    result = run(SOURCE)
    assert result.returncode == 0, result.stderr
    assert '-> (tensor<1x8x1x128xf32>, f32)' in result.stdout
    assert 'hexagon.kv_delta_planes = array<i64: 1, -1>' in result.stdout
    assert 'hexagon.resident_kv_capacity_argument = 2 : i64' in result.stdout
    assert 'tensor.concat' not in result.stdout
    assert result.stdout.count('memref.copy') == 1
    assert 'resident KV append exceeds reserved capacity' in result.stdout
    assert 'strided<[?, 128, 1024, 1]>' in result.stdout


def test_unannotated_and_incomplete_bindings_are_rejected():
    for source, count in [(SOURCE.replace('hexagon.resident_kv', 'unrelated'), 1),
                          (SOURCE, 2),
                          (SOURCE.replace('hexagon.kv_plane = 1', 'hexagon.kv_plane = -1'), 1)]:
        assert run(source, count).returncode != 0


def test_key_attention_workspace_keeps_append_and_return_as_new_rows_only():
    key = run(SOURCE.replace('hexagon.kv_plane = 1', 'hexagon.kv_plane = 0'))
    assert key.returncode == 0, key.stderr
    assert 'hexagon.resident_key_head_major' in key.stdout
    assert 'tensor.generate' in key.stdout
    assert key.stdout.count('memref.copy') == 1
    assert 'hexagon.kv_delta_planes = array<i64: 0, -1>' in key.stdout
    assert '-> (tensor<1x8x1x128xf32>, f32)' in key.stdout
    value = run(SOURCE)
    assert value.returncode == 0, value.stderr
    assert 'hexagon.resident_key_head_major' not in value.stdout


def test_extra_observation_of_the_old_tensor_is_rejected():
    source = SOURCE.replace('tensor.extract %joined', 'tensor.extract %past')
    result = run(source)
    assert result.returncode != 0
    assert 'one direct cache concat' in result.stderr


def test_singleton_history_preserves_static_concat_extent():
    source = SOURCE.replace('1x8x?x128', '1x8x2x128')
    source = source.replace('%past: tensor<1x8x2x128', '%past: tensor<1x8x1x128')
    source = source.replace(': (tensor<1x8x2x128', ': (tensor<1x8x1x128')
    result = run(source)
    assert result.returncode == 0, result.stderr
    assert 'memref<1x8x2x128xf32' in result.stdout
    assert '-> (tensor<1x8x1x128xf32>, f32)' in result.stdout
    assert 'tensor.concat' not in result.stdout
    assert result.stdout.count('memref.copy') == 1
