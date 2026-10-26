"""Exact immutable source coordinates replace K copies; FMA consumers remain."""
import re

KIND = 'gqa-key-scalar-forward-q1'
CONTRACT = 'original_Qwen_query1_immutable_K_scalar_source_index_equivalence_v1'
SDK_STATUS = 'isolated_scalar_K_forward_and_original_dead_repeat_SDK_built_NOT_model_or_DSP'
MODEL_STATUS = 'immutable_K_scalar_source_forward_query1_model_built_NOT_device_or_state_acceptance'
RUN_STATUS = 'actual_same_input_q1_K_scalar_forward_model_full_packet_gate_passed'
CHANGE = 'replace all proved pure K scalar reads by exact immutable source coordinates retaining original ordered FP consumers and V dead cleanup'


def validate_key_forward(sdk, diagnostics):
    from gqa_dead_repeat_contract import validate_dead_repeat
    from gqa_shared_copy_contract import compiler_counts
    validate_dead_repeat(sdk, diagnostics)
    compiler_counts(diagnostics, 'gqa-key-compact-q1')
    tile = 'QWEN_GQA_KEY_TILE positions=32 head-position-component-order fresh-complete-destination-proved=1 FP-arithmetic-changed=0'
    forwarded = re.findall(r'^QWEN_GQA_KEY_SCALAR_FORWARD immutable-source-index-equivalence-proved=1 '
        r'scalar-consumers-only=1 removed-complete-K-copy=1 FP-arithmetic-changed=0 reads=(\d+)$', diagnostics, re.M)
    compact = 'QWEN_GQA_KEY_COMPACT heads=8 repeats=2 contiguous-N-retained FP-arithmetic-changed=0 reads=1 SCF-producer-proved=1'
    direct = ('QWEN_GQA_KEY_SCALAR_FORWARD immutable-source-index-equivalence-proved=1 '
        'scalar-consumers-only=1 removed-complete-K-copy=1 FP-arithmetic-changed=0 reads=1')
    events = [line for line in diagnostics.splitlines() if line.startswith(
        ('QWEN_GQA_KEY_COMPACT ', 'QWEN_GQA_KEY_TILE ', 'QWEN_GQA_KEY_SCALAR_FORWARD '))]
    if (sdk.get('status') != SDK_STATUS or sdk.get('K_scalar_forwarding_contract') != CONTRACT
            or sdk.get('original_K_source_values_and_FMA_consumers_retained') is not True
            or sdk.get('contiguous_N_K_layout_retained') is not False
            or sdk.get('key_copy_complete_shape_and_consumers_retained') is not False
            or sdk.get('floating_arithmetic_changed') is not False
            or sdk.get('production_model_acceptance') is not False
            or sdk.get('diagnostic_instrumentation', False) is not False
            or diagnostics.splitlines().count(tile) != 28
            or diagnostics.count('QWEN_GQA_KEY_TILE ') != 28
            or forwarded != ['1']*28 or diagnostics.count('QWEN_GQA_KEY_SCALAR_FORWARD ') != 28
            or events != [compact, tile, direct]*28
            or diagnostics.rindex(direct) > diagnostics.index('QWEN_GQA_DEAD_REPEAT ')):
        raise ValueError('All original 28 K producers require exact scalar-source proof and unchanged FP consumers')
    for name in ('source', 'gqa_value_view_source', 'gqa_value_forward_source', 'gqa_repeat_slice_source', 'gqa_key_compact_source'):
        rows = [r for r in sdk['commands'] if '-c' in r['argv'] and r['argv'][r['argv'].index('-c')+1] == sdk[name]['path']]
        if len(rows) != 1 or type(rows[0]['returncode']) is not int or rows[0]['returncode'] != 0:
            raise ValueError('Every actual compiler translation unit must complete normally')
        for macro in ('-DQWEN_GQA_KEY_TILE_N32=1', '-DQWEN_GQA_KEY_SCALAR_FORWARD=1'):
            if rows[0]['argv'].count(macro) != int(name == 'gqa_key_compact_source'):
                raise ValueError('Only the proved K translation unit may enable scalar forwarding')
