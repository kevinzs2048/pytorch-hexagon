"""Exact N32 K schedule layered on the original compact K/dead V contract."""
KIND = 'gqa-key-tile-q1'
CONTRACT = 'original_Qwen_query1_pure_compact_K_copy_partition_N32_v1'
SDK_STATUS = 'isolated_tiled_K_copy_and_original_dead_repeat_SDK_built_NOT_model_or_DSP'
MODEL_STATUS = 'pure_K_N32_query1_model_built_NOT_device_or_state_acceptance'
RUN_STATUS = 'actual_same_input_q1_pure_K_N32_model_full_packet_gate_passed'
CHANGE = 'partition proved complete pure compact K copies into N32 tiles retaining original K V dead cleanup and ordered FMA'


def validate_key_tile(sdk, diagnostics):
    from gqa_dead_repeat_contract import validate_dead_repeat
    from gqa_shared_copy_contract import compiler_counts
    validate_dead_repeat(sdk, diagnostics)
    compiler_counts(diagnostics, 'gqa-key-compact-q1')
    marker = 'QWEN_GQA_KEY_TILE positions=32 head-position-component-order fresh-complete-destination-proved=1 FP-arithmetic-changed=0'
    if (sdk.get('status') != SDK_STATUS or sdk.get('key_copy_schedule_contract') != CONTRACT
            or sdk.get('key_copy_complete_shape_and_consumers_retained') is not True
            or sdk.get('diagnostic_instrumentation', False) is not False
            or diagnostics.splitlines().count(marker) != 28 or diagnostics.count('QWEN_GQA_KEY_TILE ') != 28):
        raise ValueError('Exactly all original 28 pure complete K producers require qualified N32 scheduling')
    for name in ('source', 'gqa_value_view_source', 'gqa_value_forward_source', 'gqa_repeat_slice_source', 'gqa_key_compact_source'):
        rows = [r for r in sdk['commands'] if '-c' in r['argv'] and r['argv'][r['argv'].index('-c')+1] == sdk[name]['path']]
        if len(rows) != 1 or rows[0]['argv'].count('-DQWEN_GQA_KEY_TILE_N32=1') != int(name == 'gqa_key_compact_source'):
            raise ValueError('Only the actual pure K translation unit may enable N32 scheduling')
