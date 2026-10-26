"""Explicit early-model contracts; neither admits a native generation recipe."""
import re


CONTRACTS = {
    'gqa-value-forward-q1': dict(
        variant='gqa-value-forward',
        sdk_status='isolated_gqa_value_forward_and_view_CSE_SDK_built_DIAGNOSTIC_NOT_model_or_DSP',
        primitive_status='standalone_V_forward_actual_ABBA32_independently_revisited_NOT_model_acceptance',
        model_status='GQA_value_forward_query1_model_built_NOT_device_or_state_acceptance',
        run_status='actual_same_input_q1_GQA_value_forward_model_full_packet_gate_passed',
        change='proven GQA V repeat scalar forwarding and original identity V shape view'),
    'gqa-key-compact-q1': dict(
        variant='gqa-key-compact',
        sdk_status='isolated_gqa_key_compact_V_forward_view_CSE_SDK_built_DIAGNOSTIC_NOT_model_or_DSP',
        primitive_status='standalone_K_compact_actual_ABBA32_independently_revisited_NOT_model_acceptance',
        model_status='GQA_key_compact_query1_model_built_NOT_device_or_state_acceptance',
        run_status='actual_same_input_q1_GQA_key_compact_model_full_packet_gate_passed',
        change='proven original SCF eight-head contiguous K transpose sharing and GQA V repeat slices'),
}


def qualified_primitive(proof, kind):
    contract = CONTRACTS[kind]
    timings = proof.get('measurements', {})
    return (proof.get('status') == contract['primitive_status'] and proof.get('DSP_calls') == 32
            and (kind != 'gqa-key-compact-q1' or proof.get('original_SCF_copy_structure_proved') is True)
            and proof.get('model_inference_acceptance') is False
            and timings.get('N1024-normal', {}).get('ratio', 1) < .995
            and all(timings.get('N%d-normal' % n, {}).get('ratio', 2) <= 1.05
                    for n in (127, 128, 129)))


def compiler_counts(diagnostics, kind):
    if kind in CONTRACTS:
        actual=re.findall(r'QWEN_GQA_REPEAT_SLICE_FORWARD slices=(\d+) copies=(\d+) FP-arithmetic-changed=0 vector-load-and-FMA-order-retained=1',diagnostics)
        if (actual!=[('28','28')] or 'QWEN_GQA_VALUE_FORWARD reads=' in diagnostics or
                diagnostics.count('QWEN_GQA_VALUE_VIEW eliminated=28 FP-arithmetic-changed=0')!=1 or
                diagnostics.find('QWEN_GQA_REPEAT_SLICE_FORWARD slices=') < diagnostics.find('QWEN_GQA_VALUE_VIEW eliminated=28 ') or
                diagnostics.count('QWEN_GQA_VALUE_VIEW eliminated_identity_copy ')!=28):
            raise ValueError('All28 original identity views must precede all28 batch-one repeat forwards')
        keys=re.findall(r'QWEN_GQA_KEY_COMPACT heads=8 repeats=2 contiguous-N-retained FP-arithmetic-changed=0 reads=(\d+) SCF-producer-proved=1',diagnostics)
        if kind=='gqa-key-compact-q1':
            if (keys!=['1']*28 or diagnostics.count('QWEN_GQA_KEY_COMPACT ')!=28 or
                    diagnostics.find('QWEN_GQA_KEY_COMPACT ') < diagnostics.find('QWEN_GQA_REPEAT_SLICE_FORWARD slices=')):
                raise ValueError('Exactly28 actual original SCF K copies and their complete scalar readers required')
        elif 'QWEN_GQA_KEY_COMPACT ' in diagnostics:
            raise ValueError('Value-only candidate cannot change K copies')
        return dict(eliminated_identity_value_copies=28,forwarded_value_reads=28,
            compact_key_copies=len(keys),forwarded_repeat_reads=28,forwarded_layout_reads=0,
            forwarded_repeat_component_slices=28)
    raise ValueError('Explicit shared-copy candidate kind required')
