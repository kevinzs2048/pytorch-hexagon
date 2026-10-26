"""Distinct q1 ordered-PV compiler candidate; an ABBA8 is never393 acceptance."""
KIND = 'ordered-pv-register-q1'
SDK_STATUS = 'isolated_ordered_PV_register_q1_model_compiler_built_NOT_DSP'
MODEL_STATUS = 'ordered_PV_register_query1_model_built_NOT_device_or_state_acceptance'
RUN_STATUS = 'actual_same_input_q1_ordered_PV_register_model_full_packet_gate_passed'
CHANGE = 'eight independent PV columns carried across the original ordered K1024 FMA loop after N32/GQA'


def require_reference(closure, audit, reference):
    if (closure['status'] != 'query2to32_dynamic_QK_invariant_vector_native_timing_and_original_states_accepted'
            or closure['DSP_calls'] != 393 or closure['model_inference_acceptance'] is not True
            or audit['status'] != 'query2to32_dynamic_QK_invariant_vector_native393_independently_revisited'
            or audit['closure'] != {k: reference[k] for k in ('path', 'sha256')}
            or audit['DSP_calls'] != 393 or audit['new_DSP_calls'] != 0
            or audit['candidate_libraries'] != closure['candidate_libraries']):
        raise ValueError('Own exact current QK393 and independent numerical audit required')


def require_compiler(sdk, diagnostics):
    if (sdk['status'] != SDK_STATUS
            or any(sdk[k] is not True for k in ('unmodified_SDK_replay_byte_identical',
                'original_link_inputs_unchanged', 'original_SDK_unchanged',
                'original_N32_and_GQA_forwarding_retained', 'per_output_FMA_order_retained'))
            or sdk['read_only_stop_before_bufferization'] is not False
            or sdk['register_transform_diagnostic_only'] is not False
            or sdk['production_model_acceptance'] is not False
            or sdk['floating_arithmetic_changed'] is not False
            or sdk['independent_column_tile'] != 8 or sdk['query_bounds'] != [1, 1]
            or sdk['accepted_sdk']['sha256'] != '4043395bddd404bcb8370f98fe068a3ddf2a481f10e1e11847a355d965f034ce'):
        raise ValueError('Distinct strict ordered q1 compiler and unchanged production dependencies required')
    for marker in ('QWEN_Q1_PV_LOOP_PREFLIGHT matched=28 admitted_functions=1',
                   'QWEN_Q1_PV_REGISTER verified=28 lanes=8 k_order=0:1024:1 no_fast_math=1',
                   'QWEN_Q1_PV_REGISTER_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE',
                   'QWEN_Q1_PV_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE'):
        if diagnostics.count(marker) != 1:
            raise ValueError('Actual complete all28 verified model transform required: ' + marker)
    if 'STOP_NO_MODEL_CODE' in diagnostics:
        raise ValueError('Diagnostic stop cannot be passed off as model code generation')
