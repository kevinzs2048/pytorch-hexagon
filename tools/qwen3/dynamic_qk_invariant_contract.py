"""Distinct guarded query2..32 QK invariant-vector candidate; never CSE-only acceptance."""
KIND='dynamic-qk-invariant-vector-q2to32'
SDK_STATUS='isolated_dynamic_QK_invariant_vector_and_CSE_SDK_built_NOT_model_or_DSP'
MODEL_STATUS='dynamic_QK_invariant_vector_query2to32_model_built_NOT_device_or_state_acceptance'
RUN_STATUS='actual_same_input_dynamic_QK_invariant_vector_full_packet_gate_passed'
CHANGE='original invariant Query scalar hoist and independent column FMA vectorization after original CSE'


def require_reference(closure,audit,reference):
    if (closure.get('status')!='query2to32_empty_tensor_CSE_native_timing_and_original_states_accepted'
            or closure.get('candidate_kind')!='empty-tensor-cse-q2to32'
            or type(closure.get('DSP_calls')) is not int or closure['DSP_calls']!=393
            or closure.get('model_inference_acceptance') is not True
            or audit.get('status')!='query2to32_empty_tensor_CSE_native393_independently_revisited'
            or audit.get('closure')!=reference or audit.get('DSP_calls')!=393
            or audit.get('new_DSP_calls')!=0 or audit.get('model_inference_acceptance') is not True
            or audit.get('candidate_DSP_calls')!=361 or audit.get('paired_baseline_DSP_calls')!=32
            or audit.get('numerical_source_revision')!=closure.get('source_revision')):
        raise ValueError('Own complete independently accepted original dynamic CSE393 required')
    return closure['candidate_libraries']['(2, 32)']


def require_compiler(sdk,diagnostics):
    if (sdk.get('status')!=SDK_STATUS or sdk.get('query_bounds')!=[2,32]
            or sdk.get('per_output_FMA_values_and_order_retained') is not True
            or sdk.get('independent_column_vectorization') is not True
            or sdk.get('production_model_acceptance') is not False
            or not all(sdk.get(k) is True for k in ('unmodified_SDK_replay_byte_identical',
                'original_link_inputs_unchanged','original_SDK_unchanged'))
            or sdk.get('only_added_translation_units')!=[sdk.get('source'),sdk.get('dynamic_QK_invariant_source')]
            or len(sdk.get('commands',[]))!=4 or any(x['returncode']!=0 for x in sdk['commands'])
            or diagnostics.count('QWEN_DYNAMIC_QK_INVARIANT_VECTOR matched=28 per_output_FMA_order=original component_loop_unchanged=1 columns=128')!=1):
        raise ValueError('Actual original SDK replay and exactly all28 ordered FMA column vectorizations required')
    for row in sdk['commands'][1:3]:
        if {x for x in row['argv'] if x.startswith('-DQWEN_')}!={'-DQWEN_DYNAMIC_QK_INVARIANT_VECTOR=1'}:
            raise ValueError('Only the explicit invariant QK mode may be enabled')
