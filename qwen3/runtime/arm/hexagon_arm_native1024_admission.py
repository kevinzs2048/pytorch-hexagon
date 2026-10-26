"""Admission for a NEW ARM1024 generation experiment, never prior acceptance.

Require separate actual full-model and actual v4-transport DSP audits. These
proofs admit a test of their combination; they do not prove generate() already
works. All digest values must come from the externally trusted bundle manifest.
"""
from hexagon_native1024_closed_probe import validate_closed_record


def admit_proofs(fixed, fixed_audit, wire, wire_audit, runtime, *, digests):
    validate_closed_record(fixed,fixed_audit,fixed['events'])
    if (fixed_audit.get('result_sha256')!=digests['fixed_result']
            or fixed_audit.get('auditor_sha256')!=digests['fixed_auditor']
            or fixed.get('shared',{}).get('sha256')!=digests['linked']
            or fixed_audit.get('one_load') is not True
            or fixed_audit.get('new_archive_losslessly_reconstructed') is not True):
        raise ValueError('Exact independently audited1024 model linkage required')
    if (wire.get('status')!='native1024_variable_KV_synthetic_DSP_probe_passed_not_model'
            or wire_audit.get('status')!='native1024_variable_KV_synthetic_DSP_independently_audited_not_model'
            or wire_audit.get('result_sha256')!=digests['wire_result']
            or wire_audit.get('auditor_sha256')!=digests['wire_auditor']
            or wire_audit.get('packet_auditor_sha256')!=digests['packet_auditor']
            or wire_audit.get('new_archives_losslessly_reconstructed') is not True):
        raise ValueError('Exact independentv4 transport proof required')
    for row in (wire,wire_audit):
        if (type(row.get('successful_DSP_calls')) is not int or row['successful_DSP_calls']!=10
                or type(row.get('expected_failed_DSP_calls')) is not int or row['expected_failed_DSP_calls']!=5
                or row.get('model_inference') is not False or row.get('full_model_pass') is not False
                or row.get('reduced_RPC_bytes') is not True or row.get('performance_acceptance') is not False):
            raise ValueError('Transport-only10positive/5negative scope required')
    previous=wire.get('session_admission',{})
    if (previous.get('previous_run',{}).get('sha256')!=digests['fixed_result']
            or previous.get('previous_audit',{}).get('sha256')!=digests['fixed_audit']
            or previous.get('clean_shutdown_verified') is not True
            or wire.get('runtime',{}).get('sha256')!=digests['runtime']):
        raise ValueError('Transport proof belongs to different preceding model/runtime')
    if (runtime.get('status')!='native1024_variable_RPC_runtime_built_not_executed'
            or runtime.get('resident_KV_protocol')!='native-generate-pair-capacity1024-v4'
            or runtime.get('resident_KV_capacity')!=1024
            or runtime.get('actual_variable_RPC_bytes') is not True
            or runtime.get('buffer_storage')!='copied'
            or runtime.get('DMA_cpu_sync') is not False or runtime.get('DSP_buffer_cache_ops') is not False):
        raise ValueError('Only the exact previously audited copiedv4 runtime is admitted')
    return dict(capacity=1024,model_generation_accepted=False,
        model_fixed_forward_audited=True,transport_audited=True,CPU_fallback=False,
        admission_scope='test_original_ARM_generate_after_separate_DSP_proofs')
