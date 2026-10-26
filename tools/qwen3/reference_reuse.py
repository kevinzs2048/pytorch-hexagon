"""Reuse a fixed independent ARM oracle only for the byte-identical test plan."""
from pathlib import Path

from build import sha


def require_reference_reuse(trace, *, plan_sha, numerical_revision, acceptance, commands,
                            recipe, reference_source_sha, codec_sha):
    if (trace.get('status') != 'independent_native_packet_trace_completed'
            or trace.get('CPU_reference_execution') is not True
            or trace.get('CPU_model_fallback') is not False
            or type(trace.get('DSP_calls')) is not int or trace['DSP_calls'] != 0
            or trace.get('plan_sha256') != plan_sha
            or trace.get('source_sha256') != reference_source_sha
            or trace.get('codec_sha256') != codec_sha
            or acceptance.get('numerical_acceptance') is not True
            or acceptance.get('numerical_revision') != numerical_revision
            or commands.get('numerical_revision') != numerical_revision
            or commands.get('recipe') != recipe):
        raise ValueError('Reuse requires the exact independently accepted plan, numerical model, source and codec')
    reference = acceptance['trace']
    if (sha(reference['path']) != reference['sha256']
            or not Path(reference['board_path']).is_absolute()):
        raise ValueError('Original accepted reference bytes or board path changed')
    return reference
