"""Reference selection and bounded evidence budget, not generation logic."""
from qwen3_prefill4_corpus import validate_coverage


def integration_plan(reference):
    if (reference.get('status')!='prefill4_natural_CPU_reference_not_DSP'
            or reference.get('capacity')!=1024 or reference.get('CPU_calls')!=56
            or reference.get('original_framework_unchanged') is not True
            or reference.get('observed_unobserved_sequences_equal') is not True
            or len(reference.get('reference',[]))!=56):
        raise ValueError('Complete original four-token CPU reference required')
    coverage=validate_coverage(reference['requests'],reference['eos_token_ids'],sequence_key='CPU_sequences')
    if coverage!=reference['coverage']:raise ValueError('Reference coverage differs')
    cursor=0;shapes=set()
    for index,request in enumerate(reference['requests']):
        begin,end=request['call_begin'],request['call_end']
        if type(begin) is not int or type(end) is not int or begin!=cursor or not begin<end<=56:
            raise ValueError('Contiguous original reference calls required')
        position=0
        for call in reference['reference'][begin:end]:
            n=call['query_tokens'];shapes.add(n)
            expected=min(4,request['prompt_tokens']-position) if position<request['prompt_tokens'] else 1
            if (type(n) is not int or n!=expected or call['request']!=index
                    or call['positions']!=list(range(position,position+n))
                    or len(call['inputs'])!=60 or len(call['outputs'])!=57):
                raise ValueError('Original full-forward trajectory differs')
            position+=n
        cursor=end
    if cursor!=56 or shapes!={1,2,3,4}:raise ValueError('All four complete model shapes required')
    return dict(reference=reference,selected=list(range(4)),expected_calls=reference['reference'],count=56,
        force_length=False,pending_status='ARM_prefill4_native_generate_56_DSP_passed_pending_independent_audit')


def archive_budget(plan):
    from hexagon_prefill1024_layout_candidate import Prefill1024LayoutCandidate
    from hexagon_new_reset_archive import LIMIT
    calls=plan['expected_calls'];total=0
    for call in calls:
        layout=Prefill1024LayoutCandidate(query_tokens=call['query_tokens'])
        # Reset originals are losslessly compressed with a hard upper bound.
        total+=(LIMIT if call['positions'][0]==0 else layout.reuse_packet_bytes)+layout.response_bytes
    maximum=Prefill1024LayoutCandidate(query_tokens=8)
    return dict(retained_packet_upper_bound=total,reserve_bytes=1024**3,
        temporary_packet_headroom=2*maximum.full_packet_bytes+maximum.response_bytes,
        metadata_headroom=64*1024**2,
        minimum_before_run=total+1024**3+2*maximum.full_packet_bytes+maximum.response_bytes+64*1024**2)
