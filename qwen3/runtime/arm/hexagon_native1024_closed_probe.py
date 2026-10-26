"""Require an independently audited, closed fixed model test before another probe.

This does not prove kernel resources are leak-free. The caller must hold the
device mutex and check current board idleness before accessing the DSP.
"""


def validate_closed_record(run,proof,events):
    for row in (run,proof):
        if (type(row.get('capacity')) is not int or row['capacity']!=1024
                or row.get('continuous_generation') is not False
                or row.get('resident_KV_verified') is not False):
            raise ValueError('Exact fixed1024 model admission required, not generation')
    if (run.get('status')!='native1024_shared_fixed_DSP_complete_NOT_generate'
            or run.get('successful_DSP_calls')!=8 or type(run.get('ARM_process_exit_code')) is not int
            or run.get('ARM_process_exit_code')!=0
            or run.get('all_outputs_bitexact') is not True or run.get('shared_weights_verified') is not True
            or proof.get('status')!='native1024_shared_fixed_DSP_independently_audited_NOT_generate'
            or proof.get('successful_DSP_calls')!=8 or proof.get('output_tensors')!=456
            or proof.get('all_outputs_bitexact') is not True or proof.get('shared_weights_verified') is not True
            or events!=run.get('events')
            or [e.get('stage') for e in events]!=['open','load','ready']+['invoke']*8+['unload','close']):
        raise ValueError('Complete independently audited closed fixed1024 model test required')
    for e in events:
        if e['stage']=='ready':continue
        if (type(e.get('rpc_result')) is not int or e['rpc_result']!=0
                or e.get('loader_result',0)!=0 or e.get('DSP_result',0)!=0 or e.get('output_result',0)!=0):
            raise ValueError('Previous DSP lifecycle failed')
    if [e.get('ordinal') for e in events[3:-2]]!=list(range(1,9)):
        raise ValueError('Previous actual invocation sequence differs')
