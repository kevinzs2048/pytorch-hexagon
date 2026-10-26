"""Compare accepted same-input runs; report callback gains with their scope."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_performance import summarize


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def accepted(directory):
    acceptance = json.loads((directory/'acceptance.json').read_text())
    native = json.loads((directory/'native-result.json').read_text())
    observer = json.loads((directory/'online-audit.json').read_text())
    if (acceptance.get('numerical_acceptance') is not True
            or observer.get('numerical_acceptance') is not True
            or sha(directory/'online-audit.json') != acceptance['online_audit']['sha256']
            or sha(directory/'native-result.json') != observer['run']['sha256']
            or native['DSP_calls'] != acceptance['DSP_calls']
            or len(observer['forwards']) != acceptance['DSP_calls']
            or json.loads((directory/'run-exit.json').read_text()) != {'board_returncode':0,'worker_returncode':0}
            or not all(x['independent_input_bitexact'] and x['independent_output_bitexact'] for x in observer['forwards'])):
        raise ValueError('Completed bitexact numerical acceptance required before performance comparison')
    chain = acceptance['compile_handoff']
    if (chain['machine_code_compilations'] or chain['frontend_invocations']
            or chain['actual_requests_and_results_rehashed'] is not True
            or chain['complete_rebuilt_artifacts_rehashed'] is not True):
        raise ValueError('Verified warm compilation chain required')
    return acceptance, native, summarize(native)


def compare(baseline, candidate):
    if (baseline['requests'] and
            [(x['prompt_ids'],x['generated_ids']) for x in baseline['requests']] !=
            [(x['prompt_ids'],x['generated_ids']) for x in candidate['requests']]):
        raise ValueError('Prompt or actual selected tokens changed')
    left, right = baseline['forwards'], candidate['forwards']
    if len(left) != len(right):
        raise ValueError('Different actual forward count')
    for old, new in zip(left, right):
        for key in ('query','request_index','cache_positions','input_sha256','output_sha256','DSO_sha256'):
            if old[key] != new[key]:
                raise ValueError('Changed input/output/model binary: '+key)
        for key in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'):
            if old['invoke'][key] != new['invoke'][key]:
                raise ValueError('Input checksum changed: '+key)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error('Fresh absolute comparison output required')
    ba, bn, bs = accepted(args.baseline)
    ca, cn, cs = accepted(args.candidate)
    if ba['numerical_revision'] != ca['numerical_revision']:
        raise ValueError('Numerical compiler revision changed')
    compare(bn, cn)
    groups = {}
    for query in bs['query_groups']:
        old = bs['query_groups'][query]['median_seconds']
        new = cs['query_groups'][query]['median_seconds']
        groups[query] = dict(baseline_forwards=bs['query_groups'][query]['forwards'],
            baseline_median_seconds=old, candidate_median_seconds=new,
            ARM_callback_reduction_percent=100*(1-new['ARM_callback_seconds']/old['ARM_callback_seconds']),
            AP_RPC_reduction_percent=100*(1-new['AP_RPC_seconds']/old['AP_RPC_seconds']))
    result = dict(status='same_input_numerics_and_exact_checksums_verified_performance_comparison',
        full_inputs_outputs_model_DSOs_and_checksums_identical=True,
        numerical_revision=ba['numerical_revision'], DSP_calls_per_run=bn['DSP_calls'],
        scope='measured_native_callback_including_online_diagnostics',
        end_to_end_improvement_claimed=False, query_groups=groups,
        candidate_generation_timings=cs['generation_timings'],
        baseline_acceptance=dict(path=str(args.baseline/'acceptance.json'),sha256=sha(args.baseline/'acceptance.json')),
        candidate_acceptance=dict(path=str(args.candidate/'acceptance.json'),sha256=sha(args.candidate/'acceptance.json')))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v['ARM_callback_reduction_percent'] for k,v in groups.items()}))


if __name__ == '__main__':
    main()
