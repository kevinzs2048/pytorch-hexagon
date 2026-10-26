"""Summarize measured callback stages; nested timers are never added twice."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def measurements(row):
    invoke, arm = row['invoke'], row['arm_timing']
    stages = arm['stages_ns']
    if (arm['total_ns'] != sum(stages.values()) or any(type(x) is not int or x < 0 for x in stages.values())
            or invoke['rpc_result'] or invoke['DSP_result'] or invoke['output_result']):
        raise ValueError('Complete successful and consistent callback timings required')
    values = {'ARM_callback_seconds': arm['total_ns']/1e9,
        'AP_RPC_seconds': invoke['seconds'], 'DSP_kernel_seconds': invoke['kernel_us']/1e6}
    values.update({'ARM_'+key+'_seconds': value/1e9 for key, value in stages.items()})
    profile = invoke.get('profile')
    if profile is not None:
        if profile.get('schema') != 'complete_buffer_transport_timing_v1':
            raise ValueError('Unsupported transport timing schema')
        for key in ('DSP_rpc_checksum_us', 'DSP_worker_checksum_us', 'DSP_entry_total_us'):
            if type(profile.get(key)) is not int or profile[key] < 0:
                raise ValueError('Complete nonnegative DSP timing required')
            values[key.replace('_us', '_seconds')] = profile[key]/1e6
        for key in ('AP_input_copy_seconds', 'AP_input_checksum_seconds', 'AP_output_poison_seconds',
                    'AP_output_copy_seconds', 'AP_total_seconds'):
            values[key] = profile[key]
        dsp = sum(values[key] for key in ('DSP_rpc_checksum_seconds', 'DSP_worker_checksum_seconds', 'DSP_entry_total_seconds'))
        # HAP microseconds and CLOCK_MONOTONIC durations are independently
        # measured. Permit only rounding-scale differences at clock boundaries.
        if values['DSP_entry_total_seconds']+0.001 < values['DSP_kernel_seconds'] or values['AP_RPC_seconds']+0.001 < dsp:
            raise ValueError('Nested DSP timers exceed their parent interval')
        values['RPC_unattributed_seconds'] = values['AP_RPC_seconds']-dsp
        values['DSP_wrapper_outside_kernel_seconds'] = values['DSP_entry_total_seconds']-values['DSP_kernel_seconds']
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < -0.001 for x in values.values()):
        raise ValueError('Finite nonnegative measurements required')
    return values


def summarize(record):
    if (not record.get('native_generate') or record.get('CPU_model_fallback') is not False
            or not record.get('forwards') or len(record['forwards']) != record['DSP_calls']
            or record.get('process_exit_code') != 0):
        raise ValueError('Completed actual native DSP session required')
    groups = {}
    for row in record['forwards']:
        groups.setdefault(str(row['query']), []).append(measurements(row))
    result = {}
    for query, rows in groups.items():
        if any(set(x) != set(rows[0]) for x in rows):
            raise ValueError('Partial profiling within one query group')
        result[query] = dict(forwards=len(rows),
            median_seconds={key: statistics.median(row[key] for row in rows) for key in rows[0]},
            min_seconds={key: min(row[key] for row in rows) for key in rows[0]},
            max_seconds={key: max(row[key] for row in rows) for key in rows[0]})
    generation = []
    for request in record.get('requests', []):
        timing = request.get('generation_timing')
        if timing is None:
            continue
        ready = timing.get('generated_token_ready_ns', [])
        count = len(request['generated_ids'])
        if (timing.get('schema') != 'native_generate_streamer_timing_v1'
                or timing.get('token_ids_verified') is not True or len(ready) != count
                or timing.get('generated_tokens') != count or not ready
                or any(type(x) is not int or x < 0 for x in ready)
                or ready != sorted(ready)
                or timing['first_generated_token_ready_ns'] != ready[0]
                or timing['inter_token_ns'] != [b-a for a,b in zip(ready, ready[1:])]
                or timing['full_generate_ns'] < timing['native_stream_end_ns']
                or timing['native_stream_end_ns'] < ready[-1]):
            raise ValueError('Native token timings differ from completed request')
        intervals = timing['inter_token_ns']
        generation.append(dict(request_index=request['index'], generated_tokens=count,
            full_generate_seconds=timing['full_generate_ns']/1e9,
            first_token_seconds=ready[0]/1e9,
            median_inter_token_seconds=statistics.median(intervals)/1e9 if intervals else None,
            includes_capture_compiler_handoff_and_AOT_writeback=True,
            includes_online_diagnostic_overhead=True))
    if generation and len(generation) != len(record['requests']):
        raise ValueError('Partial generation timing within a session')
    return dict(scope='ARM_callback_and_nested_AP_DSP_durations',
        end_to_end_generation_measured=bool(generation), callback_includes_AOT_output_writeback=False,
        generation_timings=generation,
        nested_timers_added_to_parent=False,
        RPC_unattributed_is_not_pure_transfer=True, query_groups=result, DSP_calls=record['DSP_calls'],
        performance_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error('Fresh absolute summary output required')
    raw = args.run.read_bytes()
    result = summarize(json.loads(raw))
    result['source_run'] = dict(path=str(args.run), sha256=hashlib.sha256(raw).hexdigest())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Measured DSP forwards:', result['DSP_calls'])


if __name__ == '__main__':
    main()
