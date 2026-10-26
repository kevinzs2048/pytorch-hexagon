"""Host-side forward-stage observations, not end-to-end generation benchmarks.

The host wait for an invocation includes ARM file I/O, integrity checks and
RPC. It must not be added to the nested ARM RPC or DSP kernel measurements.
Compilation, generation control and AOT output writeback are outside this span.
"""
import time


FORWARD_STAGES = (
    'bind_and_pack_inputs',
    'select_compiled_library',
    'validate_and_store_request',
    'upload_request',
    'verify_remote_request',
    'wait_for_board_invocation',
    'record_invocation_and_download_response',
    'validate_and_store_response',
    'remove_remote_scratch',
    'unpack_response',
)

ARM_FORWARD_STAGES = (
    'validate_frozen_parameters',
    'pack_and_validate_inputs',
    'select_compiled_library',
    'record_request',
    'exchange_with_AP',
    'record_response',
    'release_file_scratch',
    'reconstruct_owned_outputs',
)


def validate_host_timing(record):
    return _validate_timing(record, 'host_forward_stage_timing_v1', FORWARD_STAGES)


def _validate_timing(record, schema, stages):
    if (record.get('schema') != schema
            or record.get('clock') != 'perf_counter_ns'
            or type(record.get('stages_ns')) is not dict
            or set(record['stages_ns']) != set(stages)
            or type(record.get('total_ns')) is not int
            or any(type(value) is not int or value < 0
                   for value in record['stages_ns'].values())
            or record['total_ns'] != sum(record['stages_ns'].values())):
        raise ValueError('Incomplete or inconsistent host forward timing')
    return record


def audit_arm_forward_timings(record):
    """Optional historical field; compiler/AOT writeback/generate excluded."""
    header = record.get('arm_forward_timing')
    if header is None:
        if any('arm_timing' in row for row in record['forwards']):
            raise ValueError('ARM timing lacks an explicit scope')
        return None
    schema = 'arm_local_forward_stage_timing_v1'
    if (header.get('schema') != schema or header.get('includes_compilation_or_generate') is not False
            or header.get('includes_AOT_output_writeback') is not False
            or not record['forwards'] or len(record['forwards']) != record['DSP_calls']):
        raise ValueError('Complete and explicitly scoped ARM callback timing required')
    totals = dict.fromkeys(ARM_FORWARD_STAGES, 0)
    for row in record['forwards']:
        timing = _validate_timing(row.get('arm_timing', {}), schema, ARM_FORWARD_STAGES)
        for stage, elapsed in timing['stages_ns'].items():
            totals[stage] += elapsed
    return dict(measured_forwards=len(record['forwards']), total_ns=sum(totals.values()),
                stage_totals_ns=totals, scope='ARM_compiled_callback_not_end_to_end_generation',
                nested_RPC_and_DSP_timings_not_added=True, performance_acceptance=False)


def audit_host_forward_timings(record):
    """Optional on historical runs; newly declared timing must cover every call."""
    header = record.get('host_forward_timing')
    forwards = record['forwards']
    if header is None:
        if any('host_timing' in row for row in forwards):
            raise ValueError('Forward timing lacks its explicit scope declaration')
        return None
    if (header.get('schema') != 'host_forward_stage_timing_v1'
            or header.get('includes_compilation_or_generate') is not False
            or not forwards or len(forwards) != record['DSP_calls']):
        raise ValueError('Completed and explicitly scoped forward timing required')
    totals = dict.fromkeys(FORWARD_STAGES, 0)
    for row in forwards:
        timing = validate_host_timing(row.get('host_timing', {}))
        for stage, elapsed in timing['stages_ns'].items():
            totals[stage] += elapsed
    return dict(measured_forwards=len(forwards), total_ns=sum(totals.values()),
                stage_totals_ns=totals, includes_compilation_or_generate=False,
                performance_acceptance=False)


class HostForwardTimer:
    """Record one sequential callback; never publish a partial call as complete."""

    stage_order = FORWARD_STAGES
    schema = 'host_forward_stage_timing_v1'

    def __init__(self, *, clock=time.perf_counter_ns):
        self.clock = clock
        self.start = self.last = clock()
        if type(self.start) is not int or self.start < 0:
            raise ValueError('Nonnegative monotonic integer clock required')
        self.stages = {}

    def mark(self, stage):
        index = len(self.stages)
        if index == len(self.stage_order) or stage != self.stage_order[index]:
            raise ValueError('Host timing stage out of order')
        now = self.clock()
        if type(now) is not int or now < self.last:
            raise ValueError('Host monotonic clock moved backwards or is malformed')
        self.stages[stage] = now - self.last
        self.last = now

    def completed(self):
        return _validate_timing(dict(schema=self.schema,
            clock='perf_counter_ns', stages_ns=dict(self.stages), total_ns=self.last-self.start),
            self.schema, self.stage_order)


class ArmForwardTimer(HostForwardTimer):
    stage_order = ARM_FORWARD_STAGES
    schema = 'arm_local_forward_stage_timing_v1'
