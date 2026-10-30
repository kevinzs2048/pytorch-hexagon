"""Sequential ARM startup observations; nested framework spans are not summed."""
from hexagon_runtime.hexagon_host_timing import HostForwardTimer


class BootstrapTimer(HostForwardTimer):
    schema='ARM_native_bootstrap_stage_timing_v1'
    stage_order=('import_and_configure','load_and_verify_native_model',
                 'prepare_cache_and_decompositions','stage_runtime_and_constants')


class CaptureTimer(HostForwardTimer):
    schema='ARM_native_capture_stage_timing_v1'
    stage_order=('AOT_prepare_graph','bind_immutable_inputs','identify_live_graph',
                 'publish_live_capture','wait_for_compiler_host',
                 'verify_and_stage_reply','record_and_bind_callback')
