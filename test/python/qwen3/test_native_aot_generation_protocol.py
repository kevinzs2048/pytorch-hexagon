"""Check one-bin generation, genuine cache fallback and incomplete reports."""
import copy
import pytest
from qwen3.runtime_checks import require_generation_protocol, require_original_guard_configuration


def test_original_guard_configuration():
    settings = dict(bypass_autograd_cache_key=False,
        unsafe_skip_cache_dynamic_shape_guards=False, caching_precompile=False)
    require_original_guard_configuration(settings)
    for key in settings:
        for value in (True, None, 0):
            changed = dict(settings, **{key: value})
            with pytest.raises(ValueError): require_original_guard_configuration(changed)
        missing = dict(settings); del missing[key]
        with pytest.raises(ValueError): require_original_guard_configuration(missing)


def fixture(stages=('hit_rebind',)):
    captures = [dict(graph=dict(sha256=str(i))) for i in range(len(stages))]
    events = [dict(stage='original_native_compiler_on_AOT_' + stage,
        graph=capture['graph'], current_model_rebound=True)
        for stage, capture in zip(stages, captures)]
    native = dict(status='ARM_native_dynamic_compiler_handoff_generate_completed',
        process_exit_code=0, CPU_model_fallback=False, native_generate=True,
        DSP_calls=3, forwards=[{}, {}, {}], frontend_invocations=0,
        machine_code_compilations=0, captures=captures)
    hits = stages.count('hit_rebind')
    counters = dict(total=len(stages), autograd_cache_hit=hits,
        autograd_cache_miss=len(stages)-hits, autograd_cache_saved=len(stages)-hits)
    return native, events, counters


@pytest.mark.parametrize('stages', [('hit_rebind',), ('miss',), ('hit_rebind','hit_rebind'), ('miss','hit_rebind')])
def test_actual_one_or_both_bins(stages):
    native, events, counters = fixture(stages)
    assert require_generation_protocol(native, events, counters)['captures'] == len(stages)


def test_guard_miss_retains_original_capture():
    native, events, counters = fixture(('miss',))
    counters['autograd_cache_guard_miss'] = 1
    assert require_generation_protocol(native, events, counters)['guard_misses'] == 1


def test_original_warm_terminal_status_requires_explicit_mode():
    native, events, counters = fixture()
    native['status'] = 'ARM_native_dynamic_warm_generate_completed'
    with pytest.raises(ValueError): require_generation_protocol(native, events, counters)
    assert require_generation_protocol(native, events, counters,
        compiler_mode='original_warm_catalog')['hits'] == 1
    native['status'] = 'ARM_native_dynamic_compiler_handoff_generate_completed'
    with pytest.raises(ValueError):
        require_generation_protocol(native, events, counters, compiler_mode='original_warm_catalog')
    with pytest.raises(ValueError):
        require_generation_protocol(native, events, counters, compiler_mode='unknown')


@pytest.mark.parametrize('change', ['partial','CPU','build','extra_capture','stale_model','graph','bypass','boolean_counter'])
def test_reject_incomplete_or_unsafe_report(change):
    native, events, counters = copy.deepcopy(fixture())
    if change == 'partial': native['forwards'].pop()
    if change == 'CPU': native['CPU_model_fallback'] = True
    if change == 'build': native['machine_code_compilations'] = 1
    if change == 'extra_capture': events.append(copy.deepcopy(events[0]))
    if change == 'stale_model': events[0]['current_model_rebound'] = False
    if change == 'graph': events[0]['graph'] = dict(sha256='wrong')
    if change == 'bypass': counters['autograd_cache_bypass'] = 1
    if change == 'boolean_counter': counters['autograd_cache_hit'] = True
    with pytest.raises(ValueError): require_generation_protocol(native, events, counters)
