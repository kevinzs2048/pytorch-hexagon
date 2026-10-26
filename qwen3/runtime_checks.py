"""Shared generation and full-output checks; no experiment launchers or framework imports."""


def compare_integer(baseline,candidate,libraries):
    for run in (baseline,candidate):
        if len(run['forwards'])!=run['DSP_calls']:
            raise ValueError('Complete actual forward list required')
    if ([(r['prompt_ids'],r['generated_ids'],r['terminal_observation']) for r in baseline['requests']]
            !=[(r['prompt_ids'],r['generated_ids'],r['terminal_observation']) for r in candidate['requests']]):
        raise ValueError('Prompt, selected tokens or termination changed')
    if len(baseline['forwards'])!=len(candidate['forwards']):
        raise ValueError('Different actual forward count')
    for old,new in zip(baseline['forwards'],candidate['forwards'],strict=True):
        for key in ('query','request_index','cache_positions','input_sha256','output_sha256'):
            if old[key]!=new[key]:
                raise ValueError('Changed complete model input/output or state: '+key)
        bounds=(1,1) if new['query']==1 else (2,32)
        if type(new['query']) is not int or not 1<=new['query']<=32 or new['DSO_sha256']!=libraries[bounds]:
            raise ValueError('Actual library is not bound to accepted cold candidate')
        for key in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'):
            if old['invoke'][key]!=new['invoke'][key]:
                raise ValueError('Input checksum changed: '+key)
    return True



def validate_parameter_mode(native,mode):
    if mode=='sha256':
        if native.get('parameter_validation',{}).get('mode','sha256')!='sha256':
            raise ValueError('Actual parameter checker differs from requested SHA mode')
        return
    validation=native.get('parameter_validation',{})
    if (mode!='exact-bytes' or validation.get('mode')!=mode
            or validation.get('all_logical_parameter_bytes_checked_each_forward') is not True
            or validation.get('version_only_check') is not False
            or validation.get('initial_compiler_SHA_pins_preserved') is not True
            or type(validation.get('owned_immutable_snapshot_bytes')) is not int
            or validation['owned_immutable_snapshot_bytes']<=0):
        raise ValueError('Actual exact-byte parameter validation evidence required')



def validate_float_policy(native,specialize):
    actual=native.get('generation_options',{}).get('specialize_python_floats',False)
    if type(actual) is not bool or actual is not specialize:
        raise ValueError('Actual native float capture policy differs from requested mode')
    if specialize:
        validation=native.get('parameter_validation',{})
        ledger=validation.get('guarded_python_float_constants',{})
        if (ledger.get('count')!=142 or ledger.get('FP64_bytes_per_constant')!=8
                or ledger.get('owner_identity_and_all_bytes_checked_each_forward') is not True
                or type(validation.get('guarded_python_float_forwards_checked')) is not int
                or validation['guarded_python_float_forwards_checked']!=native['DSP_calls']
                or type(ledger.get('ledger_sha256')) is not str or len(ledger['ledger_sha256'])!=64):
            raise ValueError('Actual original module float byte guards missing')



def require_original_guard_configuration(settings):
    for key in ('bypass_autograd_cache_key', 'unsafe_skip_cache_dynamic_shape_guards',
                'caching_precompile'):
        if settings.get(key) is not False:
            raise ValueError('Original full guards required: ' + key)



def require_generation_protocol(native, events, counters, *, compiler_mode='original_live_handoff'):
    statuses = {'original_live_handoff': 'ARM_native_dynamic_compiler_handoff_generate_completed',
                'original_warm_catalog': 'ARM_native_dynamic_warm_generate_completed'}
    if compiler_mode not in statuses:
        raise ValueError('Explicit original compiler mode required')
    if (native['status'] != statuses[compiler_mode]
            or native['process_exit_code'] != 0
            or native['CPU_model_fallback'] is not False
            or native['native_generate'] is not True
            or type(native['DSP_calls']) is not int or native['DSP_calls'] <= 0
            or len(native['forwards']) != native['DSP_calls']
            or native['frontend_invocations'] != 0
            or native['machine_code_compilations'] != 0):
        raise ValueError('Complete original native generation and warm compiler required')
    captures = native['captures']
    if (not 1 <= len(captures) <= 2 or len(events) != len(captures)
            or [e['graph'] for e in events] != [c['graph'] for c in captures]
            or any(e['stage'] not in ('original_native_compiler_on_AOT_miss',
                'original_native_compiler_on_AOT_hit_rebind') for e in events)
            or any(e.get('current_model_rebound') is not True for e in events
                   if e['stage'].endswith('hit_rebind'))):
        raise ValueError('Every actual original capture needs a current native binding')
    hits = sum(e['stage'].endswith('hit_rebind') for e in events)
    misses = len(events) - hits
    expected = dict(total=len(events), autograd_cache_hit=hits,
        autograd_cache_miss=misses, autograd_cache_saved=misses,
        autograd_cache_bypass=0)
    if (any(type(counters.get(k, 0)) is not int or counters.get(k, 0) != v
            for k, v in expected.items())
            or type(counters.get('autograd_cache_guard_miss', 0)) is not int
            or counters.get('autograd_cache_guard_miss', 0) < 0):
        raise ValueError('Actual framework hit/miss/save counts and no bypass required')
    return dict(captures=len(events), hits=hits, misses=misses,
        guard_misses=counters.get('autograd_cache_guard_miss', 0))
