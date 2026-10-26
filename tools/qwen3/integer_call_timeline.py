"""Parse bounded raw INT8 call timestamps; gaps are not yet named operators."""
import statistics
import struct

MAGIC=b'QWTIM01\0'
PACKET_BYTES=128+256*64
ATTENTION_MAGIC=b'QWTIM02\0'
ATTENTION_PACKET_BYTES=PACKET_BYTES+28*64


def parse_timeline(data,query):
    return _parse_core(data,query,MAGIC,1,0)


def _parse_core(data,query,magic,expected_version,phase_count):
    if type(query) is not int or not 2<=query<=32:
        raise ValueError('Actual bounded prefill query required')
    if len(data)!=PACKET_BYTES+phase_count*64 or data[:8]!=magic:
        raise ValueError('Invalid integer timeline packet')
    version,slots,overflow,invocations,row_bytes,address,count,reserved=struct.unpack_from('<IIQQIIII',data,8)
    begin,end,begin_cycles,end_cycles=struct.unpack_from('<QQQQ',data,48)
    if ((version,slots,row_bytes,count)!=(expected_version,256,64,197) or overflow or not invocations
            or not address or reserved!=phase_count or any(data[80:128])
            or not 0<begin<end or not 0<begin_cycles<end_cycles):
        raise ValueError('Exactly197 complete bounded calls and a closed model interval required')
    rows=[];previous=begin;previous_cycles=begin_cycles
    for index in range(256):
        offset=128+64*index;raw=data[offset:offset+64]
        if index>=count:
            if any(raw):raise ValueError('Nonempty unused timeline row')
            continue
        ordinal,site,m,k,n,enter,leave,enter_cycles,leave_cycles=struct.unpack('<IIqqqQQQQ',raw)
        if (ordinal!=index or not site or not 1<=m<=query or not 1<=k<=131071
                or not 1<=n<=151936 or not previous<=enter<=leave<=end
                or not previous_cycles<=enter_cycles<=leave_cycles<=end_cycles):
            raise ValueError('Ordered valid call geometry and complete nonoverlapping timestamps required')
        rows.append(dict(ordinal=ordinal,return_address=site,relative_to_model_entry=site-address,
            m=m,k=k,n=n,enter_us=enter,leave_us=leave,enter_cycles=enter_cycles,
            leave_cycles=leave_cycles,gap_before_us=enter-previous,
            callee_us=leave-enter,gap_before_cycles=enter_cycles-previous_cycles,
            callee_cycles=leave_cycles-enter_cycles))
        previous=leave;previous_cycles=leave_cycles
    if not any(r['m']==query for r in rows):raise ValueError('Actual selected prefill extent was not observed')
    total=end-begin;callee=sum(r['callee_us']for r in rows)
    gaps=sum(r['gap_before_us']for r in rows);tail=end-previous
    if callee+gaps+tail!=total:raise ValueError('Timeline accounting failed')
    shape=lambda r:(r['m'],r['k'],r['n'])
    pattern=[shape(r)for r in rows[:7]]
    repeated=all(shape(rows[7*layer+slot])==pattern[slot]for layer in range(28)for slot in range(7))
    groups=[]
    if repeated:
        for slot in range(7):
            group=[rows[7*layer+slot]for layer in range(28)]
            groups.append(dict(slot=slot,shape=list(pattern[slot]),calls=28,
                gap_before_us=sum(r['gap_before_us']for r in group),
                callee_us=sum(r['callee_us']for r in group),
                median_gap_before_us=statistics.median(r['gap_before_us']for r in group)))
    return dict(invocations=invocations,entry_address=address,overflow=overflow,query=query,
        begin_us=begin,end_us=end,begin_cycles=begin_cycles,end_cycles=end_cycles,rows=rows,
        total_us=total,callee_us=callee,gaps_us=gaps,tail_us=tail,
        complete_interval_accounted=True,repeated_28_by_7_shape_pattern=repeated,
        repeated_slots=groups,scope='timestamps_between_original_INT8_calls_NOT_named_operator_costs')


def parse_attention_timeline(data,query):
    result=_parse_core(data,query,ATTENTION_MAGIC,2,28)
    if not result['repeated_28_by_7_shape_pattern']:
        raise ValueError('The observed28-layer seven-call geometry is required')
    phases=[]
    for layer in range(28):
        fields=struct.unpack_from('<IIQQQQQQQ',data,PACKET_BYTES+layer*64)
        actual,calls,first,last,first_cycles,last_cycles,expected,m,reserved=fields
        left,right=result['rows'][7*layer+2:7*layer+4]
        if (actual!=layer or calls!=expected or expected!=query*16*1024 or m!=query or reserved
                or left['m']!=query or right['m']!=query
                or not left['leave_us']<=first<=last<=right['enter_us']
                or not left['leave_cycles']<=first_cycles<=last_cycles<=right['enter_cycles']):
            raise ValueError('Every original softmax call and both boundaries must fit its measured layer interval')
        before=first-left['leave_us'];during=last-first;after=right['enter_us']-last
        if before+during+after!=right['gap_before_us']:
            raise ValueError('The three observed phase intervals do not close')
        phases.append(dict(layer=layer,calls=calls,query=m,
            first_enter_us=first,last_leave_us=last,first_enter_cycles=first_cycles,last_leave_cycles=last_cycles,
            before_first_exp_us=before,first_to_last_exp_us=during,after_last_exp_us=after,
            complete_gap_us=right['gap_before_us']))
    result['attention_phases']=phases
    result['attention_phase_totals']={key:sum(row[key]for row in phases)for key in
        ('before_first_exp_us','first_to_last_exp_us','after_last_exp_us','complete_gap_us')}
    result['scope']='observed_intervals_before_during_after_original_softmax_calls_NOT_individual_operator_costs'
    return result


def revisit_timeline_packets(board, build, forward, query=1):
    import json
    from pathlib import Path
    from build import entry
    from revisit_key_tile_external_profile import fnv
    attention=build.get('attention_phase_timeline')is True
    magic=ATTENTION_MAGIC if attention else MAGIC
    packet_bytes=ATTENTION_PACKET_BYTES if attention else PACKET_BYTES
    parser=parse_attention_timeline if attention else parse_timeline
    run = json.loads((board/'result.json').read_text())
    events = [json.loads(line) for line in (board/'stream.stdout.txt').read_text().splitlines()]
    if (run['status'] != 'diagnostic_model_memory_complete_packet_bitexact'
            or run['process_exit_code'] != 0 or run['query'] != query
            or run['model_calls'] != 4 or run['diagnostic_query_calls'] != 2 or run['DSP_calls'] != 6
            or len(run['rows']) != 4 or len(run['profiles']) != 2
            or run['stream_events'] != events or events[-1]['stage'] != 'close'
            or events[-1]['rpc_result'] != 0
            or run['input_sha256'] != forward['input_sha256']
            or run['expected_output_sha256'] != forward['output_sha256']
            or run['baseline_library_sha256'] != forward['DSO_sha256']
            or run['profile_library_sha256'] != build['library']['sha256']
            or run['model_object_unchanged'] is not True
            or run['model_graph_changed'] is not False or run['power_votes_changed'] is not False):
        raise ValueError('Exact original model ABBA4, two raw queries and normal close required')
    dynamic = True
    if (type(query) is not int or not 2<=query<=32
            or build.get('integer_call_timeline') is not True
            or (not attention and build.get('timeline_only_wrapped_symbol')!='_mlir_ciface___qwen_i8_multi_matmul')
            or (attention and (build.get('timeline_only_wrapped_symbol')is not None
                or build.get('timeline_wrapped_symbols')!=['_mlir_ciface___qwen_i8_multi_matmul',
                    '_mlir_ciface___hexagon_softmax_exp_compat']))
            or build.get('external_profile_symbols')!={}
            or build.get('external_profile_kinds')!=[] or build.get('required_profile_kinds')!=[]
            or build.get('query_bounds')!=[2,32]
            or build.get('input_bytes_are_maximum_capacity') is not True):
        raise ValueError('Only the original dynamic INT8 callee may be timestamped')
    invokes = [e for e in events if e['stage'] == 'invoke']
    if (len(invokes) != 6 or [e['ordinal'] for e in invokes] != list(range(1, 7))
            or any(e[k] != 0 for e in invokes for k in ('rpc_result', 'DSP_result', 'output_result'))):
        raise ValueError('Six unique consecutive normally completed RPC invocations required')
    used = []
    for i, row in enumerate(run['rows']):
        e = row['actual_invoke']
        if (row['index'] != i or row['library'] != ('baseline', 'profile', 'profile', 'baseline')[i]
                or row['all_logits_and_56_KV_byteexact'] is not True
                or row['output_sha256'] != forward['output_sha256'] or e not in invokes
                or e['library_slot'] != (0, 1, 1, 0)[i]
                or e['input_bytes'] != forward['invoke']['input_bytes'] or e['output_bytes'] != 235488768
                or (dynamic and not 16 < e['input_bytes'] <= 235012880)
                or (not dynamic and e['input_bytes'] != 234885520)
                or any(e[k] != forward['invoke']['AP_input_fnv1a32'] for k in
                    ('AP_input_fnv1a32', 'DSP_input_fnv1a32', 'RPC_input_fnv1a32'))):
            raise ValueError('Every complete original model packet, FNV and actual library slot must match')
        used.append(e)
    profiles, raws = [], []
    query_fnv = fnv(magic + bytes(8))
    for i, row in enumerate(run['profiles']):
        path = board/('profile-' + str(i+1) + '.bin')
        # Profile calls follow model indices one and two, respectively.
        actual = parser(path.read_bytes(),query)
        e = row['actual_invoke']
        if (row['index'] != i+1 or actual != row['profile'] or actual['invocations'] != i+1
                or len(actual['rows']) != 197 or e not in invokes
                or e['library_slot'] != 1 or e['input_bytes'] != 16 or e['output_bytes'] != packet_bytes
                or any(e[k] != query_fnv for k in
                    ('AP_input_fnv1a32', 'DSP_input_fnv1a32', 'RPC_input_fnv1a32'))):
            raise ValueError('Raw external counters must bind all 197 original INT8 calls and their actual queries')
        profiles.append(actual); raws.append(entry(path)); used.append(e)
    if sorted(e['ordinal'] for e in used) != list(range(1, 7)):
        raise ValueError('No repeated or omitted model/query evidence allowed')
    signature=lambda p:[(r['relative_to_model_entry'],r['m'],r['k'],r['n'])for r in p['rows']]
    if signature(profiles[0])!=signature(profiles[1]):
        raise ValueError('Both unchanged full-model passes must use identical call sites and shapes')
    for i,profile in enumerate(profiles):
        if profile['total_us']>run['rows'][i+1]['actual_invoke']['kernel_us']+10:
            raise ValueError('The complete call timeline exceeds its actual enclosing kernel interval')
    accounting={key:statistics.mean(p[key]for p in profiles)
        for key in ('total_us','callee_us','gaps_us','tail_us')}
    slots=[]
    if all(p['repeated_28_by_7_shape_pattern']for p in profiles):
        for i in range(7):
            rows=[p['repeated_slots'][i]for p in profiles]
            if rows[0]['shape']!=rows[1]['shape']:raise ValueError('Repeated shape pattern changed')
            slots.append(dict(slot=i,shape=rows[0]['shape'],calls=28,
                gap_before_us=statistics.mean(r['gap_before_us']for r in rows),
                callee_us=statistics.mean(r['callee_us']for r in rows)))
    timing = {}
    for field in ('kernel_us', 'seconds'):
        baseline = statistics.mean(r['actual_invoke'][field] for r in run['rows'] if r['library'] == 'baseline')
        diagnostic = statistics.mean(r['actual_invoke'][field] for r in run['rows'] if r['library'] == 'profile')
        timing[field] = dict(baseline=baseline, diagnostic=diagnostic, diagnostic_overhead_ratio=diagnostic/baseline)
    phase_totals=({key:statistics.mean(p['attention_phase_totals'][key]for p in profiles)
        for key in profiles[0]['attention_phase_totals']}if attention else None)
    return dict(timeline_accounting=accounting,repeated_slots=slots,attention_phase_totals=phase_totals,
        raw_profiles=raws, raw_stdout=entry(board/'stream.stdout.txt'),
        result=entry(board/'result.json'), paired_instrumentation_timing=timing,
        complete_output_hash=forward['output_sha256'], complete_diagnostic_output_file_retained=False,
        model_DSP_calls=4, diagnostic_query_DSP_calls=2, new_DSP_calls=0,
        instrumented_times_are_not_production_gain=True, model_inference_acceptance=False,
        scope='measured_intervals_between_original_INT8_call_sites_NOT_named_operator_costs')

