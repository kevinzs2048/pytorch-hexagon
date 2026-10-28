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
