"""Independent completed-run audit for bounded library slots, not model quality."""
import math


def audit_library_events(events, forward_slots):
    position=0
    def take(stage,slot=None):
        nonlocal position
        if position>=len(events):raise ValueError('Incomplete library lifecycle')
        row=events[position];position+=1
        if row.get('stage')!=stage:raise ValueError('Unexpected library lifecycle stage')
        if slot is not None and (type(row.get('library_slot')) is not int or row['library_slot']!=slot):
            raise ValueError('Library lifecycle ownership differs')
        if stage!='ready':
            if (type(row.get('rpc_result')) is not int or row['rpc_result']!=0
                    or type(row.get('seconds')) not in (int,float)
                    or not math.isfinite(row['seconds']) or row['seconds']<0):
                raise ValueError('Failed or malformed library operation')
        if stage in ('load','entry_load','unload') and (
                type(row.get('loader_result')) is not int or row['loader_result']!=0):
            raise ValueError('Library loader did not complete successfully')
        return row
    take('open');take('load');ready=take('ready')
    if (ready.get('library_switching') is not True or ready.get('dynamic_sizes') is not True
            or type(ready.get('ARM_pid')) is not int or ready['ARM_pid']<=0
            or any(type(ready.get(key)) is not int or not 0<ready[key]<=256*1024**2
                   for key in ('input_bytes','output_bytes'))):
        raise ValueError('Explicit bounded library-switching process required')
    loaded=1;current=0;actual=[]
    while position<len(events) and events[position].get('stage')!='unload':
        stage=events[position].get('stage')
        if stage=='entry_open':
            if loaded>=4:raise ValueError('Library slot capacity exceeded')
            take('entry_open',loaded);take('entry_load',loaded)
            row=take('select',loaded)
            if row.get('cached') is not False:raise ValueError('New library misreported as reused')
            current=loaded;loaded+=1
        elif stage=='select':
            row=take('select')
            slot=row.get('library_slot')
            if type(slot) is not int or not 0<=slot<loaded or row.get('cached') is not True:
                raise ValueError('Selection does not name an existing loaded library')
            current=slot
        elif stage=='invoke':
            row=take('invoke',current)
            if (type(row.get('ordinal')) is not int or row['ordinal']!=len(actual)+1
                    or any(type(row.get(key)) is not int or row[key]!=0 for key in ('DSP_result','output_result'))
                    or type(row.get('kernel_us')) is not int or row['kernel_us']<0
                    or any(type(row.get(key)) is not int or not 0<row[key]<=ready[key]
                           for key in ('input_bytes','output_bytes'))):
                raise ValueError('Invalid library invocation result or bounds')
            hashes=[row.get(key) for key in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')]
            if any(type(value) is not int or not 0<=value<=0xffffffff for value in hashes) or len(set(hashes))!=1:
                raise ValueError('Library invocation input transport differs')
            actual.append(current)
        else:raise ValueError('Unexpected operation while libraries are live')
    for index in reversed(range(loaded)):
        take('unload',index);take('close',index)
    if (position!=len(events) or not actual
            or any(type(slot) is not int for slot in forward_slots) or actual!=forward_slots):
        raise ValueError('Library invocation/cleanup coverage differs')
    return dict(one_AP_process=True,loaded_libraries=loaded,DSP_calls=len(actual),
                all_loaded_owners_closed=True,physical_weight_sharing_verified=False)
