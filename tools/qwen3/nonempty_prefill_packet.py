"""Derive a bounded prefill request while retaining actual prior KV bits and positions."""
import struct

def resize_nonempty_prefill_packet(contract,payload,query):
    import numpy as np
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for,pack_request
    if (type(query)is not int or not 2<=query<=32 or contract['variables']!={'query':[2,32]}
            or payload[:16]!=b'HXDY\1\0\0\0'+struct.pack('<q',6)):
        raise ValueError('Original q6 packet and unchanged runtime query2..32 domain required')
    layout=layout_for(contract,{'query':6});rows=layout['inputs']
    if len(payload)!=16+layout['input_bytes']or len(rows)!=60:
        raise ValueError('Original complete60-tensor packet required')
    pieces=[payload[16+r['offset']:16+r['offset']+r['bytes']]for r in rows]
    for i,shape in ((0,[1,6]),(1,[6]),(2,[1,6])):
        if rows[i]['dtype']!='int64'or rows[i]['shape']!=shape:
            raise ValueError('Exact original token and two position tensor layouts required')
    tokens=np.frombuffer(pieces[0],dtype='<i8')
    positions=np.frombuffer(pieces[1],dtype='<i8')
    if positions.tolist()!=list(range(8,14))or pieces[2]!=pieces[1]or np.any(tokens<0):
        raise ValueError('Actual contiguous nonempty positions8..13 required')
    start=8
    if start+query>1024:raise ValueError('No request may exceed original KV capacity')
    if rows[5]['dtype']!='float32'or rows[5]['shape']!=[1,1,6,1024]:
        raise ValueError('Original full causal mask layout required')
    mask=np.frombuffer(pieces[5],dtype='<f4').reshape(6,1024);blocked=mask[0,-1]
    expected=np.where(np.arange(1024)[None,:]<=positions[:,None],np.float32(0),blocked).astype('<f4')
    if not np.isfinite(blocked)or blocked>=0 or not np.array_equal(mask.view('<u4'),expected.view('<u4')):
        raise ValueError('Original complete causal mask bits must match the retained positions')
    nonempty=False
    for i in (3,4,*range(6,60)):
        if rows[i]['dtype']!='float32'or rows[i]['shape']!=[1,8,1024,128]or rows[i]['strides']!=[1048576,131072,128,1]:
            raise ValueError('All56 original canonical full KV tensors required')
        words=np.frombuffer(pieces[i],dtype='<u4').reshape(1,8,1024,128)
        if words[:,:,start:,:].any():raise ValueError('No future or stale KV beyond original occupied prefix allowed')
        nonempty=nonempty or bool(words[:,:,:start,:].any())
    if not nonempty:raise ValueError('An actual nonempty retained KV prefix is required')
    if pack_request(contract,{'query':6},pieces)[0]!=payload:
        raise ValueError('Complete original canonical input packet and padding required')
    pieces[0]=np.resize(tokens,query).astype('<i8').tobytes()
    pieces[1]=pieces[2]=np.arange(start,start+query,dtype='<i8').tobytes()
    pieces[5]=np.where(np.arange(1024)[None,:]<=np.arange(start,start+query)[:,None],np.float32(0),blocked).astype('<f4').tobytes()
    return pack_request(contract,{'query':query},pieces)[0]
