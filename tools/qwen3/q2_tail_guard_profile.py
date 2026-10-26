"""Strict bounded decoder for actual all-layer integer guard counters."""
import struct
MAGIC=b'QWQKG01\0'
PACKET_BYTES=1024

def parse(data,query):
    if len(data)!=PACKET_BYTES or data[:8]!=MAGIC or query not in (2,8,32):
        raise ValueError('Exact guard profile packet and runtime query required')
    version,count,overflow=struct.unpack_from('<III',data,8)
    invocation=struct.unpack_from('<Q',data,24)[0]
    expected=0 if query==2 else 28
    if version!=1 or count!=expected or overflow or not invocation or any(data[20:24]+data[32:128]):
        raise ValueError('Every original actual guard call and zero overflow required')
    rows=[]
    for i in range(28):
        raw=data[128+32*i:160+32*i]
        if i>=count:
            if any(raw):raise ValueError('Nonzero unused guard row')
            continue
        ordinal,m,prefix,broadcast,us,cycles=struct.unpack('<IIIIQQ',raw)
        if ordinal!=i or m!=query or prefix not in (0,128,256,384,512,640,1024) or broadcast not in (0,1):
            raise ValueError('Original bounded guard return and exact layer order required')
        rows.append(dict(layer=i,query=m,prefix=prefix,broadcast=broadcast,guard_us=us,guard_cycles=cycles))
    return dict(query=query,invocation=invocation,rows=rows,total_guard_us=sum(r['guard_us']for r in rows),
                optimized_layers=sum(r['prefix']<1024 for r in rows),fallback_layers=sum(r['prefix']==1024 for r in rows))
