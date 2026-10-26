"""Bounded operator cases; runtime eligibility is checked from actual bytes."""
import struct

BASE_CASES=(
    (8,1024,0,'zero',0),(8,1024,0,'sparse8',128),
    (8,1024,0,'sparse129',256),(32,1024,0,'sparse513',640),
    (2,1024,0,'sparse8',128),(7,1024,0,'sparse8',128),
    (8,1023,0,'sparse8',1024),(9,129,0,'cancellation',1024),
    (16,1024,1,'sparse8',1024),(8,1024,0,'negative_keys',1024),
    (8,1024,0,'seed_tail',384),(8,1024,0,'negative_seed',1024),
    (8,1024,0,'infinity',128),(8,1024,0,'qnan',0),
    (8,1024,0,'snan',0),(31,1024,0,'subnormal',128))
CASES=tuple((*row,rounding)for rounding in range(4)for row in BASE_CASES)
INPUT_CAPACITY=128+4*(16*128*1024+16*32*128+16*32*1024)
OUTPUT_CAPACITY=128+4*16*32*1024
FP_FLAGS=0x3e
FP_ROUND=3<<22


def make_case(np,q,n,flag,pattern,extent,rounding):
    rng=np.random.default_rng(20261014+q+n);rows=1 if flag else q
    keys=(rng.random((16,128,n),dtype=np.float32)*2-1).astype(np.float32)
    query=(rng.random((16,rows,128),dtype=np.float32)*2-1).astype(np.float32)
    initial=np.zeros((16,q,n),np.float32)
    prefix={'zero':0,'sparse8':8,'sparse129':129,'sparse513':513,'qnan':0,'snan':0}.get(pattern,8)
    keys[:,:,min(prefix,n):]=0
    if pattern=='cancellation':
        query[:]=np.tile(np.array([16777216.,1.,-16777216.,1.],np.float32),32)
        keys[:,:,:min(8,n)]=1
    elif pattern=='negative_keys':keys[:,:,8:]=-0.0
    elif pattern=='seed_tail':initial.view(np.uint32)[-1,-1,300]=0x80000000
    elif pattern=='negative_seed':initial.view(np.uint32)[-1,-1,800]=0x80000000
    elif pattern in ('infinity','qnan','snan'):
        query.view(np.uint32).fill({'infinity':0x7f800000,'qnan':0x7fc12345,'snan':0x7f812345}[pattern])
    elif pattern=='subnormal':query.view(np.uint32).fill(1)
    actual=1024
    if n==1024 and not flag:
        nonzero=np.any(keys.view(np.uint32)!=0,axis=(0,1))|np.any(initial.view(np.uint32)!=0,axis=(0,1))
        columns=np.flatnonzero(nonzero)
        prefix=0 if not columns.size else ((int(columns[-1])//128)+1)*128
        actual=1024 if prefix>=768 else prefix
    if actual!=extent:raise ValueError('Case metadata disagrees with the actual complete positive-zero proof')
    payload=b'Q2TAIL01'+struct.pack('<IIII',q,n,flag,rounding)+bytes(104)+keys.tobytes()+query.tobytes()+initial.tobytes()
    return keys,query,initial,payload


def parse_output(data,q,n,flag,rounding,extent,kind):
    if kind not in ('baseline','tail') or len(data)!=128+4*16*q*n:
        raise ValueError('Complete actual QK output and known library required')
    elapsed,outq,outn,outflag,outround=struct.unpack_from('<QIIII',data)
    cycles=struct.unpack_from('<Q',data,24)[0]
    before,after,restored,actual_extent,original,flags=struct.unpack_from('<IIIIII',data,32)
    guard_us,calls,internal_extent=struct.unpack_from('<QQq',data,56)
    expected_calls=int(kind=='tail' and 8<=q<=32)
    if ((outq,outn,outflag,outround)!=(q,n,flag,rounding) or not elapsed or not cycles
            or before!=((original&~(FP_FLAGS|FP_ROUND))|(rounding<<22))
            or after&FP_ROUND!=rounding<<22 or restored!=original or flags!=after&FP_FLAGS
            or actual_extent!=extent or calls!=expected_calls
            or (calls and internal_extent!=extent) or any(data[80:128])):
        raise ValueError('Actual rounding, FP flags, restored state, guard extent or invocation count differs')
    return dict(kernel_us=elapsed,cycles=cycles,before_USR=before,after_USR=after,
        restored_USR=restored,original_USR=original,FP_flags=flags,guard_extent=extent,
        internal_guard_calls=calls,internal_guard_extent=internal_extent,
        eligibility=(kind=='tail' and q>=8 and n==1024 and extent<1024 and not flag),guard_query_us=guard_us)
