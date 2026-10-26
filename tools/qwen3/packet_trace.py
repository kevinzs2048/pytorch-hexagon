"""Canonical tensor packet diagnostics; no model arithmetic or graph selection."""
import hashlib
import struct
from pathlib import Path
import numpy as np

MAGIC=b'HXDY\x01\x00\x00\x00'
TOLERANCE=1e-4


def array(value):
    if hasattr(value,'detach'):
        value=value.detach().contiguous().numpy()
    value=np.asarray(value)
    if value.dtype not in (np.dtype('<f4'),np.dtype('<i8')):
        raise ValueError('Actual FP32/int64 packet tensors required')
    return np.ascontiguousarray(value)


def describe_packet(layout,side,values,*,query=None,save=None):
    if side not in ('inputs','outputs') or len(values)!=len(layout[side]):
        raise ValueError('Exact packet tensor list required')
    header=MAGIC+struct.pack('<q',query) if side=='inputs' else b''
    expected=layout[side[:-1]+'_bytes']
    writer=Path(save).open('xb') if save is not None else None
    digest=hashlib.sha256();fields=[];offset=0
    def emit(data):
        digest.update(data)
        if writer is not None:writer.write(data)
    try:
        emit(header)
        for slot,value in zip(layout[side],values,strict=True):
            value=array(value)
            if list(value.shape)!=slot['shape'] or str(value.dtype)!=slot['dtype']:
                raise ValueError('Reference tensor disagrees with actual ABI')
            raw=memoryview(value).cast('B')
            if len(raw)!=slot['bytes'] or slot['offset']<offset:
                raise ValueError('Reference packet byte layout differs')
            emit(bytes(slot['offset']-offset));emit(raw)
            fields.append(dict(index=slot['index'],shape=slot['shape'],dtype=slot['dtype'],
                bytes=slot['bytes'],sha256=hashlib.sha256(raw).hexdigest()))
            offset=slot['offset']+slot['bytes']
        emit(bytes(expected-offset))
    finally:
        if writer is not None:writer.close()
    return dict(sha256=digest.hexdigest(),bytes=len(header)+expected,fields=fields)


def describe_bytes(layout,side,payload,*,query=None):
    head=16 if side=='inputs' else 0
    if len(payload)!=head+layout[side[:-1]+'_bytes']:
        raise ValueError('Actual packet length differs')
    if head and payload[:head]!=MAGIC+struct.pack('<q',query):
        raise ValueError('Actual packet header differs')
    values=[np.frombuffer(payload,dtype='<f4' if s['dtype']=='float32' else '<i8',
        offset=head+s['offset'],count=s['bytes']//(4 if s['dtype']=='float32' else 8)).reshape(s['shape']) for s in layout[side]]
    description=describe_packet(layout,side,values,query=query)
    if description['sha256']!=hashlib.sha256(payload).hexdigest():
        raise ValueError('Actual packet padding is not canonical zero bytes')
    return description,values


def validate_qwen_layout(layout,query):
    ins,outs=layout['inputs'],layout['outputs']
    if len(ins)!=60 or len(outs)!=57:
        raise ValueError('Current Qwen tensor ABI requires 60 inputs and 57 outputs')
    shapes=[[1,query],[query],[1,query]]
    for s,shape in zip(ins[:3],shapes,strict=True):
        if s['dtype']!='int64' or s['shape']!=shape:raise ValueError('Native token/position ABI differs')
    if ins[5]['dtype']!='float32' or ins[5]['shape']!=[1,1,query,1024]:raise ValueError('Native mask ABI differs')
    for s in ins[3:5]+ins[6:]+outs[:56]:
        if s['dtype']!='float32' or s['shape']!=[1,8,1024,128]:raise ValueError('Native full KV ABI differs')
    if outs[-1]['dtype']!='float32' or outs[-1]['shape']!=[1,1,151936]:raise ValueError('Full last-token logits required')


def actual_native_inputs(kwargs,cache):
    kv=[v for layer in cache.layers for v in (layer.keys,layer.values)]
    mask=kwargs['attention_mask']
    if isinstance(mask,dict):mask=mask['full_attention']
    if mask.ndim!=4:raise ValueError('Actual native 4D mask required; do not reconstruct or infer it')
    if kwargs['past_key_values'] is not cache:raise ValueError('Native reference cache identity differs')
    return [kwargs['input_ids'],kwargs['cache_position'],kwargs['position_ids']]+kv[:2]+[mask]+kv[2:]


def actual_native_outputs(out,cache):
    if out.past_key_values is not cache:raise ValueError('Native output cache ownership differs')
    return [v for layer in cache.layers for v in (layer.keys,layer.values)]+[out.logits[:,-1:,:]]
