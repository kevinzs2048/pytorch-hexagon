"""Structural identity for a bounded native ATen graph, NOT serialization.

PyTorch export remains the graph format. This digest checks ordered dataflow,
literal arguments, tensor metadata and actual immutable constants after native
weight binding. Node spelling is not semantic; output list/tuple is normalized
only at the flat AOT return boundary. Unsupported metadata is rejected.
"""
import hashlib
import json
import struct


def graph_identity(gm):
    import torch
    from hexagon_dynamo_parameters import digest,spec
    nodes=list(gm.graph.nodes);indices={n:i for i,n in enumerate(nodes)}
    def value(v):
        if isinstance(v,torch.fx.Node):return dict(node=indices[v])
        if v is None or type(v) in (bool,int,str):return dict(type=type(v).__name__,value=v)
        if type(v) is float:return dict(float64_bits=struct.pack('!d',v).hex())
        if isinstance(v,(tuple,list)):
            return dict(type='tuple' if isinstance(v,tuple) else 'list',items=[value(x) for x in v])
        if isinstance(v,dict):
            if any(type(k) is not str for k in v):raise ValueError('String keyword keys required')
            return dict(type='dict',items={k:value(v[k]) for k in sorted(v)})
        if isinstance(v,(torch.dtype,torch.device,torch.layout,torch.memory_format)):
            return dict(type=type(v).__name__,value=str(v))
        raise ValueError('Unsupported native graph literal: '+type(v).__name__)
    def meta(v):
        if isinstance(v,torch.Tensor):return dict(tensor=spec(v))
        if isinstance(v,(tuple,list)):return dict(sequence=[meta(x) for x in v])
        return value(v)
    rows=[]
    for node in nodes:
        row=dict(op=node.op)
        if node.op=='placeholder':row['tensor_metadata']=meta(node.meta['val'])
        elif node.op=='get_attr':
            tensor=gm
            for part in node.target.split('.'):tensor=getattr(tensor,part)
            if not isinstance(tensor,torch.Tensor):raise ValueError('Only tensor graph constants supported')
            from torch._subclasses.fake_tensor import is_fake
            from torch.utils._python_dispatch import _disable_current_modes
            if is_fake(tensor):raise ValueError('Actual immutable constant bytes required')
            with _disable_current_modes():
                row.update(constant=dict(spec=spec(tensor),sha256=digest(tensor)))
        elif node.op=='call_function':
            row.update(target=str(node.target),args=value(node.args),kwargs=value(node.kwargs),
                       tensor_metadata=meta(node.meta['val']))
        elif node.op=='output':
            if len(node.args)!=1 or not isinstance(node.args[0],(tuple,list)) or node.kwargs:
                raise ValueError('Flat AOT tensor return sequence required')
            if any(not isinstance(x,torch.fx.Node) for x in node.args[0]):
                raise ValueError('Only tensor node outputs supported')
            row['return_nodes']=[indices[x] for x in node.args[0]]
        else:raise ValueError('Unsupported native graph operation: '+node.op)
        rows.append(row)
    raw=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return dict(schema=1,sha256=hashlib.sha256(raw).hexdigest(),nodes=len(rows),
        inputs=sum(n.op=='placeholder' for n in nodes),constants=sum(n.op=='get_attr' for n in nodes))
