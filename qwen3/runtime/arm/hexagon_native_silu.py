"""Keep native ATen SiLU until the Hexagon-specific compilation boundary.

The native graph stays executable by PyTorch with its original SiLU. A private
compiler graph expands only identified bounded FP32 SiLUs; the caller MUST bind
the named SiLU exponent lowering and runtime rather than generic math.exp.
This module defines the explicit policy helper; callers must opt in. Adding
this file does not change any default backend or the original CPU reference.
"""
import copy
import torch


def preserve_native_silu(table):
    op=torch.ops.aten.silu.default
    if op not in table:
        raise ValueError('Expected a standard SiLU decomposition entry')
    selected=dict(table)
    del selected[op]
    return selected


def lower_native_silu_for_hexagon(gm):
    before=gm.code
    native=[n for n in gm.graph.nodes if n.op=='call_function' and n.target==torch.ops.aten.silu.default]
    if not native:
        raise ValueError('No native SiLU to lower')
    for node in native:
        value=node.meta.get('val')
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.ndim!=3
                or any(type(n) is not int for n in value.shape)
                or not 1<=value.shape[0]<=28 or value.shape[1] not in (1,2) or value.shape[2]!=3072
                or len(node.args)!=1 or node.kwargs):
            raise ValueError('Only explicit bounded FP32 native SiLU is supported')
    # Copy graph structure, not 1.22GB of immutable parameter storage. The new
    # GraphModule shares constant tensors but never writes them.
    lowered=torch.fx.GraphModule(gm,copy.deepcopy(gm.graph))
    count=0
    for node in list(lowered.graph.nodes):
        if node.op!='call_function' or node.target!=torch.ops.aten.silu.default:
            continue
        x=node.args[0]
        with lowered.graph.inserting_before(node):
            negative=lowered.graph.call_function(torch.ops.aten.neg.default,(x,))
            exponential=lowered.graph.call_function(torch.ops.aten.exp.default,(negative,))
            denominator=lowered.graph.call_function(torch.ops.aten.add.Tensor,(exponential,1.0))
            result=lowered.graph.call_function(torch.ops.aten.div.Tensor,(x,denominator))
            for created in (negative,exponential,denominator,result):
                created.meta=dict(node.meta)
                created.meta['hexagon_native_silu_lowering']=True
        node.replace_all_uses_with(result)
        lowered.graph.erase_node(node)
        count+=1
    lowered.graph.lint();lowered.recompile()
    if gm.code!=before or count!=len(native):
        raise ValueError('Native graph changed or incomplete SiLU lowering')
    return lowered,dict(native_silu_count=count,native_graph_unchanged=True,
                        compiler_graph_only=True,required_policy='fp32-silu-exp-compat=true',
                        required_symbol='__hexagon_silu_exp_compat')
