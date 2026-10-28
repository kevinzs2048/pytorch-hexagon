"""SiLU IR fixture from Qwen3 development revision feb45fd.

Retained independently of the historical inspection/compilation launcher.
"""
NONE = '#llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>'


def module(kind,effects,dynamic_row=False):
    declaration=' attributes {llvm.readnone}' if effects=='declaration' else ''
    if effects=='declaration-contract':
        declaration=' attributes {llvm.readnone, passthrough = ["nounwind", "willreturn"]}'
    if effects=='call-contract':
        declaration=' attributes {passthrough = ["nounwind", "willreturn"]}'
    call_attributes=' {memory_effects = '+NONE+'}' if effects=='call' else ''
    if effects=='call-contract':call_attributes=' {memory_effects = '+NONE+', no_unwind}'
    head='module {\nfunc.func private @__hexagon_silu_exp_compat(f32) -> f32'+declaration+'\n'
    def call(result,value):
        return f'{result} = func.call @__hexagon_silu_exp_compat({value}){call_attributes} : (f32) -> f32'
    if kind=='scalar':
        return head+'''func.func @probe(%x: f32) -> f32 attributes {llvm.emit_c_interface} {
'''+call('%discard','%x')+'\n'+call('%left','%x')+'\n'+call('%right','%x')+'''
%sum = arith.addf %left, %right : f32
return %sum : f32
}}
'''
    ty='tensor<1x?x3072xf32>' if dynamic_row else 'tensor<1x1x3072xf32>'
    empty=('%axis = arith.constant 1 : index\n%query = tensor.dim %x, %axis : '+ty+'\n%empty = tensor.empty(%query) : '+ty
           if dynamic_row else '%empty = tensor.empty() : '+ty)
    maps='indexing_maps = [affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,b,c)>], iterator_types = ["parallel", "parallel", "parallel"]'
    maps3='indexing_maps = [affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,b,c)>], iterator_types = ["parallel", "parallel", "parallel"]'
    return head+f'''func.func @probe(%x: {ty}, %other: {ty}) -> {ty} attributes {{llvm.emit_c_interface}} {{
%one = arith.constant 1.0 : f32
{empty}
%denominator = linalg.generic {{{maps}}} ins(%x : {ty}) outs(%empty : {ty}) {{
^bb0(%v: f32,%unused: f32):
%negative = arith.negf %v : f32
'''+call('%exp','%negative')+f'''
%den = arith.addf %exp, %one : f32
linalg.yield %den : f32
}} -> {ty}
%division = linalg.generic {{{maps3}}} ins(%x,%denominator : {ty},{ty}) outs(%empty : {ty}) {{
^bb0(%v: f32,%d: f32,%unused: f32):
%q = arith.divf %v,%d : f32
linalg.yield %q : f32
}} -> {ty}
%product = linalg.generic {{{maps3}}} ins(%division,%other : {ty},{ty}) outs(%empty : {ty}) {{
^bb0(%v: f32,%o: f32,%unused: f32):
%p = arith.mulf %v,%o : f32
linalg.yield %p : f32
}} -> {ty}
return %product : {ty}
}}}}
'''
