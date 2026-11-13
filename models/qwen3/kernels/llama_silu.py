# SPDX-License-Identifier: MIT
# MIT License
# 
# Copyright (c) 2023-2026 The ggml authors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SiLU aligned with llama.cpp Hexagon commit f818065d753700341b5355ce36ca676ecfb5d139.

Formula/guards: htp/hvx-sigmoid.h and htp/hvx-inverse.h. QFloat conversion
folds follow the pinned Hexagon SDK build, as verified by stage captures.
The arithmetic is Triton; packed primitive operations lower to HVX through
Hexagon-MLIR. No native C SiLU and no strict-reference rounding repair.
ABI: FP32 rows=1..32, width=3072, four programs, contiguous output.
"""
import triton
import triton.language as tl

@triton.jit
def _binary(a,b,code:tl.constexpr):
    return tl.inline_asm_elementwise(code,constraints="=v,v,v",args=[a,b],dtype=tl.int32,is_pure=True,pack=32)
@triton.jit
def _to_sf(a):
    return tl.inline_asm_elementwise("$0.sf = $1.qf32",constraints="=v,v",args=[a],dtype=tl.int32,is_pure=True,pack=32)
@triton.jit
def _int_to_sf(a):
    return tl.inline_asm_elementwise("$0.sf = $1.w",constraints="=v,v",args=[a],dtype=tl.int32,is_pure=True,pack=32)
@triton.jit
def _qf_mul_sf(a,b):
    return _binary(a,b,"$0.qf32 = vmpy($1.sf,$2.sf)")
@triton.jit
def _qf_mul(a,b):
    return _binary(a,b,"$0.qf32 = vmpy($1.qf32,$2.qf32)")
@triton.jit
def _qf_sub_sf(a,b):
    return _binary(a,b,"$0.qf32 = vsub($1.sf,$2.sf)")
@triton.jit
def _qf_add_mixed(a,b):
    return _binary(a,b,"$0.qf32 = vadd($1.qf32,$2.sf)")
@triton.jit
def _qf_add(a,b):
    return _binary(a,b,"$0.qf32 = vadd($1.qf32,$2.qf32)")
@triton.jit
def _qf_sub(a,b):
    return _binary(a,b,"$0.qf32 = vsub($1.qf32,$2.qf32)")
@triton.jit
def _qf_sub_mixed(a,b):
    return _binary(a,b,"$0.qf32 = vsub($1.qf32,$2.sf)")
@triton.jit
def _sigmoid_bits(x):
    two=tl.full((32,),0x40000000,tl.int32)
    log2=tl.full((32,),0x3fb8aa3b,tl.int32)
    half=tl.full((32,),0x3f000000,tl.int32)
    c1=tl.full((32,),0x3d009076,tl.int32)
    c2=tl.full((32,),0x3e8d74bd,tl.int32)
    v=_qf_mul_sf(_to_sf(_qf_mul_sf(x,log2)),half)
    bits=_to_sf(v)
    exponent=((bits>>23)&255)-127
    integer=((bits&0x007fffff)|0x00800000)>>tl.minimum(31,23-exponent)
    integer=tl.where(exponent<0,0,integer)
    integer=tl.where(bits<0,-integer,integer)
    t=_qf_sub_mixed(v,_int_to_sf(integer))
    square=_qf_mul(t,t)
    a=_qf_add_mixed(_qf_mul_sf(_to_sf(square),c2),log2)
    b=_qf_add(_qf_mul(_qf_mul_sf(_to_sf(t),c1),square),t)
    numerator=_to_sf(_qf_add(b,a))+(integer<<24)
    # Match the SDK-generated llama.cpp path: retain the QFloat denominator
    # and iterate in QFloat; converting both operands to FP32 here changes it.
    denominator_qf=_qf_sub_sf(numerator,_to_sf(_qf_sub(b,a)))
    denominator=_to_sf(denominator_qf)
    inv=0x7eeeebb3-denominator
    r=_qf_mul_sf(inv,_to_sf(_qf_sub_sf(two,_to_sf(_qf_mul_sf(inv,denominator)))))
    r=_qf_mul(r,_qf_sub_sf(two,_to_sf(_qf_mul(r,denominator_qf))))
    r=_qf_mul(r,_qf_sub_sf(two,_to_sf(_qf_mul(r,denominator_qf))))
    return _to_sf(_qf_mul_sf(numerator,_to_sf(r)))
@triton.jit
def _silu(x):
    safe=tl.where((x>-87.0)&(x<87.0),x,0.0)
    sig=_sigmoid_bits(safe.to(tl.int32,bitcast=True))
    sig=tl.where(87.0>x,sig,0x3f800000)
    sig=tl.where(-87.0<x,sig,0)
    return _to_sf(_qf_mul_sf(x.to(tl.int32,bitcast=True),sig)).to(tl.float32,bitcast=True)
@triton.jit
def silu_forward(X,Y,rows,stride,BLOCK:tl.constexpr=32):
    tl.static_assert(BLOCK==32)
    lane=tl.arange(0,BLOCK)
    for row in range(rows):
        # Four independent vectors, as in llama.cpp's sigmoid loop. Packed
        # conversions must lower to intrinsics so they do not split scheduling.
        for col in range(tl.program_id(0)*768,(tl.program_id(0)+1)*768,4*BLOCK):
            x0=tl.load(X+row*stride+col+lane)
            x1=tl.load(X+row*stride+col+32+lane)
            x2=tl.load(X+row*stride+col+64+lane)
            x3=tl.load(X+row*stride+col+96+lane)
            y0=_silu(x0)
            y1=_silu(x1)
            y2=_silu(x2)
            y3=_silu(x3)
            tl.store(Y+row*3072+col+lane,y0)
            tl.store(Y+row*3072+col+32+lane,y1)
            tl.store(Y+row*3072+col+64+lane,y2)
            tl.store(Y+row*3072+col+96+lane,y3)
