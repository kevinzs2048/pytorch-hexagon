//===- FP32PrefillBlockOrder.h - Bounded native PV compatibility ---*- C++ -*-===//
// SPDX-License-Identifier: BSD-3-Clause
//
// Opt-in compatibility with the pinned native CPU Qwen prefill PV order.
// This is not a general accuracy optimization or global reassociation policy.
#ifndef HEXAGON_TRANSFORMS_FP32PREFILLBLOCKORDER_H
#define HEXAGON_TRANSFORMS_FP32PREFILLBLOCKORDER_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::hexagon {
inline void applyFP32PrefillPVBlockOrder(IRRewriter &rw,
                                       linalg::BatchMatmulOp op) {
  auto aType=cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto bType=cast<RankedTensorType>(op.getDpsInputs()[1].getType());
  auto outType=cast<RankedTensorType>(op.getResult(0).getType());
  // The enclosing candidate matcher must also check pure tensors, FP32,
  // canonical indexing maps, and a positive-zero output initializer.
  bool symbolic = (aType.getRank() == 3 && aType.getDimSize(0) == 16 &&
                   (aType.isDynamicDim(1) || (aType.getDimSize(1) >= 1 && aType.getDimSize(1) <= 32)) &&
                   aType.isDynamicDim(2)) &&
      bType.getShape() == ArrayRef<int64_t>({16,ShapedType::kDynamic,128}) &&
      outType.getShape() == ArrayRef<int64_t>({16,aType.getDimSize(1),128});
  assert(symbolic || aType.getShape()==ArrayRef<int64_t>({16,aType.getDimSize(1),1024}));
  assert(symbolic || ((aType.getDimSize(1)>=1 && aType.getDimSize(1)<1024) &&
         bType.getShape()==ArrayRef<int64_t>({16,1024,128})));
  const auto loc=op.getLoc();auto *ctx=op.getContext();rw.setInsertionPoint(op);
  Value dynamicRow;
  if (symbolic || outType.isDynamicDim(1))
    dynamicRow=tensor::DimOp::create(rw,loc,op.getDpsInputs()[0],1);
  if (symbolic) {
    Value cap=arith::ConstantIndexOp::create(rw,loc,1024);
    for (auto pair : {std::pair<Value,int64_t>{op.getDpsInputs()[0],2},
                      std::pair<Value,int64_t>{op.getDpsInputs()[1],1}}) {
      Value extent=tensor::DimOp::create(rw,loc,pair.first,pair.second);
      Value valid=arith::CmpIOp::create(rw,loc,arith::CmpIPredicate::eq,extent,cap);
      cf::AssertOp::create(rw,loc,valid,"ARMV8 PV order requires 1024 KV slots");
    }
  }
  auto h=getAffineDimExpr(0,ctx),row=getAffineDimExpr(1,ctx);
  auto k=getAffineDimExpr(2,ctx),col=getAffineDimExpr(3,ctx);
  SmallVector<AffineMap> dotMaps{AffineMap::get(4,0,{h,row,k},ctx),
      AffineMap::get(4,0,{h,k,col},ctx),AffineMap::get(4,0,{h,row,col},ctx)};
  SmallVector<utils::IteratorType> dotIters{utils::IteratorType::parallel,
      utils::IteratorType::parallel,utils::IteratorType::reduction,utils::IteratorType::parallel};
  auto slice=[&](Value input,int64_t dimension,int64_t begin,int64_t width)->Value {
    auto type=cast<RankedTensorType>(input.getType());SmallVector<int64_t> shape(type.getShape());
    shape[dimension]=width;
    SmallVector<OpFoldResult> offsets(3,rw.getIndexAttr(0)),strides(3,rw.getIndexAttr(1)),sizes;
    offsets[dimension]=rw.getIndexAttr(begin);
    for(int64_t i=0;i<3;++i)
      sizes.push_back(shape[i]==ShapedType::kDynamic
          ? OpFoldResult(tensor::DimOp::create(rw,loc,input,i))
          : OpFoldResult(rw.getIndexAttr(shape[i])));
    return tensor::ExtractSliceOp::create(rw,loc,RankedTensorType::get(shape,rw.getF32Type()),
                                          input,offsets,sizes,strides);
  };
  auto buildPV=[&](bool singleton)->Value {
  Value total=op.getDpsInits()[0];
  // Pinned OpenBLAS 0.3.30 ARMV8: SGEMM_Q=352, SGEMM_UNROLL_M=16.
  // driver/level3/level3.c balances the last two K blocks. For K=1024
  // this yields [352,336,336], not equal-sized blocks or a single sum.
  // query=1 dispatches native GEMV: one uninterrupted 1024-term FMA chain.
  // Larger query dispatches GEMM: each microkernel starts at +0 and adds to C.
  // Preserve that FMA/AddF boundary; do not permit reassociation.
  constexpr int64_t nativeBlockLimit=352, nativeUnroll=16;
  for(int64_t begin=0;begin<1024;){
    int64_t width=1024-begin;
    if(singleton) width=1024;
    else if(width>=2*nativeBlockLimit) width=nativeBlockLimit;
    else if(width>nativeBlockLimit)
      width=((width/2+nativeUnroll-1)/nativeUnroll)*nativeUnroll;
    Value a=slice(op.getDpsInputs()[0],2,begin,width);
    Value b=slice(op.getDpsInputs()[1],1,begin,width);
    Value empty=outType.isDynamicDim(1)
        ? Value(tensor::EmptyOp::create(rw,loc,outType.getShape(),rw.getF32Type(),ValueRange{dynamicRow}))
        : Value(tensor::EmptyOp::create(rw,loc,outType.getShape(),rw.getF32Type()));
    Value zero=arith::ConstantOp::create(rw,loc,rw.getF32FloatAttr(0.0));
    Value initial=linalg::FillOp::create(rw,loc,ValueRange{zero},ValueRange{empty}).getResult(0);
    auto partial=linalg::GenericOp::create(rw,loc,TypeRange{outType},ValueRange{a,b},
        ValueRange{initial},dotMaps,dotIters,
        [](OpBuilder &builder,Location location,ValueRange args){
          Value sum=math::FmaOp::create(builder,location,args[0],args[1],args[2]);
          linalg::YieldOp::create(builder,location,sum);
        });
    partial->setAttr("hexagon.fp32_prefill_pv_block_start",rw.getI64IntegerAttr(begin));
    if(singleton) return partial.getResult(0);
    Value sumEmpty=outType.isDynamicDim(1)
        ? Value(tensor::EmptyOp::create(rw,loc,outType.getShape(),rw.getF32Type(),ValueRange{dynamicRow}))
        : Value(tensor::EmptyOp::create(rw,loc,outType.getShape(),rw.getF32Type()));
    SmallVector<AffineMap> sumMaps(3,rw.getMultiDimIdentityMap(3));
    SmallVector<utils::IteratorType> sumIters(3,utils::IteratorType::parallel);
    auto sum=linalg::GenericOp::create(rw,loc,TypeRange{outType},ValueRange{total,partial.getResult(0)},
        ValueRange{sumEmpty},sumMaps,sumIters,
        [](OpBuilder &builder,Location location,ValueRange args){
          Value value=arith::AddFOp::create(builder,location,args[0],args[1]);
          linalg::YieldOp::create(builder,location,value);
        });
    total=sum.getResult(0);
    begin+=width;
  }
  return total;
  };
  Value total;
  if(outType.isDynamicDim(1)) {
    Value one=arith::ConstantIndexOp::create(rw,loc,1);
    Value singleton=arith::CmpIOp::create(rw,loc,arith::CmpIPredicate::eq,dynamicRow,one);
    auto select=scf::IfOp::create(rw,loc,TypeRange{outType},singleton,true);
    {
      OpBuilder::InsertionGuard guard(rw);
      rw.setInsertionPointToStart(&select.getThenRegion().front());
      scf::YieldOp::create(rw,loc,buildPV(true));
    }
    {
      OpBuilder::InsertionGuard guard(rw);
      rw.setInsertionPointToStart(&select.getElseRegion().front());
      scf::YieldOp::create(rw,loc,buildPV(false));
    }
    total=select.getResult(0);
  } else total=buildPV(outType.getDimSize(1)==1);
  total.getDefiningOp()->setAttr("hexagon.fp32_prefill_dot_order",
      rw.getStringAttr("pv-armv8-gemv1-gemm352-336-336-fma-ordered-add"));
  rw.replaceOp(op,ValueRange{total});
}
} // namespace mlir::hexagon
#endif
