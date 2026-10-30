//===- FP32AttentionDotOrder.h - Explicit short attention order ---*- C++ -*-===//
// SPDX-License-Identifier: BSD-3-Clause.
//
// Opt-in numerical compatibility policy for the tested FP32 four-token
// attention shapes. It is not a universally more accurate GEMM algorithm.
// The pinned CPU implementation uses layout-dependent algorithms: QK's
// transposed K storage follows four partials, PV follows sequential FMA.
#ifndef HEXAGON_TRANSFORMS_FP32ATTENTIONDOTORDER_H
#define HEXAGON_TRANSFORMS_FP32ATTENTIONDOTORDER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

namespace mlir::hexagon {
inline bool isFP32AttentionDotCandidate(linalg::BatchMatmulOp op,bool decodeQK=false,bool decodePV=false,
                                      bool pairQK=false,bool pairPV=false,bool squareDecode=false,int64_t decodeCapacity=0,
                                      int64_t pairCapacity=0,int64_t prefillQuery=0) {
  if (!op.hasPureTensorSemantics() || op->hasAttr("library_call") ||
      op.getNumDpsInputs()!=2 || op.getNumDpsInits()!=1 || op.getNumResults()!=1)
    return false;
  auto a=cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto b=cast<RankedTensorType>(op.getDpsInputs()[1].getType());
  auto c=cast<RankedTensorType>(op.getDpsInits()[0].getType());
  for (auto type : {a,b,c})
    if (type.getRank()!=3 || !type.hasStaticShape() || type.getEncoding() ||
        !type.getElementType().isF32()) return false;
  const int64_t batch=a.getDimSize(0), m=a.getDimSize(1), k=a.getDimSize(2), n=b.getDimSize(2);
  // N=24, N=32 and N=64 are separately checked native CPU attention shapes.
  // N=64 is an opt-in candidate until independent DSP controls also pass.
  // Do not infer support for intermediate sizes from these bounded cases.
  // The two-query CPU controls independently match four interleaved QK
  // partials (NO decode prefix), and sequential unfused PV. Never silently
  // broaden the single-query/four-query policies to this separate ABI.
  bool shapeAllowed=prefillQuery ? (batch==16 && (prefillQuery==3 || prefillQuery==4 || prefillQuery==8) && m==prefillQuery &&
      ((k==128 && n==1024) || (k==1024 && n==128)))
                            : pairQK ? (m==2 && k==128 && (n==64 || (pairCapacity && n==pairCapacity)))
                            : pairPV ? (m==2 && (k==64 || (pairCapacity && k==pairCapacity)) && n==128)
                            : decodePV ? (m==1 && (k==64 || (squareDecode && k==128) || (decodeCapacity && k==decodeCapacity)) && n==128)
                            : decodeQK ? (m==1 && k==128 && ((n>=5 && n<=12) || n==24 || n==32 || n==64 || (squareDecode && n==128) || (decodeCapacity && n==decodeCapacity)))
                            : (m==4 && ((k==128 && n==4) || (k==4 && n==128)));
  if (batch<1 || batch>1024 || !shapeAllowed ||
      b.getDimSize(0)!=batch || b.getDimSize(1)!=k ||
      c.getShape()!=ArrayRef<int64_t>({batch,m,n})) return false;
  auto *ctx=op.getContext();
  auto h=getAffineDimExpr(0,ctx), row=getAffineDimExpr(1,ctx);
  auto col=getAffineDimExpr(2,ctx), reduction=getAffineDimExpr(3,ctx);
  SmallVector<AffineMap> maps{AffineMap::get(4,0,{h,row,reduction},ctx),
      AffineMap::get(4,0,{h,reduction,col},ctx),AffineMap::get(4,0,{h,row,col},ctx)};
  if (op.getIndexingMapsArray()!=maps) return false;
  auto fill=op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  FloatAttr initial;
  if (!fill || !fill.hasPureTensorSemantics() || fill->hasAttr("library_call") ||
      !matchPattern(fill.getInputs()[0],m_Constant(&initial)) ||
      !initial.getValue().isZero() || initial.getValue().isNegative()) return false;
  Block *fillBody=fill.getBody();
  auto fillYield=dyn_cast<linalg::YieldOp>(fillBody->getTerminator());
  if (fillBody->getOperations().size()!=1 || !fillYield ||
      fillYield.getNumOperands()!=1 || fillYield.getOperand(0)!=fillBody->getArgument(0)) return false;
  Block *body=op.getBody();
  if (body->getOperations().size()!=3 || body->getNumArguments()!=3) return false;
  auto mul=dyn_cast<arith::MulFOp>(body->front());
  auto yield=dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!mul || !yield || yield.getNumOperands()!=1) return false;
  auto add=yield.getOperand(0).getDefiningOp<arith::AddFOp>();
  return add && mul.getLhs()==body->getArgument(0) && mul.getRhs()==body->getArgument(1) &&
      mul.getFastmath()==arith::FastMathFlags::none && add.getFastmath()==arith::FastMathFlags::none &&
      ((add.getLhs()==mul.getResult() && add.getRhs()==body->getArgument(2)) ||
       (add.getRhs()==mul.getResult() && add.getLhs()==body->getArgument(2)));
}

// Explicit native CPU controls R1589/R1590 support this bounded candidate.
// No default shape admission, no global contraction or reassociation permission.
inline void applyFP32PrefillAttentionOrder(IRRewriter &rw,linalg::BatchMatmulOp op) {
  auto *ctx=op.getContext();
  auto h=getAffineDimExpr(0,ctx), row=getAffineDimExpr(1,ctx);
  auto reduction=getAffineDimExpr(2,ctx), col=getAffineDimExpr(3,ctx);
  SmallVector<AffineMap> maps{AffineMap::get(4,0,{h,row,reduction},ctx),
      AffineMap::get(4,0,{h,reduction,col},ctx),AffineMap::get(4,0,{h,row,col},ctx)};
  SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
      utils::IteratorType::parallel,utils::IteratorType::reduction,utils::IteratorType::parallel};
  rw.setInsertionPoint(op);
  auto ordered=linalg::GenericOp::create(rw,op.getLoc(),op.getResultTypes(),
      op.getDpsInputs(),op.getDpsInits(),maps,iters,
      [](OpBuilder &builder,Location location,ValueRange args) {
        Value sum=math::FmaOp::create(builder,location,args[0],args[1],args[2]);
        linalg::YieldOp::create(builder,location,sum);
      });
  ordered->setAttr("hexagon.fp32_prefill_dot_order",rw.getStringAttr("sequential-fma-output-columns"));
  rw.replaceOp(op,ordered.getResults());
}

inline LogicalResult applyFP32DecodePVOrder(IRRewriter &rw,linalg::BatchMatmulOp op,bool pair=false) {
  // Put the sequential reduction before independent output columns. The
  // ordinary tiler can vectorize columns without splitting the FP32 sum into
  // independent partials. Native CPU corpus R503 checks unfused left-fold
  // arithmetic; this opt-in is not a reassociation/fast-math permission.
  auto *ctx=op.getContext();
  auto h=getAffineDimExpr(0,ctx), row=getAffineDimExpr(1,ctx);
  auto reduction=getAffineDimExpr(2,ctx), col=getAffineDimExpr(3,ctx);
  SmallVector<AffineMap> maps{AffineMap::get(4,0,{h,row,reduction},ctx),
      AffineMap::get(4,0,{h,reduction,col},ctx),AffineMap::get(4,0,{h,row,col},ctx)};
  SmallVector<utils::IteratorType> iters{utils::IteratorType::parallel,
      utils::IteratorType::parallel,utils::IteratorType::reduction,utils::IteratorType::parallel};
  rw.setInsertionPoint(op);
  auto ordered=linalg::GenericOp::create(rw,op.getLoc(),op.getResultTypes(),
      op.getDpsInputs(),op.getDpsInits(),maps,iters,
      [](OpBuilder &builder,Location location,ValueRange args) {
        Value product=arith::MulFOp::create(builder,location,args[0],args[1]);
        Value sum=arith::AddFOp::create(builder,location,args[2],product);
        linalg::YieldOp::create(builder,location,sum);
      });
  ordered->setAttr(pair ? "hexagon.fp32_pair_pv_order" : "hexagon.fp32_decode_pv_order",
                   rw.getStringAttr("sequential-unfused-output-columns"));
  rw.replaceOp(op,ordered.getResults());
  return success();
}

inline LogicalResult applyFP32AttentionDotOrder(IRRewriter &rw,linalg::BatchMatmulOp op,bool pair=false) {
  auto type=cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  const Location loc=op.getLoc();
  rw.setInsertionPoint(op);
  if (type.getDimSize(2)==4) {
    // Explicit FMA is intentional here, never enabled via global fast-math.
    auto resultType=cast<RankedTensorType>(op.getResult(0).getType());
    auto generic=linalg::GenericOp::create(rw,loc,TypeRange{resultType},
        op.getDpsInputs(),op.getDpsInits(),op.getIndexingMapsArray(),op.getIteratorTypesArray(),
        [](OpBuilder &builder,Location location,ValueRange args) {
          Value sum=math::FmaOp::create(builder,location,args[0],args[1],args[2]);
          linalg::YieldOp::create(builder,location,sum);
        });
    generic->setAttr("hexagon.fp32_attention_dot_order",rw.getStringAttr("context-sequential-fma"));
    rw.replaceOp(op,generic.getResults());
    return success();
  }
  auto split=linalg::splitReduction(rw,op,[](linalg::LinalgOp) {
    return linalg::SplitReductionOptions{4,3,true};
  });
  if (failed(split)) return failure();
  auto combiner=split->resultCombiningLinalgOp;
  Value partial=split->splitLinalgOp->getResult(0);
  split->splitLinalgOp->setAttr(pair ? "hexagon.fp32_pair_qk_partials" : "hexagon.fp32_attention_dot_partials",
                              rw.getI64IntegerAttr(4));
  rw.setInsertionPoint(combiner);
  auto slice=[&](Value input,int64_t offset,int64_t length)->Value {
    auto t=cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> shape(t.getShape());shape[3]=length;
    auto sliced=RankedTensorType::get(shape,rw.getF32Type());
    SmallVector<OpFoldResult> offsets(4,rw.getIndexAttr(0)),strides(4,rw.getIndexAttr(1)),sizes;
    offsets[3]=rw.getIndexAttr(offset);
    for (auto extent:shape) sizes.push_back(rw.getIndexAttr(extent));
    return tensor::ExtractSliceOp::create(rw,loc,sliced,input,offsets,sizes,strides);
  };
  for (int64_t half=2;half>=1;half/=2) {
    Value left=slice(partial,0,half),right=slice(partial,half,half);
    auto t=cast<RankedTensorType>(left.getType());
    Value empty=tensor::EmptyOp::create(rw,loc,t.getShape(),rw.getF32Type());
    SmallVector<AffineMap> maps(3,rw.getMultiDimIdentityMap(4));
    SmallVector<utils::IteratorType> iters(4,utils::IteratorType::parallel);
    partial=linalg::GenericOp::create(rw,loc,TypeRange{t},ValueRange{left,right},ValueRange{empty},maps,iters,
        [](OpBuilder &builder,Location location,ValueRange args) {
          Value sum=arith::AddFOp::create(builder,location,args[0],args[1]);
          linalg::YieldOp::create(builder,location,sum);
        }).getResult(0);
  }
  combiner.getDpsInputOperand(0)->set(partial);
  combiner->setAttr(pair ? "hexagon.fp32_pair_qk_order" : "hexagon.fp32_attention_dot_order",
                    rw.getStringAttr("qk-four-partials-opposite-half"));
  return success();
}
} // namespace mlir::hexagon
#endif
