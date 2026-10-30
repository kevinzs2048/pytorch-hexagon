//===- FP32DecodeQKOrder.h - Explicit cached QK order -------------*- C++ -*-===//
// SPDX-License-Identifier: BSD-3-Clause.
//
// Opt-in numerical compatibility for [B,1,128] x [B,128,N], N=5..12, 24 or 32.
// The pinned native CPU's GEMV path handles four-column groups, a pair tail,
// and a single tail differently. Each pair uses eight partials for its first
// column, four for its second, and a four-term scalar tail for both.
// This is deliberately separate from the four-token prefill policy.
#ifndef HEXAGON_TRANSFORMS_FP32DECODEQKORDER_H
#define HEXAGON_TRANSFORMS_FP32DECODEQKORDER_H

#include "hexagon/Transforms/FP32AttentionDotOrder.h"

namespace mlir::hexagon {
inline LogicalResult applyFP32DecodeQKOrder(IRRewriter &rw,linalg::BatchMatmulOp op) {
  const Location loc=op.getLoc();
  const auto outputType=cast<RankedTensorType>(op.getResult(0).getType());
  const int64_t batch=outputType.getDimSize(0);
  rw.setInsertionPoint(op);
  auto zero=[&](ArrayRef<int64_t> shape)->Value {
    Value empty=tensor::EmptyOp::create(rw,loc,shape,rw.getF32Type());
    Value value=arith::ConstantOp::create(rw,loc,rw.getF32FloatAttr(0.0));
    return linalg::FillOp::create(rw,loc,value,empty).getResult(0);
  };
  auto slice=[&](Value input,ArrayRef<int64_t> offsets,ArrayRef<int64_t> shape)->Value {
    SmallVector<OpFoldResult> positions,sizes,strides(shape.size(),rw.getIndexAttr(1));
    for (auto offset:offsets)positions.push_back(rw.getIndexAttr(offset));
    for (auto extent:shape)sizes.push_back(rw.getIndexAttr(extent));
    return tensor::ExtractSliceOp::create(rw,loc,RankedTensorType::get(shape,rw.getF32Type()),
        input,positions,sizes,strides);
  };
  auto add=[&](Value left,Value right)->Value {
    auto type=cast<RankedTensorType>(left.getType());
    Value empty=tensor::EmptyOp::create(rw,loc,type.getShape(),rw.getF32Type());
    SmallVector<AffineMap> maps(3,rw.getMultiDimIdentityMap(type.getRank()));
    SmallVector<utils::IteratorType> iters(type.getRank(),utils::IteratorType::parallel);
    return linalg::GenericOp::create(rw,loc,TypeRange{type},ValueRange{left,right},ValueRange{empty},maps,iters,
        [](OpBuilder &builder,Location location,ValueRange args) {
          Value result=arith::AddFOp::create(builder,location,args[0],args[1]);
          linalg::YieldOp::create(builder,location,result);
        }).getResult(0);
  };
  auto combine=[&](Value a,Value b,Value accum,int64_t index)->Value {
    auto *ctx=rw.getContext();
    auto h=getAffineDimExpr(0,ctx),m=getAffineDimExpr(1,ctx),n=getAffineDimExpr(2,ctx);
    auto k=getAffineConstantExpr(index,ctx);
    SmallVector<AffineMap> maps{AffineMap::get(3,0,{h,m,k},ctx),
        AffineMap::get(3,0,{h,k,n},ctx),rw.getMultiDimIdentityMap(3)};
    SmallVector<utils::IteratorType> iters(3,utils::IteratorType::parallel);
    return linalg::GenericOp::create(rw,loc,TypeRange{accum.getType()},ValueRange{a,b},ValueRange{accum},maps,iters,
        [](OpBuilder &builder,Location location,ValueRange args) {
          Value product=arith::MulFOp::create(builder,location,args[0],args[1]);
          Value result=arith::AddFOp::create(builder,location,args[2],product);
          linalg::YieldOp::create(builder,location,result);
        }).getResult(0);
  };
  auto columnGroup=[&](int64_t column,int64_t columns,int64_t lanes,int64_t block)->Value {
    Value a=op.getDpsInputs()[0];
    Value b=slice(op.getDpsInputs()[1],{0,0,column},{batch,128,columns});
    Value prefix=zero({batch,1,columns});
    for (int64_t k=0;k<4;++k)prefix=combine(a,b,prefix,k);
    const int64_t end=4+(128-4)/block*block,groups=(end-4)/lanes;
    Value aMiddle=slice(a,{0,0,4},{batch,1,end-4});
    Value bMiddle=slice(b,{0,4,0},{batch,end-4,columns});
    auto aType=RankedTensorType::get({batch,1,groups,lanes},rw.getF32Type());
    auto bType=RankedTensorType::get({batch,groups,lanes,columns},rw.getF32Type());
    aMiddle=tensor::ExpandShapeOp::create(rw,loc,aType,aMiddle,
        ArrayRef<ReassociationIndices>{{0},{1},{2,3}});
    bMiddle=tensor::ExpandShapeOp::create(rw,loc,bType,bMiddle,
        ArrayRef<ReassociationIndices>{{0},{1,2},{3}});
    auto prefixType=RankedTensorType::get({batch,1,columns,1},rw.getF32Type());
    Value expandedPrefix=tensor::ExpandShapeOp::create(rw,loc,prefixType,prefix,
        ArrayRef<ReassociationIndices>{{0},{1},{2,3}});
    Value initial=zero({batch,1,columns,lanes});
    SmallVector<OpFoldResult> offsets(4,rw.getIndexAttr(0)),strides(4,rw.getIndexAttr(1)),sizes;
    for (auto extent:prefixType.getShape())sizes.push_back(rw.getIndexAttr(extent));
    initial=tensor::InsertSliceOp::create(rw,loc,expandedPrefix,initial,offsets,sizes,strides);
    auto *ctx=rw.getContext();
    auto h=getAffineDimExpr(0,ctx),m=getAffineDimExpr(1,ctx),n=getAffineDimExpr(2,ctx);
    auto lane=getAffineDimExpr(3,ctx),r=getAffineDimExpr(4,ctx);
    SmallVector<AffineMap> maps{AffineMap::get(5,0,{h,m,r,lane},ctx),
        AffineMap::get(5,0,{h,r,lane,n},ctx),AffineMap::get(5,0,{h,m,n,lane},ctx)};
    SmallVector<utils::IteratorType> iters(5,utils::IteratorType::parallel);iters[4]=utils::IteratorType::reduction;
    auto partial=linalg::GenericOp::create(rw,loc,TypeRange{initial.getType()},
        ValueRange{aMiddle,bMiddle},ValueRange{initial},maps,iters,
        [](OpBuilder &builder,Location location,ValueRange args) {
          Value product=arith::MulFOp::create(builder,location,args[0],args[1]);
          Value result=arith::AddFOp::create(builder,location,args[2],product);
          linalg::YieldOp::create(builder,location,result);
        });
    partial->setAttr("hexagon.fp32_decode_qk_lanes",rw.getI64IntegerAttr(lanes));
    partial->setAttr("hexagon.fp32_decode_qk_middle_end",rw.getI64IntegerAttr(end));
    Value value=partial.getResult(0);
    for (int64_t half=lanes/2;half>=1;half/=2)
      value=add(slice(value,{0,0,0,0},{batch,1,columns,half}),
                slice(value,{0,0,0,half},{batch,1,columns,half}));
    value=tensor::CollapseShapeOp::create(rw,loc,
        RankedTensorType::get({batch,1,columns},rw.getF32Type()),value,
        ArrayRef<ReassociationIndices>{{0},{1},{2,3}});
    for (int64_t k=end;k<128;++k)value=combine(a,b,value,k);
    return value;
  };
  Value result=tensor::EmptyOp::create(rw,loc,outputType.getShape(),rw.getF32Type());
  auto insert=[&](Value part,int64_t column)->Value {
    auto type=cast<RankedTensorType>(part.getType());
    SmallVector<OpFoldResult> offsets{rw.getIndexAttr(0),rw.getIndexAttr(0),rw.getIndexAttr(column)},sizes,
        strides(3,rw.getIndexAttr(1));
    for (auto extent:type.getShape())sizes.push_back(rw.getIndexAttr(extent));
    return tensor::InsertSliceOp::create(rw,loc,part,result,offsets,sizes,strides);
  };
  const int64_t groupedColumns=outputType.getDimSize(2)/4*4;
  // Isolated capacity128 candidate: columns are independent parallel indices.
  // Preserve each column's prefix, four partials, reduction order and tail.
  // Other admitted capacities retain their original grouping.
  if (outputType.getDimSize(2)==128 || outputType.getDimSize(2)==256 || outputType.getDimSize(2)==512 || outputType.getDimSize(2)==1024)
    result=insert(columnGroup(0,groupedColumns,4,4),0);
  else
    for (int64_t column=0;column<groupedColumns;column+=4)
      result=insert(columnGroup(column,4,4,4),column);
  for (int64_t column=groupedColumns;column<outputType.getDimSize(2);++column)
    result=insert(columnGroup(column,1,column%2==0?8:4,8),column);
  result.getDefiningOp()->setAttr("hexagon.fp32_decode_qk_order",rw.getStringAttr("prefix4-four-eight-opposite-tail"));
  rw.replaceOp(op,result);
  return success();
}
} // namespace mlir::hexagon
#endif
