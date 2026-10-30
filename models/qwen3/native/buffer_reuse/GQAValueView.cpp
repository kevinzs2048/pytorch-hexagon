// SPDX-License-Identifier: BSD-3-Clause
// Prove an identity value-layout copy and replace it by a tensor shape view.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace mlir;
bool integer(Value v,int64_t n) {
  auto c=v.getDefiningOp<arith::ConstantIndexOp>(); return c && c.value()==n;
}
bool integer(OpFoldResult v,int64_t n) {
  if (auto a=dyn_cast<Attribute>(v)) {
    auto i=dyn_cast<IntegerAttr>(a);return i && i.getInt()==n;
  }
  return integer(cast<Value>(v),n);
}
// Only identity dynamic dimensions are propagated; no runtime assumption.
Value extent(Value tensor,unsigned axis,unsigned depth=0) {
  if (depth>16) return {};
  if (auto loop=tensor.getDefiningOp<scf::ForOp>()) {
    if (loop.getInitArgs().size()!=1 || loop.getNumResults()!=1) return {};
    return extent(loop.getInitArgs()[0],axis,depth+1);
  }
  if (auto empty=tensor.getDefiningOp<tensor::EmptyOp>()) {
    auto sizes=empty.getMixedSizes();return dyn_cast<Value>(sizes[axis]);
  }
  if (auto expand=tensor.getDefiningOp<tensor::ExpandShapeOp>())
    return dyn_cast<Value>(expand.getMixedOutputShape()[axis]);
  if (auto collapse=tensor.getDefiningOp<tensor::CollapseShapeOp>()) {
    auto groups=collapse.getReassociationIndices();auto source=collapse.getSrc();
    auto type=cast<RankedTensorType>(source.getType());Value dynamic;
    for (int64_t d:groups[axis]) {
      if (type.isDynamicDim(d)) {
        if (dynamic) return {};dynamic=extent(source,d,depth+1);
        if (!dynamic) return {};
      } else if (type.getDimSize(d)!=1) return {};
    }
    return dynamic;
  }
  return {};
}
bool yield(scf::ForOp loop,Value value) {
  auto y=cast<scf::YieldOp>(loop.getBody()->getTerminator());
  return y.getNumOperands()==1 && y.getOperand(0)==value;
}
bool broadcastIdentity(Value index,Value column,linalg::GenericOp generic,Value size) {
  if (index==column) return true;
  auto select=index.getDefiningOp<arith::SelectOp>();
  if (!select || !integer(select.getTrueValue(),0) || select.getFalseValue()!=column ||
      generic.getNumDpsInputs()!=1 || select.getCondition()!=generic.getBody()->getArgument(0)) return false;
  auto input=generic.getDpsInputs()[0].getDefiningOp<tensor::FromElementsOp>();
  if (!input || input.getElements().size()!=1 ||
      cast<RankedTensorType>(input.getType()).getRank()!=0) return false;
  auto comparison=input.getElements()[0].getDefiningOp<arith::CmpIOp>();
  if (!comparison || comparison.getPredicate()!=arith::CmpIPredicate::eq) return false;
  return (comparison.getLhs()==size && integer(comparison.getRhs(),1)) ||
         (comparison.getRhs()==size && integer(comparison.getLhs(),1));
}
bool eliminate(scf::ForOp root) {
  if (root.getInitArgs().size()!=1 || root.getNumResults()!=1) return false;
  auto result=cast<RankedTensorType>(root.getResult(0).getType());
  if (!result.getElementType().isF32() || result.getShape()!=ArrayRef<int64_t>({16,ShapedType::kDynamic,128})) return false;
  auto empty=root.getInitArgs()[0].getDefiningOp<tensor::EmptyOp>();
  if (!empty || empty.getDynamicSizes().size()!=1) return false;
  Value size=empty.getDynamicSizes()[0];SmallVector<scf::ForOp> loops{root};
  for (auto child:root.getBody()->getOps<scf::ForOp>()) loops.push_back(child);
  if (loops.size()!=2) return false;
  for (auto loop:loops)
    if (loop.getInitArgs().size()!=1 || loop.getNumResults()!=1 ||
        !integer(loop.getLowerBound(),0) || !integer(loop.getStep(),1)) return false;
  scf::ForOp inner=loops[1];
  if (inner.getInitArgs()[0]!=root.getRegionIterArgs()[0] || !yield(root,inner.getResult(0))) return false;
  linalg::GenericOp generic;tensor::InsertSliceOp insert;bool valid=true;
  root.walk([&](Operation *op) {
    bool known=isa<scf::ForOp,scf::YieldOp,tensor::ExtractSliceOp,tensor::InsertSliceOp,
        tensor::ExtractOp,linalg::GenericOp,linalg::IndexOp,linalg::YieldOp,arith::SelectOp,
        arith::ConstantIndexOp>(op);
    if (!known) valid=false;
    if (auto g=dyn_cast<linalg::GenericOp>(op)) {if (generic) valid=false;generic=g;}
    if (auto i=dyn_cast<tensor::InsertSliceOp>(op)) {if (insert) valid=false;insert=i;}
  });
  if (!valid || !generic || !insert || generic.getNumResults()!=1 || generic.getNumDpsInits()!=1 ||
      generic.getNumDpsInputs()>1 || generic.getNumLoops()!=1 || generic.getNumReductionLoops() ||
      !generic.getBody()->getArgument(generic.getNumDpsInputs()).use_empty() ||
      generic->getBlock()!=inner.getBody() || insert->getBlock()!=inner.getBody()) return false;
  auto slice=generic.getDpsInits()[0].getDefiningOp<tensor::ExtractSliceOp>();
  if (!slice || slice.getSource()!=inner.getRegionIterArgs()[0] || insert.getSource()!=generic.getResult(0) ||
      insert.getDest()!=inner.getRegionIterArgs()[0] || !yield(inner,insert.getResult()) ||
      slice.getMixedOffsets()!=insert.getMixedOffsets() || slice.getMixedSizes()!=insert.getMixedSizes() ||
      slice.getMixedStrides()!=insert.getMixedStrides()) return false;
  auto offsets=slice.getMixedOffsets(),sizes=slice.getMixedSizes(),strides=slice.getMixedStrides();
  if (offsets.size()!=3 || !integer(offsets[2],0) || !integer(sizes[0],1) || !integer(sizes[1],1) ||
      !integer(sizes[2],128) || !llvm::all_of(strides,[](OpFoldResult s){return integer(s,1);})) return false;
  Value head=dyn_cast<Value>(offsets[0]),column=dyn_cast<Value>(offsets[1]);
  auto headLoop=llvm::find_if(loops,[&](scf::ForOp l){return l.getInductionVar()==head;});
  auto columnLoop=llvm::find_if(loops,[&](scf::ForOp l){return l.getInductionVar()==column;});
  if (headLoop==loops.end() || columnLoop==loops.end() || headLoop==columnLoop ||
      !integer(headLoop->getUpperBound(),16) || columnLoop->getUpperBound()!=size) return false;
  auto y=cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  if (y.getValues().size()!=1) return false;
  auto extract=y.getValues()[0].getDefiningOp<tensor::ExtractOp>();
  if (!extract || extract.getIndices().size()!=4 || !root.isDefinedOutsideOfLoop(extract.getTensor())) return false;
  auto sourceType=cast<RankedTensorType>(extract.getTensor().getType());
  if (sourceType.getShape()!=ArrayRef<int64_t>({1,16,ShapedType::kDynamic,128}) ||
      !sourceType.getElementType().isF32() || extent(extract.getTensor(),2)!=size) return false;
  auto indices=extract.getIndices();auto component=indices[3].getDefiningOp<linalg::IndexOp>();
  if (!integer(indices[0],0) || indices[1]!=head || !component || component.getDim()!=0 ||
      !broadcastIdentity(indices[2],column,generic,size)) return false;
  // The selected N==1 index is also the original column (its domain is [0,1)).
  OpBuilder builder(root);
  SmallVector<ReassociationIndices> groups{{0,1},{2},{3}};
  auto view=tensor::CollapseShapeOp::create(builder,root.getLoc(),result,extract.getTensor(),groups);
  root.getResult(0).replaceAllUsesWith(view.getResult());root.erase();
  llvm::errs()<<"QWEN_GQA_VALUE_VIEW eliminated_identity_copy source="<<sourceType<<" result="<<result
              <<" dynamic-domain-proved FP-arithmetic-changed=0\n";
  return true;
}
} // namespace

mlir::LogicalResult eliminateQwenValueLayoutCopy(mlir::ModuleOp module) {
  using namespace mlir;unsigned changed=0;
  for (auto function:module.getOps<func::FuncOp>()) {
    if (function.isDeclaration()) continue;SmallVector<scf::ForOp> roots;
    for (auto loop:function.front().getOps<scf::ForOp>())
      if (loop.getNumResults()==1 && isa<RankedTensorType>(loop.getResult(0).getType())) roots.push_back(loop);
    unsigned local=0;for (auto loop:roots) local+=eliminate(loop);changed+=local;
#ifdef QWEN_GQA_VALUE_FORWARD
    if (local==28 && function.getSymName()=="Qwen3DynamicBoundProbe")
      function->setAttr("qwen.gqa_identity_view_complete",BoolAttr::get(module.getContext(),true));
#endif
  }
  llvm::errs()<<"QWEN_GQA_VALUE_VIEW eliminated="<<changed<<" FP-arithmetic-changed=0\n";
  return success();
}
