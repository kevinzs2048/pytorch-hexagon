// SPDX-License-Identifier: BSD-3-Clause
// Prototype: eight independent QK columns retain the original ascending
// 128-component FMA recurrence; all dynamic column tails remain masked.
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
bool constant(Value value,int64_t expected) {
  auto c=value.getDefiningOp<arith::ConstantIndexOp>();return c && c.value()==expected;
}
bool loop(scf::ForOp op) {
  return op && constant(op.getLowerBound(),0) && op.getNumRegionIterArgs()==1;
}
struct Candidate {
  scf::ForOp head,m,k,n;
  affine::AffineMinOp extent;
  tensor::ExtractSliceOp key,accum;
  arith::SelectOp select;
  tensor::ExtractOp query;
  tensor::DimOp dim;
  vector::CreateMaskOp mask;
  vector::TransferReadOp keyRead,accumRead;
  vector::BroadcastOp broadcast;
  math::FmaOp fma;
  vector::TransferWriteOp write;
  tensor::InsertSliceOp insert;
};
std::optional<Candidate> prove(scf::ForOp head) {
  if (!loop(head) || !constant(head.getUpperBound(),16) || !constant(head.getStep(),1) ||
      head.getBody()->getOperations().size()!=2) return {};
  auto m=dyn_cast<scf::ForOp>(head.getBody()->front());
  if (!loop(m) || !constant(m.getStep(),1) || m.getBody()->getOperations().size()!=2 ||
      m.getInitArgs()[0]!=head.getRegionIterArgs()[0]) return {};
  auto k=dyn_cast<scf::ForOp>(m.getBody()->front());
  if (!loop(k) || !constant(k.getUpperBound(),128) || !constant(k.getStep(),1) ||
      k.getBody()->getOperations().size()!=2 || k.getInitArgs()[0]!=m.getRegionIterArgs()[0]) return {};
  auto n=dyn_cast<scf::ForOp>(k.getBody()->front());
  if (!loop(n) || !constant(n.getStep(),128) || n.getInitArgs()[0]!=k.getRegionIterArgs()[0] ||
      n.getBody()->getOperations().size()!=14) return {};
  Candidate c{head,m,k,n};auto it=n.getBody()->begin();
  c.extent=dyn_cast<affine::AffineMinOp>(*it++);c.key=dyn_cast<tensor::ExtractSliceOp>(*it++);
  c.accum=dyn_cast<tensor::ExtractSliceOp>(*it++);c.select=dyn_cast<arith::SelectOp>(*it++);
  c.query=dyn_cast<tensor::ExtractOp>(*it++);c.dim=dyn_cast<tensor::DimOp>(*it++);
  c.mask=dyn_cast<vector::CreateMaskOp>(*it++);c.keyRead=dyn_cast<vector::TransferReadOp>(*it++);
  c.accumRead=dyn_cast<vector::TransferReadOp>(*it++);c.broadcast=dyn_cast<vector::BroadcastOp>(*it++);
  c.fma=dyn_cast<math::FmaOp>(*it++);c.write=dyn_cast<vector::TransferWriteOp>(*it++);
  c.insert=dyn_cast<tensor::InsertSliceOp>(*it++);
  if (!c.extent || !c.key || !c.accum || !c.select || !c.query || !c.dim || !c.mask ||
      !c.keyRead || !c.accumRead || !c.broadcast || !c.fma || !c.write || !c.insert) return {};
  auto ctx=head.getContext();auto map=AffineMap::get(1,1,
      {getAffineSymbolExpr(0,ctx)-getAffineDimExpr(0,ctx),getAffineConstantExpr(128,ctx)},ctx);
  if (c.extent.getAffineMap()!=map || c.extent.getOperands()!=ValueRange({n.getInductionVar(),n.getUpperBound()})) return {};
  auto keytype=dyn_cast<RankedTensorType>(c.key.getSource().getType());
  auto qtype=dyn_cast<RankedTensorType>(c.query.getTensor().getType());
  if (!keytype || keytype.getShape()!=ArrayRef<int64_t>({16,128,ShapedType::kDynamic}) ||
      !qtype || qtype.getShape()!=ArrayRef<int64_t>({1,16,ShapedType::kDynamic,128}) ||
      !keytype.getElementType().isF32() || !qtype.getElementType().isF32() ||
      c.key.getOffsets()!=ValueRange({head.getInductionVar(),k.getInductionVar(),n.getInductionVar()}) ||
      c.accum.getSource()!=n.getRegionIterArgs()[0] ||
      c.accum.getOffsets()!=ValueRange({head.getInductionVar(),m.getInductionVar(),n.getInductionVar()}) ||
      c.query.getIndices()!=ValueRange({head.getLowerBound(),head.getInductionVar(),c.select.getResult(),k.getInductionVar()}) ||
      !constant(c.select.getTrueValue(),0) || c.select.getFalseValue()!=m.getInductionVar() ||
      c.dim.getSource()!=c.key.getResult() || !constant(c.dim.getIndex(),0) ||
      c.mask.getOperands()!=ValueRange({c.dim.getResult()}) ||
      c.keyRead.getBase()!=c.key.getResult() || c.accumRead.getBase()!=c.accum.getResult() ||
      c.broadcast.getSource()!=c.query.getResult() ||
      c.fma->getOperand(0)!=c.broadcast.getResult() || c.fma->getOperand(1)!=c.keyRead.getResult() ||
      c.fma->getOperand(2)!=c.accumRead.getResult() || c.fma.getFastmath()!=arith::FastMathFlags::none ||
      c.write.getVector()!=c.fma.getResult() || c.write.getBase()!=c.accum.getResult() ||
      c.insert.getSource()!=c.write.getResult() || c.insert.getDest()!=n.getRegionIterArgs()[0] ||
      c.insert.getOffsets()!=c.accum.getOffsets() ||
      cast<scf::YieldOp>(n.getBody()->getTerminator()).getOperand(0)!=c.insert.getResult() ||
      cast<scf::YieldOp>(k.getBody()->getTerminator()).getOperand(0)!=n.getResult(0) ||
      cast<scf::YieldOp>(m.getBody()->getTerminator()).getOperand(0)!=k.getResult(0) ||
      cast<scf::YieldOp>(head.getBody()->getTerminator()).getOperand(0)!=m.getResult(0)) return {};
  for (auto slice : {c.key,c.accum})
    if (slice.getType().getShape()!=ArrayRef<int64_t>({ShapedType::kDynamic}) ||
        slice.getStaticSizes()!=ArrayRef<int64_t>({1,1,ShapedType::kDynamic}) ||
        slice.getSizes()!=ValueRange({c.extent.getResult()}) ||
        slice.getStaticStrides()!=ArrayRef<int64_t>({1,1,1})) return {};
  if (c.insert.getStaticSizes()!=c.accum.getStaticSizes() ||
      c.insert.getSizes()!=c.accum.getSizes() || c.insert.getStaticStrides()!=c.accum.getStaticStrides()) return {};
  auto vector128=VectorType::get({128},Float32Type::get(ctx));
  auto identity=AffineMap::getMultiDimIdentityMap(1,ctx);
  for (auto read : {c.keyRead,c.accumRead})
    if (read.getType()!=vector128 || read.getMask()!=c.mask.getResult() ||
        read.getPermutationMap()!=identity || read.getIndices().size()!=1 ||
        !constant(read.getIndices()[0],0)) return {};
  if (c.fma.getType()!=vector128 || c.broadcast.getType()!=vector128 ||
      c.write.getPermutationMap()!=identity || c.write.getMask()!=c.mask.getResult() ||
      c.write.getIndices().size()!=1 || !constant(c.write.getIndices()[0],0)) return {};
  return c;
}
Operation *clone8(OpBuilder &b,Operation *op,IRMapping &map) {
  auto copy=b.clone(*op,map);
  for (auto result : copy->getResults())
    if (auto vector=dyn_cast<VectorType>(result.getType()))
      if (vector.getShape()==ArrayRef<int64_t>({128})) result.setType(VectorType::get({8},vector.getElementType()));
  return copy;
}
scf::ForOp rewrite(Candidate c) {
  OpBuilder b(c.head);auto loc=c.head.getLoc();auto eight=arith::ConstantIndexOp::create(b,loc,8);
  auto head=scf::ForOp::create(b,loc,c.head.getLowerBound(),c.head.getUpperBound(),c.head.getStep(),c.head.getInitArgs());
  b.setInsertionPointToStart(head.getBody());
  auto m=scf::ForOp::create(b,loc,c.m.getLowerBound(),c.m.getUpperBound(),c.m.getStep(),head.getRegionIterArgs());
  b.setInsertionPointToStart(m.getBody());
  auto n=scf::ForOp::create(b,loc,c.n.getLowerBound(),c.n.getUpperBound(),eight,m.getRegionIterArgs());
  b.setInsertionPointToStart(n.getBody());IRMapping map;
  map.map(c.head.getInductionVar(),head.getInductionVar());map.map(c.m.getInductionVar(),m.getInductionVar());
  map.map(c.n.getInductionVar(),n.getInductionVar());map.map(c.n.getRegionIterArgs()[0],n.getRegionIterArgs()[0]);
  auto extent=clone8(b,c.extent,map);auto ctx=b.getContext();
  extent->setAttr("map",AffineMapAttr::get(AffineMap::get(1,1,
      {getAffineSymbolExpr(0,ctx)-getAffineDimExpr(0,ctx),getAffineConstantExpr(8,ctx)},ctx)));
  clone8(b,c.accum,map);
  // Both original tensor slices have exactly this same bounded extent.
  auto mask=vector::CreateMaskOp::create(b,loc,VectorType::get({8},b.getI1Type()),ValueRange{map.lookup(c.extent.getResult())});
  map.map(c.mask.getResult(),mask.getResult());clone8(b,c.accumRead,map);
  auto k=scf::ForOp::create(b,loc,c.k.getLowerBound(),c.k.getUpperBound(),c.k.getStep(),ValueRange{map.lookup(c.accumRead.getResult())});
  b.setInsertionPointToStart(k.getBody());map.map(c.k.getInductionVar(),k.getInductionVar());
  map.map(c.accumRead.getResult(),k.getRegionIterArgs()[0]);
  for (Operation *op : {c.key.getOperation(),c.select.getOperation(),c.query.getOperation(),c.keyRead.getOperation(),c.broadcast.getOperation(),c.fma.getOperation()}) clone8(b,op,map);
  b.setInsertionPointToEnd(k.getBody());scf::YieldOp::create(b,loc,map.lookup(c.fma.getResult()));
  b.setInsertionPointAfter(k);map.map(c.fma.getResult(),k.getResult(0));clone8(b,c.write,map);clone8(b,c.insert,map);
  b.setInsertionPointToEnd(n.getBody());scf::YieldOp::create(b,loc,map.lookup(c.insert.getResult()));
  b.setInsertionPointToEnd(m.getBody());scf::YieldOp::create(b,loc,n.getResults());
  b.setInsertionPointToEnd(head.getBody());scf::YieldOp::create(b,loc,m.getResults());
  c.head.getResult(0).replaceAllUsesWith(head.getResult(0));c.head.erase();return head;
}
}
mlir::LogicalResult retainOriginalDynamicPV(mlir::ModuleOp module);
mlir::LogicalResult instrumentQwenLoopRegions(mlir::ModuleOp module) {
  using namespace mlir;SmallVector<Candidate> candidates;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto order=function->getAttrOfType<StringAttr>("hexagon.fp32_prefill_order");
    if (function.getSymName()!="Qwen3DynamicBoundProbe" || function.isDeclaration() ||
        function.getNumArguments()!=65 || function.getNumResults()!=57 || !order ||
        order.getValue()!="qk-fma-pv-openblas-armv8-k1024") continue;
    for (Operation &op : function.getBody().front())
      if (auto head=dyn_cast<scf::ForOp>(op))
        if (auto candidate=prove(head)) candidates.push_back(*candidate);
  }
  if (candidates.size()!=28) {module.emitError("Strict original all28 QK register proofs required");return failure();}
  llvm::errs()<<"QWEN_DYNAMIC_QK_REGISTER_BEFORE_BEGIN\n";
  candidates.front().head->print(llvm::errs(),OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs()<<"\nQWEN_DYNAMIC_QK_REGISTER_BEFORE_END\n";
  SmallVector<scf::ForOp> rewritten;
  for (auto candidate : candidates) rewritten.push_back(rewrite(candidate));
  if (failed(verify(module))) return failure();
  llvm::errs()<<"QWEN_DYNAMIC_QK_REGISTER verified=28 lanes=8 component_order=original128 tails=original no_fast_math=1\nQWEN_DYNAMIC_QK_REGISTER_AFTER_BEGIN\n";
  rewritten.front()->print(llvm::errs(),OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs()<<"\nQWEN_DYNAMIC_QK_REGISTER_AFTER_END\nQWEN_DYNAMIC_QK_REGISTER_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE\n";
  return retainOriginalDynamicPV(module);
}
