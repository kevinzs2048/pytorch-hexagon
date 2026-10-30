// SPDX-License-Identifier: BSD-3-Clause
// Prototype only: register eight independent columns inside EACH original
// 352/336/336 recurrence. Keep all three merges and the GEMV branch untouched.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
bool constant(Value value,int64_t expected) {
  auto op=value.getDefiningOp<arith::ConstantIndexOp>();return op && op.value()==expected;
}
bool bound(scf::ForOp op,int64_t upper,int64_t step) {
  return constant(op.getLowerBound(),0) && constant(op.getUpperBound(),upper) &&
      constant(op.getStep(),step) && op.getNumRegionIterArgs()==1;
}
struct Candidate {
  scf::ForOp head,m,k,n;
  tensor::ExtractSliceOp p,accum,v;
  vector::TransferReadOp pRead,vRead,accumRead;
  vector::BroadcastOp broadcast;
  math::FmaOp fma;
  vector::TransferWriteOp write;
  tensor::InsertSliceOp insert;
};
std::optional<Candidate> prove(scf::ForOp head,int64_t depth,int64_t offset) {
  if (!bound(head,16,1) || head.getBody()->getOperations().size()!=2) return {};
  auto m=dyn_cast<scf::ForOp>(head.getBody()->front());
  if (!m || !constant(m.getLowerBound(),0) || !constant(m.getStep(),1) ||
      m.getNumRegionIterArgs()!=1 || m.getBody()->getOperations().size()!=2 ||
      m.getInitArgs()[0]!=head.getRegionIterArgs()[0]) return {};
  auto k=dyn_cast<scf::ForOp>(m.getBody()->front());
  if (!k || !bound(k,depth,1) || k.getBody()->getOperations().size()!=3 ||
      k.getInitArgs()[0]!=m.getRegionIterArgs()[0]) return {};
  auto iterator=k.getBody()->begin();auto p=dyn_cast<tensor::ExtractSliceOp>(*iterator++);
  auto n=dyn_cast<scf::ForOp>(*iterator++);
  if (!p || !n || !bound(n,128,32) || n.getBody()->getOperations().size()!=10 ||
      n.getInitArgs()[0]!=k.getRegionIterArgs()[0] ||
      p.getType().getRank()!=0 || !p.getType().getElementType().isF32() ||
      p.getOffsets()!=ValueRange({head.getInductionVar(),m.getInductionVar(),k.getInductionVar()})) return {};
  auto probability=p.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!probability || probability.getType().getShape()!=ArrayRef<int64_t>({16,ShapedType::kDynamic,depth}) ||
      probability.getStaticOffsets()!=ArrayRef<int64_t>({0,0,offset}) ||
      probability.getStaticSizes()!=ArrayRef<int64_t>({16,ShapedType::kDynamic,depth}) ||
      dyn_cast<Value>(probability.getMixedSizes()[1])!=m.getUpperBound() ||
      probability.getStaticStrides()!=ArrayRef<int64_t>({1,1,1})) return {};
  Candidate c{head,m,k,n,p};
  for (Operation &op : n.getBody()->without_terminator()) {
    if (auto slice=dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (slice.getSource()==n.getRegionIterArgs()[0]) c.accum=slice;else c.v=slice;
    } else if (auto read=dyn_cast<vector::TransferReadOp>(op)) {
      if (read.getBase()==p.getResult()) c.pRead=read;
      else if (c.v && read.getBase()==c.v.getResult()) c.vRead=read;
      else if (c.accum && read.getBase()==c.accum.getResult()) c.accumRead=read;
      else return {};
    } else if (auto b=dyn_cast<vector::BroadcastOp>(op)) c.broadcast=b;
    else if (auto f=dyn_cast<math::FmaOp>(op)) c.fma=f;
    else if (auto w=dyn_cast<vector::TransferWriteOp>(op)) c.write=w;
    else if (auto insert=dyn_cast<tensor::InsertSliceOp>(op)) c.insert=insert;
    else return {};
  }
  if (!c.accum || !c.v || !c.pRead || !c.vRead || !c.accumRead ||
      !c.broadcast || !c.fma || !c.write || !c.insert) return {};
  auto values=c.v.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!values || values.getType().getShape()!=ArrayRef<int64_t>({16,depth,128}) ||
      values.getStaticOffsets()!=ArrayRef<int64_t>({0,offset,0}) ||
      values.getStaticSizes()!=ArrayRef<int64_t>({16,depth,128}) ||
      values.getStaticStrides()!=ArrayRef<int64_t>({1,1,1}) ||
      !values.getType().getElementType().isF32() ||
      c.v.getStaticStrides()!=ArrayRef<int64_t>({1,1,1}) ||
      c.accum.getStaticStrides()!=ArrayRef<int64_t>({1,1,1}) ||
      c.p.getStaticStrides()!=ArrayRef<int64_t>({1,1,1}) ||
      c.insert.getStaticStrides()!=ArrayRef<int64_t>({1,1,1}) ||
      c.insert.getStaticSizes()!=ArrayRef<int64_t>({1,1,32}) ||
      c.insert.getOffsets()!=c.accum.getOffsets() ||
      c.v.getStaticSizes()!=ArrayRef<int64_t>({1,1,32}) ||
      c.v.getOffsets()!=ValueRange({head.getInductionVar(),k.getInductionVar(),n.getInductionVar()}) ||
      c.accum.getStaticSizes()!=ArrayRef<int64_t>({1,1,32}) ||
      c.accum.getOffsets()!=ValueRange({head.getInductionVar(),m.getInductionVar(),n.getInductionVar()}) ||
      c.fma.getFastmath()!=arith::FastMathFlags::none ||
      c.fma->getOperand(0)!=c.broadcast.getResult() ||
      c.fma->getOperand(1)!=c.vRead.getResult() ||
      c.fma->getOperand(2)!=c.accumRead.getResult() ||
      c.broadcast.getSource()!=c.pRead.getResult() ||
      c.write.getVector()!=c.fma.getResult() || c.write.getBase()!=c.accum.getResult() ||
      c.insert.getSource()!=c.write.getResult() || c.insert.getDest()!=n.getRegionIterArgs()[0] ||
      cast<scf::YieldOp>(n.getBody()->getTerminator()).getOperand(0)!=c.insert.getResult() ||
      cast<scf::YieldOp>(k.getBody()->getTerminator()).getOperand(0)!=n.getResult(0) ||
      cast<scf::YieldOp>(m.getBody()->getTerminator()).getOperand(0)!=k.getResult(0) ||
      cast<scf::YieldOp>(head.getBody()->getTerminator()).getOperand(0)!=m.getResult(0)) return {};
  auto vector32=VectorType::get({32},Float32Type::get(head.getContext()));
  if (c.fma.getType()!=vector32 || c.vRead.getType()!=vector32 ||
      c.accumRead.getType()!=vector32 || c.broadcast.getType()!=vector32 ||
      c.pRead.getType()!=VectorType::get({},Float32Type::get(head.getContext())) ||
      c.pRead.getMask() || c.vRead.getMask() || c.accumRead.getMask() || c.write.getMask()) return {};
  auto identity=AffineMap::getMultiDimIdentityMap(1,head.getContext());
  if (c.vRead.getPermutationMap()!=identity || c.accumRead.getPermutationMap()!=identity ||
      c.vRead.getIndices().size()!=1 || c.accumRead.getIndices().size()!=1 ||
      !constant(c.vRead.getIndices()[0],0) || !constant(c.accumRead.getIndices()[0],0) ||
      c.write.getIndices().size()!=1 || !constant(c.write.getIndices()[0],0) ||
      !c.pRead.getIndices().empty()) return {};
  return c;
}
Operation *cloneWidth(OpBuilder &b, Operation *op, IRMapping &map, int64_t width) {
  Operation *copy=b.clone(*op,map);
  for (Value result : copy->getResults()) {
    if (auto tensor=dyn_cast<RankedTensorType>(result.getType())) {
      if (tensor.getShape()==ArrayRef<int64_t>({32}))
        result.setType(RankedTensorType::get({width},tensor.getElementType()));
    } else if (auto vector=dyn_cast<VectorType>(result.getType())) {
      if (vector.getShape()==ArrayRef<int64_t>({32}))
        result.setType(VectorType::get({width},vector.getElementType()));
    }
  }
  if (isa<tensor::ExtractSliceOp,tensor::InsertSliceOp>(copy)) {
    auto sizes=copy->getAttrOfType<DenseI64ArrayAttr>("static_sizes");
    SmallVector<int64_t> resized(sizes.asArrayRef());
    if (resized.back()==32) resized.back()=width;
    copy->setAttr("static_sizes",b.getDenseI64ArrayAttr(resized));
  }
  return copy;
}
void rewrite(Candidate c) {
  OpBuilder b(c.head);auto loc=c.head.getLoc();auto eight=arith::ConstantIndexOp::create(b,loc,8);
  auto head=scf::ForOp::create(b,loc,c.head.getLowerBound(),c.head.getUpperBound(),c.head.getStep(),c.head.getInitArgs());
  b.setInsertionPointToStart(head.getBody());
  auto m=scf::ForOp::create(b,loc,c.m.getLowerBound(),c.m.getUpperBound(),c.m.getStep(),head.getRegionIterArgs());
  b.setInsertionPointToStart(m.getBody());
  auto n=scf::ForOp::create(b,loc,c.n.getLowerBound(),c.n.getUpperBound(),eight,m.getRegionIterArgs());
  b.setInsertionPointToStart(n.getBody());
  IRMapping map;map.map(c.head.getInductionVar(),head.getInductionVar());
  map.map(c.m.getInductionVar(),m.getInductionVar());map.map(c.n.getInductionVar(),n.getInductionVar());
  map.map(c.n.getRegionIterArgs()[0],n.getRegionIterArgs()[0]);
  cloneWidth(b,c.accum,map,8);cloneWidth(b,c.accumRead,map,8);
  auto k=scf::ForOp::create(b,loc,c.k.getLowerBound(),c.k.getUpperBound(),c.k.getStep(),ValueRange{map.lookup(c.accumRead.getResult())});
  b.setInsertionPointToStart(k.getBody());map.map(c.k.getInductionVar(),k.getInductionVar());
  map.map(c.accumRead.getResult(),k.getRegionIterArgs()[0]);cloneWidth(b,c.p,map,8);
  for (Operation &op : c.n.getBody()->without_terminator()) {
    if (&op==c.accum.getOperation() || &op==c.accumRead.getOperation() ||
        &op==c.write.getOperation() || &op==c.insert.getOperation()) continue;
    cloneWidth(b,&op,map,8);
  }
  b.setInsertionPointToEnd(k.getBody());scf::YieldOp::create(b,loc,map.lookup(c.fma.getResult()));
  b.setInsertionPointAfter(k);map.map(c.fma.getResult(),k.getResult(0));
  cloneWidth(b,c.write,map,8);cloneWidth(b,c.insert,map,8);
  b.setInsertionPointToEnd(n.getBody());scf::YieldOp::create(b,loc,map.lookup(c.insert.getResult()));
  b.setInsertionPointToEnd(m.getBody());scf::YieldOp::create(b,loc,n.getResults());
  b.setInsertionPointToEnd(head.getBody());scf::YieldOp::create(b,loc,m.getResults());
  c.head.getResult(0).replaceAllUsesWith(head.getResult(0));c.head.erase();
}
}
mlir::LogicalResult instrumentQwenLoopRegions(mlir::ModuleOp module) {
  using namespace mlir;SmallVector<Candidate> candidates;SmallVector<scf::IfOp> regions;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto order=function->getAttrOfType<StringAttr>("hexagon.fp32_prefill_order");
    if (function.getSymName()!="Qwen3DynamicBoundProbe" || function.isDeclaration() ||
        function.getNumArguments()!=65 || function.getNumResults()!=57 || !order ||
        order.getValue()!="qk-fma-pv-openblas-armv8-k1024") continue;
    for (Operation &op : function.getBody().front()) {
      auto conditional=dyn_cast<scf::IfOp>(op);
      if (!conditional) continue;
      auto attribute=conditional->getAttrOfType<StringAttr>("hexagon.fp32_prefill_dot_order");
      if (!attribute || attribute.getValue()!="pv-armv8-gemv1-gemm352-336-336-fma-ordered-add") continue;
      SmallVector<scf::ForOp> roots;
      for (Operation &nested : conditional.getElseRegion().front())
        if (auto loop=dyn_cast<scf::ForOp>(nested)) roots.push_back(loop);
      if (roots.size()!=4) {module.emitError("Original PV three partial recurrences and merge loop required");return failure();}
      for (unsigned i=0;i<3;++i) {
        auto candidate=prove(roots[i],i==0?352:336,i==0?0:i==1?352:688);
        if (!candidate) {module.emitError("Strict original per-partial PV register proof failed");return failure();}
        candidates.push_back(*candidate);
      }
      regions.push_back(conditional);
    }
  }
  if (regions.size()!=28 || candidates.size()!=84) {
    module.emitError("All28 original PV regions and all84 original partials required");return failure();
  }
  llvm::errs()<<"QWEN_DYNAMIC_PV_PREFLIGHT matched=28 recurrences=4 partial_merges=3\nQWEN_DYNAMIC_PV_FIRST_BEGIN\n";
  regions.front()->print(llvm::errs(),OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs()<<"\nQWEN_DYNAMIC_PV_FIRST_END\n";
  for (auto candidate : candidates) rewrite(candidate);
  if (failed(verify(module))) return failure();
  llvm::errs()<<"QWEN_DYNAMIC_PV_REGISTER verified=84 lanes=8 per_partial_k_order=original partial_merges=original no_fast_math=1\nQWEN_DYNAMIC_PV_REGISTER_FIRST_BEGIN\n";
  regions.front()->print(llvm::errs(),OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs()<<"\nQWEN_DYNAMIC_PV_REGISTER_FIRST_END\n";
#ifdef QWEN_DYNAMIC_PV_REGISTER_MODEL
  llvm::errs()<<"QWEN_DYNAMIC_PV_REGISTER_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE\n";
  return success();
#else
  llvm::errs()<<"QWEN_DYNAMIC_PV_STOP_NO_MODEL_CODE\n";
  return failure();
#endif
}
