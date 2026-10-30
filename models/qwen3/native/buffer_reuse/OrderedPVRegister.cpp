// SPDX-License-Identifier: BSD-3-Clause
// Diagnostic only. Keep each lane's k=0..1023 FP32 FMA recurrence unchanged.
// Interchange independent columns with k, carrying eight FP32 accumulators.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#if defined(QWEN_Q1_ZERO_TAIL_PREFLIGHT) || defined(QWEN_Q1_ZERO_TAIL_MODEL_CANDIDATE)
mlir::LogicalResult guardQwenPhysicalPVRegister(mlir::ModuleOp,mlir::scf::ForOp);
#endif
namespace {
using namespace mlir;
bool bound(scf::ForOp op, int64_t upper, int64_t step) {
  auto a=op.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto b=op.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto c=op.getStep().getDefiningOp<arith::ConstantIndexOp>();
  return a && b && c && a.value()==0 && b.value()==upper && c.value()==step &&
      op.getNumRegionIterArgs()==1;
}
struct Candidate {
  scf::ForOp head,k,n;
  tensor::ExtractSliceOp p,accum,v;
  vector::TransferReadOp pRead,vRead,accumRead;
  vector::BroadcastOp broadcast;
  math::FmaOp fma;
  vector::TransferWriteOp write;
  tensor::InsertSliceOp insert;
};
std::optional<Candidate> prove(scf::ForOp head) {
  if (!bound(head,16,1) || head.getBody()->getOperations().size()!=2) return {};
  auto k=dyn_cast<scf::ForOp>(head.getBody()->front());
  if (!k || !bound(k,1024,1) || k.getBody()->getOperations().size()!=3) return {};
  auto it=k.getBody()->begin();
  auto p=dyn_cast<tensor::ExtractSliceOp>(*it++);
  auto n=dyn_cast<scf::ForOp>(*it++);
  if (!p || !n || !bound(n,128,32) || n.getBody()->getOperations().size()!=20) return {};
  auto ptype=dyn_cast<RankedTensorType>(p.getSource().getType());
  if (!ptype || ptype.getShape()!=ArrayRef<int64_t>({16,1024}) ||
      p.getType().getRank()!=0 || !p.getType().getElementType().isF32() ||
      p.getOffsets()!=ValueRange({head.getInductionVar(),k.getInductionVar()})) return {};
  Candidate c{head,k,n,p};
  for (Operation &op : n.getBody()->without_terminator()) {
    if (auto slice=dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (slice.getSource()==n.getRegionIterArgs()[0]) c.accum=slice;
      else c.v=slice;
    } else if (auto read=dyn_cast<vector::TransferReadOp>(op)) {
      if (read.getBase()==p.getResult()) c.pRead=read;
      else if (c.v && read.getBase()==c.v.getResult()) c.vRead=read;
      else if (c.accum && read.getBase()==c.accum.getResult()) c.accumRead=read;
      else return {};
    } else if (auto b=dyn_cast<vector::BroadcastOp>(op)) c.broadcast=b;
    else if (auto f=dyn_cast<math::FmaOp>(op)) c.fma=f;
    else if (auto w=dyn_cast<vector::TransferWriteOp>(op)) c.write=w;
    else if (auto insert=dyn_cast<tensor::InsertSliceOp>(op)) c.insert=insert;
    else if (!isa<arith::AddIOp,arith::DivUIOp,arith::RemUIOp,
                 arith::MulIOp,arith::SelectOp,tensor::ExtractOp>(op)) return {};
  }
  if (!c.accum || !c.v || !c.pRead || !c.vRead || !c.accumRead ||
      !c.broadcast || !c.fma || !c.write || !c.insert) return {};
  auto vtype=dyn_cast<RankedTensorType>(c.v.getSource().getType());
  if (!vtype || vtype.getShape()!=ArrayRef<int64_t>({1,8,1,ShapedType::kDynamic,128}) ||
      c.v.getStaticSizes()!=ArrayRef<int64_t>({1,1,1,1,32}) ||
      c.accum.getStaticSizes()!=ArrayRef<int64_t>({1,32}) ||
      c.accum.getOffsets()!=ValueRange({head.getInductionVar(),n.getInductionVar()}) ||
      c.fma.getFastmath()!=arith::FastMathFlags::none ||
      c.fma->getOperand(0)!=c.broadcast.getResult() ||
      c.fma->getOperand(1)!=c.vRead.getResult() ||
      c.fma->getOperand(2)!=c.accumRead.getResult() ||
      c.broadcast.getSource()!=c.pRead.getResult() ||
      c.write.getVector()!=c.fma.getResult() || c.write.getBase()!=c.accum.getResult() ||
      c.insert.getSource()!=c.write.getResult() || c.insert.getDest()!=n.getRegionIterArgs()[0] ||
      cast<scf::YieldOp>(n.getBody()->getTerminator()).getOperand(0)!=c.insert.getResult() ||
      cast<scf::YieldOp>(k.getBody()->getTerminator()).getOperand(0)!=n.getResult(0) ||
      cast<scf::YieldOp>(head.getBody()->getTerminator()).getOperand(0)!=k.getResult(0)) return {};
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
scf::ForOp rewrite(Candidate c) {
  OpBuilder b(c.head);
  auto loc=c.head.getLoc(); auto eight=arith::ConstantIndexOp::create(b,loc,8);
  auto head=scf::ForOp::create(b,loc,c.head.getLowerBound(),c.head.getUpperBound(),
      c.head.getStep(),c.head.getInitArgs());
  b.setInsertionPointToStart(head.getBody());
  auto n=scf::ForOp::create(b,loc,c.n.getLowerBound(),c.n.getUpperBound(),eight,
      head.getRegionIterArgs());
  b.setInsertionPointToStart(n.getBody());
  IRMapping map; map.map(c.head.getInductionVar(),head.getInductionVar());
  map.map(c.n.getInductionVar(),n.getInductionVar());
  map.map(c.n.getRegionIterArgs()[0],n.getRegionIterArgs()[0]);
  cloneWidth(b,c.accum,map,8); cloneWidth(b,c.accumRead,map,8);
  auto k=scf::ForOp::create(b,loc,c.k.getLowerBound(),c.k.getUpperBound(),
      c.k.getStep(),ValueRange{map.lookup(c.accumRead.getResult())});
  b.setInsertionPointToStart(k.getBody());
  map.map(c.k.getInductionVar(),k.getInductionVar());
  map.map(c.accumRead.getResult(),k.getRegionIterArgs()[0]);
  cloneWidth(b,c.p,map,8);
  for (Operation &op : c.n.getBody()->without_terminator()) {
    if (&op==c.accum.getOperation() || &op==c.accumRead.getOperation() ||
        &op==c.write.getOperation() || &op==c.insert.getOperation()) continue;
    cloneWidth(b,&op,map,8);
  }
  b.setInsertionPointToEnd(k.getBody());
  scf::YieldOp::create(b,loc,map.lookup(c.fma.getResult()));
  b.setInsertionPointAfter(k);
  map.map(c.fma.getResult(),k.getResult(0));
  cloneWidth(b,c.write,map,8); cloneWidth(b,c.insert,map,8);
  b.setInsertionPointToEnd(n.getBody());
  scf::YieldOp::create(b,loc,map.lookup(c.insert.getResult()));
  b.setInsertionPointToEnd(head.getBody()); scf::YieldOp::create(b,loc,n.getResults());
  c.head.getResult(0).replaceAllUsesWith(head.getResult(0)); c.head.erase();
  return head;
}
}
mlir::LogicalResult diagnoseQwenOrderedPVRegister(mlir::ModuleOp module,
    llvm::ArrayRef<mlir::scf::ForOp> loops) {
  using namespace mlir;
  SmallVector<Candidate> candidates;
  for (auto loop : loops) {
    auto c=prove(loop);
    if (!c) { module.emitError("PV register strict original loop proof failed"); return failure(); }
    candidates.push_back(*c);
  }
  if (candidates.size()!=28) return failure();
  // Publish actual captured constants needed to reconstruct a physical GQA
  // fixture; do not infer addresses from standalone contiguous PV layouts.
  candidates.front().head.walk([&](Operation *op) {
    if (isa<arith::DivUIOp,arith::RemUIOp,arith::MulIOp>(op)) {
      auto c=op->getOperand(1).getDefiningOp<arith::ConstantIndexOp>();
      if (c) llvm::errs()<<"QWEN_Q1_PV_ADDRESS_CONSTANT operation="
          <<op->getName()<<" rhs="<<c.value()<<"\n";
    }
  });
  scf::ForOp first;
  for (auto c : candidates) {
    auto rewritten=rewrite(c);
#if defined(QWEN_Q1_ZERO_TAIL_PREFLIGHT) || defined(QWEN_Q1_ZERO_TAIL_MODEL_CANDIDATE)
    if(failed(guardQwenPhysicalPVRegister(module,rewritten)))return failure();
#endif
    if (!first) first=rewritten;
  }
  if (failed(verify(module))) return failure();
#if defined(QWEN_Q1_ZERO_TAIL_PREFLIGHT) || defined(QWEN_Q1_ZERO_TAIL_MODEL_CANDIDATE)
  llvm::errs()<<"QWEN_Q1_ZERO_TAIL_RUNTIME_EXTENT verified=28 original_FMA_clones=3 read_only_buffers=1 model_ABI_unchanged=1\n";
  first->print(llvm::errs(),OpPrintingFlags().useLocalScope().printGenericOpForm());
#ifdef QWEN_Q1_ZERO_TAIL_PREFLIGHT
  llvm::errs()<<"\nQWEN_Q1_ZERO_TAIL_PREFLIGHT_STOP_BEFORE_MODEL_CODEGEN\n";
  return failure();
#else
  llvm::errs()<<"\nQWEN_Q1_ZERO_TAIL_MODEL_CANDIDATE_CONTINUE_NOT_ACCEPTED\n";
#endif
#endif
  llvm::errs()<<"QWEN_Q1_PV_REGISTER verified=28 lanes=8 k_order=0:1024:1 no_fast_math=1\n"
      <<"QWEN_Q1_PV_REGISTER_FIRST_BEGIN\n";
  first->print(llvm::errs(),OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs()<<"\nQWEN_Q1_PV_REGISTER_FIRST_END\n";
#ifdef QWEN_ORDERED_PV_REGISTER_MODEL
  llvm::errs()<<"QWEN_Q1_PV_REGISTER_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE\n";
#else
  llvm::errs()<<"QWEN_Q1_PV_REGISTER_STOP_NO_MODEL_CODE\n";
#endif
  return success();
}
