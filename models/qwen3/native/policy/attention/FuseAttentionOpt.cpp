// SPDX-License-Identifier: BSD-3-Clause
// Opt-in replacement of the admitted Qwen attention policy by a runtime kernel.
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace mlir;
namespace {
Value peel(Value v, int stopHeads = 0) {
  for (unsigned i = 0; i < 16; ++i) {
    auto *op = v.getDefiningOp();
    auto type = dyn_cast<RankedTensorType>(v.getType());
    if (stopHeads && type && type.getRank() == 4 && type.getDimSize(0) == 1 &&
        type.getDimSize(1) == stopHeads && type.getDimSize(3) == 128 &&
        (!op || !op->hasAttr("hexagon.resident_key_head_major")))
      return v;
    if (!op)
      break;
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ReshapeOp,
            tensor::ExtractSliceOp>(op)) {
      v = op->getOperand(0);
      continue;
    }
    if (auto t = dyn_cast<linalg::TransposeOp>(op)) {
      v = t.getInput();
      continue;
    }
    if (auto g = dyn_cast<linalg::GenericOp>(op)) {
      if (!g.getDpsInputs().empty())
        break;
      auto y = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
      auto x = y && y.getNumOperands() == 1
                   ? y.getOperand(0).getDefiningOp<tensor::ExtractOp>()
                   : tensor::ExtractOp();
      if (!x)
        break;
      bool pure = true;
      for (auto &inner : g.getBody()->getOperations())
        if (!isa<linalg::IndexOp, arith::ConstantOp, arith::CmpIOp,
                 arith::SelectOp, tensor::ExtractOp, linalg::YieldOp>(inner))
          pure = false;
      if (!pure)
        break;
      v = x.getTensor();
      continue;
    }
    if (auto g = dyn_cast<tensor::GenerateOp>(op)) {
      if (!g->hasAttr("hexagon.resident_key_head_major"))
        break;
      auto y = cast<tensor::YieldOp>(g.getBody().front().getTerminator());
      auto x = y.getValue().getDefiningOp<tensor::ExtractOp>();
      if (!x || !llvm::equal(x.getIndices(), g.getBody().getArguments()))
        break;
      v = x.getTensor();
      continue;
    }
    break;
  }
  return v;
}
bool shape(Value v, ArrayRef<int64_t> dims) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.getElementType().isF32() && t.getShape() == dims;
}
template <class Op> bool elementwise(linalg::GenericOp g, unsigned inputs) {
  if (!g || g.getDpsInputs().size() != inputs || g.getNumResults() != 1 ||
      g.getNumReductionLoops() != 0 || g.getBody()->getOperations().size() != 2)
    return false;
  auto op = dyn_cast<Op>(g.getBody()->front());
  auto yield = cast<linalg::YieldOp>(g.getBody()->getTerminator());
  if (!op || yield.getOperand(0) != op->getResult(0))
    return false;
  for (unsigned i = 0; i < inputs; ++i)
    if (op->getOperand(i) != g.getBody()->getArgument(i))
      return false;
  return true;
}
struct Match {
  linalg::TransposeOp end;
  linalg::GenericOp exp;
  Value q, k, v, mask;
};
// Decode has a separate, explicitly admitted numerical contract: unfused QK
// below KV=4, sequential QK FMA thereafter, and one sequential PV FMA block.
bool orderedFMA(linalg::GenericOp op) {
  if (!op || op.getDpsInputs().size() != 2 || op.getDpsInits().size() != 1 ||
      op.getNumResults() != 1 || op.getBody()->getOperations().size() != 2)
    return false;
  auto fma = dyn_cast<math::FmaOp>(op.getBody()->front());
  auto yield = cast<linalg::YieldOp>(op.getBody()->getTerminator());
  if (!fma || fma.getFastmath() != arith::FastMathFlags::none ||
      yield.getOperand(0) != fma.getResult())
    return false;
  for (unsigned i = 0; i < 3; ++i)
    if (fma->getOperand(i) != op.getBody()->getArgument(i))
      return false;
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  auto value = fill ? fill.getInputs()[0].getDefiningOp<arith::ConstantOp>()
                    : arith::ConstantOp();
  auto zero = value ? dyn_cast<FloatAttr>(value.getValue()) : FloatAttr();
  if (!zero || !zero.getValue().isZero() || zero.getValue().isNegative())
    return false;
  if (fill.getBody()->getOperations().size() != 1 ||
      fill.getBody()->getTerminator()->getOperand(0) != fill.getBody()->getArgument(0))
    return false;
  auto *ctx = op.getContext();
  auto h = getAffineDimExpr(0, ctx), m = getAffineDimExpr(1, ctx);
  auto k = getAffineDimExpr(2, ctx), n = getAffineDimExpr(3, ctx);
  SmallVector<AffineMap> maps{AffineMap::get(4, 0, {h,m,k}, ctx),
      AffineMap::get(4, 0, {h,k,n}, ctx), AffineMap::get(4, 0, {h,m,n}, ctx)};
  SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::reduction,
      utils::IteratorType::parallel};
  return op.getIndexingMapsArray() == maps && op.getIteratorTypesArray() == iterators;
}
linalg::GenericOp decodeQK(scf::IfOp select) {
  if (!select->hasAttr("hexagon.fp32_effective_decode_order") ||
      select.getThenRegion().empty() || select.getElseRegion().empty() ||
      select.getThenRegion().front().getOperations().size() != 2 ||
      select.getElseRegion().front().getOperations().size() != 2)
    return {};
  auto plain = dyn_cast<linalg::BatchMatmulOp>(select.getThenRegion().front().front());
  auto fused = dyn_cast<linalg::GenericOp>(select.getElseRegion().front().front());
  auto compare = select.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!plain || !orderedFMA(fused) || !compare ||
      compare.getPredicate() != arith::CmpIPredicate::ult ||
      plain.getDpsInputs() != fused.getDpsInputs() ||
      plain.getDpsInits() != fused.getDpsInits())
    return {};
  auto *body = plain.getBody();
  if (body->getOperations().size() != 3)
    return {};
  auto multiply = dyn_cast<arith::MulFOp>(body->front());
  auto add = body->getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!multiply || !add || multiply.getFastmath() != arith::FastMathFlags::none ||
      add.getFastmath() != arith::FastMathFlags::none ||
      multiply.getLhs() != body->getArgument(0) || multiply.getRhs() != body->getArgument(1) ||
      !((add.getLhs() == multiply.getResult() && add.getRhs() == body->getArgument(2)) ||
        (add.getRhs() == multiply.getResult() && add.getLhs() == body->getArgument(2))))
    return {};
  auto length = compare.getLhs().getDefiningOp<tensor::DimOp>();
  auto threshold = compare.getRhs().getDefiningOp<arith::ConstantIndexOp>();
  constexpr int64_t D = ShapedType::kDynamic;
  if (!length || length.getSource() != fused.getDpsInputs()[1] ||
      length.getConstantIndex() != 2 || !threshold || threshold.value() != 4 ||
      !shape(fused.getDpsInputs()[0], {16,1,128}) ||
      !shape(fused.getDpsInputs()[1], {16,128,D}) ||
      select.getThenRegion().front().getTerminator()->getOperand(0) != plain.getResult(0) ||
      select.getElseRegion().front().getTerminator()->getOperand(0) != fused.getResult(0))
    return {};
  return fused;
}
bool match(Operation *root, linalg::GenericOp qk, Match &m, bool decode = false) {
  constexpr int64_t D = ShapedType::kDynamic;
  linalg::GenericOp pv;
  Operation *outer = nullptr;
  const int64_t query = decode ? 1 : D;
  for (auto *op = root->getNextNode(); op; op = op->getNextNode()) {
    if (op->hasAttr("hexagon.fp32_prefill_dot_order") ||
        (isa<scf::IfOp>(op) && op->hasAttr("hexagon.fp32_effective_decode_order")))
      return false;
    if (op->hasAttr("hexagon.fp32_effective_prefill_pv_order")) {
      if (outer)
        return false;
      outer = op;
      op->walk([&](linalg::GenericOp g) {
        if (g->hasAttr("hexagon.fp32_effective_pv_block"))
          pv = g;
      });
    }
    if (decode && op->hasAttr("hexagon.fp32_effective_decode_order") &&
        isa<linalg::GenericOp>(op)) {
      if (outer)
        return false;
      outer = op;
      pv = cast<linalg::GenericOp>(op);
      if (!orderedFMA(pv) || !shape(pv.getDpsInputs()[0], {16,1,D}) ||
          !shape(pv.getDpsInputs()[1], {16,D,128}))
        return false;
    }
    auto t = dyn_cast<linalg::TransposeOp>(op);
    if (t && shape(t.getResult()[0], {1, query, 16, 128}) &&
        t.getPermutation() == ArrayRef<int64_t>({0, 2, 1, 3})) {
      if (!outer || !pv || peel(t.getInput()).getDefiningOp() != outer)
        return false;
      m.end = t;
      break;
    }
  }
  if (!m.end || qk.getDpsInputs().size() != 2 || pv.getDpsInputs().size() != 2)
    return false;
  m.q = peel(qk.getDpsInputs()[0]);
  m.k = peel(qk.getDpsInputs()[1], 8);
  m.v = peel(pv.getDpsInputs()[1], 8);
  if (!shape(m.q, {1, 16, query, 128}) || !shape(m.k, {1, 8, D, 128}) ||
      !shape(m.v, {1, 8, D, 128}))
    return false;
  auto norm = peel(pv.getDpsInputs()[0]).getDefiningOp<linalg::GenericOp>();
  if (!elementwise<arith::MulFOp>(norm, 2) ||
      !norm->hasAttr("hexagon.fp32_arm_softmax_normalization"))
    return false;
  auto exp = norm.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  auto reciprocal = norm.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
  if (!exp || exp.getDpsInputs().size() != 1 ||
      exp.getBody()->getOperations().size() != 2 || !reciprocal ||
      !reciprocal->hasAttr("hexagon.fp32_arm_softmax_reciprocal") ||
      reciprocal.getDpsInputs().size() != 1)
    return false;
  auto call = dyn_cast<func::CallOp>(exp.getBody()->front());
  if (!call || call.getCallee() != "__hexagon_softmax_exp_compat")
    return false;
  m.exp = exp;
  auto sum = reciprocal.getDpsInputs()[0].getDefiningOp<tensor::GenerateOp>();
  if (!sum || !sum->hasAttr("hexagon.fp32_arm_softmax_sum"))
    return false;
  bool sumInput = true;
  sum.walk([&](tensor::ExtractOp x) {
    if (x.getTensor() != exp.getResult(0))
      sumInput = false;
  });
  if (!sumInput)
    return false;
  auto sub = exp.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!elementwise<arith::SubFOp>(sub, 2))
    return false;
  auto max = peel(sub.getDpsInputs()[1]).getDefiningOp<linalg::GenericOp>();
  if (!max || max.getDpsInputs().size() != 1 ||
      max.getDpsInputs()[0] != sub.getDpsInputs()[0] ||
      max.getNumReductionLoops() != 1)
    return false;
  auto add = sub.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!elementwise<arith::AddFOp>(add, 2))
    return false;
  m.mask = add.getDpsInputs()[1];
  if (!shape(m.mask, {1, 1, query, D}))
    return false;
  auto scale = add.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!elementwise<arith::MulFOp>(scale, 2) ||
      peel(scale.getDpsInputs()[0]) != root->getResult(0))
    return false;
  auto c = scale.getDpsInputs()[1].getDefiningOp<arith::ConstantOp>();
  auto attr =
      c ? dyn_cast<DenseFPElementsAttr>(c.getValue()) : DenseFPElementsAttr();
  if (!attr || !attr.isSplat() ||
      attr.getSplatValue<APFloat>().convertToFloat() != 0.08838834764831845f)
    return false;
  return true;
}
struct FusePass : PassWrapper<FusePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePass)
  FusePass() = default;
  FusePass(const FusePass &o) : PassWrapper(o) {}
  StringRef getArgument() const final { return "qwen-fused-attention"; }
  StringRef getDescription() const final {
    return "Outline admitted Qwen attention to an explicit runtime strategy";
  }
  Option<unsigned> expected{*this, "expected-count", llvm::cl::init(28)};
  Option<std::string> implementation{*this, "implementation",
      llvm::cl::init("fp32-ordered-workers-v1")};
  void runOnOperation() override {
    if (implementation != "fp32-ordered-workers-v1") {
      getOperation().emitError("unsupported attention implementation");
      return signalPassFailure();
    }
    SmallVector<Match> matches;
    bool valid = true;
    getOperation().walk([&](linalg::GenericOp op) {
      if (!op->hasAttr("hexagon.fp32_prefill_dot_order"))
        return;
      Match m;
      if (!match(op, op, m)) {
        op.emitError("unsupported fused attention dataflow");
        valid = false;
      } else
        matches.push_back(m);
    });
    getOperation().walk([&](scf::IfOp select) {
        if (!select->hasAttr("hexagon.fp32_effective_decode_order"))
          return;
        auto qk = decodeQK(select);
        Match m;
        if (!qk || !match(select, qk, m, true)) {
          select.emitError("unsupported ordered decode attention dataflow");
          valid = false;
        } else
          matches.push_back(m);
      });
    if (!valid || matches.size() != expected || !expected) {
      getOperation().emitError(
          "fused attention count differs from explicit admission");
      signalPassFailure();
      return;
    }
    SmallVector<tensor::GenerateOp> keyCopies;
    getOperation().walk([&](tensor::GenerateOp g) {
      if (!g->hasAttr("hexagon.resident_key_head_major"))
        return;
      auto y = cast<tensor::YieldOp>(g.getBody().front().getTerminator());
      auto x = y.getValue().getDefiningOp<tensor::ExtractOp>();
      if (!x || g.getBody().front().getOperations().size() != 2 ||
          !llvm::equal(x.getIndices(), g.getBody().getArguments()) ||
          x.getTensor().getType() != g.getResult().getType())
        valid = false;
      else
        keyCopies.push_back(g);
    });
    if (!valid) {
      getOperation().emitError("unexpected key workspace mapping");
      signalPassFailure();
      return;
    }
    IRRewriter b(&getContext());
    func::FuncOp callee;
    llvm::SmallPtrSet<Block *, 4> blocks;
    for (auto &m : matches) {
      m.exp->setAttr("qwen.obsolete_fused_exp", b.getUnitAttr());
      blocks.insert(m.end->getBlock());
      auto loc = m.end.getLoc();
      b.setInsertionPoint(m.end);
      SmallVector<Value> operands;
      for (Value v : {m.q, m.k, m.v, m.mask}) {
        auto t = cast<RankedTensorType>(v.getType());
        auto layout = StridedLayoutAttr::get(
            &getContext(), ShapedType::kDynamic,
            SmallVector<int64_t>(4, ShapedType::kDynamic));
        auto mt = MemRefType::get(t.getShape(), t.getElementType(), layout);
        operands.push_back(
            bufferization::ToBufferOp::create(b, loc, mt, v, true));
      }
      auto resultType = cast<RankedTensorType>(m.end.getResult()[0].getType());
      Value n = tensor::DimOp::create(b, loc, m.q, 2);
      auto outType = MemRefType::get(resultType.getShape(), b.getF32Type());
      SmallVector<Value> dynamicSizes;
      if (resultType.isDynamicDim(1))
        dynamicSizes.push_back(n);
      Value out = memref::AllocOp::create(b, loc, outType, dynamicSizes,
                                          b.getI64IntegerAttr(128));
      operands.push_back(out);
      auto type = b.getFunctionType(ValueRange(operands).getTypes(),
                                    TypeRange{b.getI32Type()});
      if (!callee) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(getOperation().getBody());
        callee = func::FuncOp::create(b, loc,
            "qwen_ordered_fp32_attention_v1", type);
        callee.setPrivate();
        callee->setAttr("llvm.emit_c_interface", b.getUnitAttr());
      } else if (callee.getFunctionType() != type) {
        signalPassFailure();
        return;
      }
      Value status =
          func::CallOp::create(b, loc, callee, operands).getResult(0);
      Value zero = arith::ConstantIntOp::create(b, loc, 0, 32);
      Value ok =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, status, zero);
      cf::AssertOp::create(b, loc, ok, "fused attention kernel failed");
      Value result = bufferization::ToTensorOp::create(b, loc, resultType, out,
                                                       true, false);
      b.replaceOp(m.end, result);
    }
    // The fused kernel consumes resident strides directly. Remaining shape
    // assertions must not keep the old full head-major key copy alive.
    for (auto g : keyCopies) {
      auto y = cast<tensor::YieldOp>(g.getBody().front().getTerminator());
      b.replaceOp(g,
                  y.getValue().getDefiningOp<tensor::ExtractOp>().getTensor());
    }
    // The admitted scalar exp helper is pure, but func.call conservatively
    // keeps its containing tensor generic alive. Remove only matched unused
    // exp generics, then ordinary dead tensor producers, preserving assertions.
    OpPassManager cleanup(ModuleOp::getOperationName());
    cleanup.addPass(createCanonicalizerPass());
    if (failed(runPipeline(cleanup, getOperation()))) {
      signalPassFailure();
      return;
    }
    bool changed;
    do {
      changed = false;
      for (Block *block : blocks)
        for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*block)))
          if (isOpTriviallyDead(&op) ||
              (op.hasAttr("qwen.obsolete_fused_exp") && op.use_empty())) {
            b.eraseOp(&op);
            changed = true;
          }
    } while (changed);
    bool retained = false;
    getOperation().walk([&](Operation *op) {
      if (op->hasAttr("hexagon.fp32_prefill_dot_order") ||
          op->hasAttr("hexagon.fp32_effective_decode_order") ||
          op->hasAttr("qwen.obsolete_fused_exp"))
        retained = true;
    });
    if (retained) {
      getOperation().emitError(
          "attention has extra observable results; old computation remains");
      signalPassFailure();
      return;
    }
    getOperation()->setAttr("qwen.attention_execution",
                            b.getStringAttr(implementation.getValue()));
    llvm::errs() << "fused Qwen attention sites: " << matches.size() << "\n";
  }
};
} // namespace
int main(int argc, char **argv) {
  DialectRegistry r;
  r.insert<affine::AffineDialect, arith::ArithDialect,
           bufferization::BufferizationDialect, cf::ControlFlowDialect,
           func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
           memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
           vector::VectorDialect>();
  registerTransformsPasses();
  PassRegistration<FusePass>();
  return failed(
      MlirOptMain(argc, argv, "Qwen fused attention opt-in lowering\n", r));
}
