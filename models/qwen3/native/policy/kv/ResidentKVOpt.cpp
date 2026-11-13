// SPDX-License-Identifier: BSD-3-Clause
// Opt-in ABI adaptation for a session-owned token-major KV allocation.
// The caller reserves capacity before entry and binds disjoint KV planes.
// Direct cache results become new KV deltas; result-to-plane metadata records
// this explicit ABI change for the caller that reconstructs Python cache views.
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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace {
struct Candidate {
  func::FuncOp function;
  BlockArgument past;
  tensor::ConcatOp concat;
  int64_t resultIndex;
};

bool kvType(Type type) {
  auto t = dyn_cast<RankedTensorType>(type);
  return t && !t.getEncoding() && t.getElementType().isF32() && t.getRank() == 4 &&
         t.getDimSize(0) == 1 && t.getDimSize(1) == 8 && t.getDimSize(3) == 128;
}

struct ResidentKVPass : PassWrapper<ResidentKVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResidentKVPass)
  ResidentKVPass() = default;
  ResidentKVPass(const ResidentKVPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "qwen-resident-kv"; }
  StringRef getDescription() const final { return "Append only new KV into caller-reserved storage"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    cf::ControlFlowDialect, memref::MemRefDialect, tensor::TensorDialect>();
  }
  Option<unsigned> expected{*this, "expected-count", llvm::cl::init(56)};
  void runOnOperation() override {
    SmallVector<Candidate> candidates;
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (!function->hasAttr("hexagon.resident_kv")) continue;
      if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody()) ||
          !isa<func::ReturnOp>(function.front().getTerminator())) {
        function.emitError("resident KV requires one straight-line function block");
        signalPassFailure(); return;
      }
      llvm::DenseSet<int64_t> planes;
      for (auto arg : function.getArguments()) {
        auto plane = function.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "hexagon.kv_plane");
        if (!plane) continue;
        tensor::ConcatOp concat;
        bool valid = kvType(arg.getType()) && plane.getInt() >= 0 && plane.getInt() < 128 &&
                     planes.insert(plane.getInt()).second;
        for (Operation *user : arg.getUsers()) {
          if (isa<tensor::DimOp>(user)) continue;
          auto cat = dyn_cast<tensor::ConcatOp>(user);
          if (concat || !cat || cat.getDim() != 2 || cat.getInputs().size() != 2 ||
              cat.getInputs()[0] != arg || !kvType(cat.getInputs()[1].getType()) ||
              !kvType(cat.getType())) { valid = false; break; }
          concat = cat;
        }
        int64_t resultIndex = -1;
        if (concat)
          for (OpOperand &use : concat.getResult().getUses())
            if (isa<func::ReturnOp>(use.getOwner())) {
              if (resultIndex != -1) valid = false;
              resultIndex = use.getOperandNumber();
            }
        if (!valid || !concat) {
          function.emitError("resident KV requires one direct cache concat, at most one cache result, and unique plane bindings");
          signalPassFailure(); return;
        }
        candidates.push_back({function, arg, concat, resultIndex});
      }
    }
    if (candidates.size() != expected || !expected) {
      getOperation().emitError("resident KV append count differs from explicit ABI");
      signalPassFailure(); return;
    }
    IRRewriter b(&getContext());
    llvm::DenseMap<Operation *, Value> capacities;
    llvm::DenseMap<Operation *, SmallVector<int64_t>> deltaPlanes;
    for (auto candidate : candidates) {
      auto f = candidate.function;
      if (!capacities.count(f)) {
        unsigned index = f.getNumArguments();
        if (failed(f.insertArgument(index, b.getI64Type(), DictionaryAttr{}, f.getLoc()))) {
          signalPassFailure(); return;
        }
        capacities[f] = f.getArgument(index);
        deltaPlanes[f].assign(f.getNumResults(), -1);
        f->setAttr("hexagon.resident_kv_capacity_argument", b.getI64IntegerAttr(index));
      }
      auto layout = StridedLayoutAttr::get(&getContext(), 0, {ShapedType::kDynamic, 128, 1024, 1});
      f.setArgAttr(candidate.past.getArgNumber(), "bufferization.buffer_layout", layout);
      f.setArgAttr(candidate.past.getArgNumber(), "bufferization.writable", b.getBoolAttr(true));
      auto cat = candidate.concat;
      auto loc = cat.getLoc(); b.setInsertionPoint(cat);
      Value past = candidate.past, added = cat.getInputs()[1];
      if (candidate.resultIndex >= 0) {
        auto ret = cast<func::ReturnOp>(f.front().getTerminator());
        ret.setOperand(candidate.resultIndex, added);
        deltaPlanes[f][candidate.resultIndex] = f.getArgAttrOfType<IntegerAttr>(
            candidate.past.getArgNumber(), "hexagon.kv_plane").getInt();
      }
      Value oldLength = tensor::DimOp::create(b, loc, past, 2);
      Value newLength = tensor::DimOp::create(b, loc, added, 2);
      Value old64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), oldLength);
      Value new64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), newLength);
      Value capacity = capacities[f];
      Value bounded = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, old64, capacity);
      cf::AssertOp::create(b, loc, bounded, "resident KV history exceeds reserved capacity");
      Value remaining = arith::SubIOp::create(b, loc, capacity, old64);
      Value fits = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, new64, remaining);
      cf::AssertOp::create(b, loc, fits, "resident KV append exceeds reserved capacity");
      Value maxCapacity = arith::ConstantIntOp::create(b, loc, 524287, 64);
      Value addressable = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, capacity, maxCapacity);
      cf::AssertOp::create(b, loc, addressable, "resident KV plane exceeds DSP address range");
      Value total = arith::AddIOp::create(b, loc, oldLength, newLength);
      auto type = cast<RankedTensorType>(past.getType());
      auto pastMem = MemRefType::get(type.getShape(), b.getF32Type(), layout);
      Value buffer = bufferization::ToBufferOp::create(b, loc, pastMem, past);
      Value batchStride = arith::MulIOp::create(b, loc, total, arith::ConstantIndexOp::create(b, loc, 1024));
      auto resultType = cast<RankedTensorType>(cat.getType());
      OpFoldResult extent = resultType.isDynamicDim(2)
          ? OpFoldResult(total) : OpFoldResult(b.getIndexAttr(resultType.getDimSize(2)));
      SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), b.getIndexAttr(8), extent, b.getIndexAttr(128)};
      SmallVector<OpFoldResult> strides{batchStride, b.getIndexAttr(128), b.getIndexAttr(1024), b.getIndexAttr(1)};
      auto memType = MemRefType::get(resultType.getShape(), b.getF32Type(), layout);
      Value combined = memref::ReinterpretCastOp::create(b, loc, memType, buffer,
          b.getIndexAttr(0), sizes, strides);
      sizes[2] = newLength;
      SmallVector<OpFoldResult> offsets{b.getIndexAttr(0), b.getIndexAttr(0), oldLength, b.getIndexAttr(0)};
      SmallVector<OpFoldResult> steps(4, b.getIndexAttr(1));
      Value tail = memref::SubViewOp::create(b, loc, combined, offsets, sizes, steps);
      auto newType = cast<RankedTensorType>(added.getType());
      auto dynamicLayout = StridedLayoutAttr::get(&getContext(), ShapedType::kDynamic,
          {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic});
      auto newMem = MemRefType::get(newType.getShape(), b.getF32Type(), dynamicLayout);
      Value source = bufferization::ToBufferOp::create(b, loc, newMem, added, true);
      memref::CopyOp::create(b, loc, source, tail);
      Value result = bufferization::ToTensorOp::create(b, loc, resultType, combined,
          true /* restrict */, false /* writable: attention must not overwrite KV */);
      if ((f.getArgAttrOfType<IntegerAttr>(candidate.past.getArgNumber(),
                                         "hexagon.kv_plane").getInt() & 1) == 0) {
        // Keep persistent KV token-major for append-only updates. Attention's
        // K transpose otherwise fuses through GQA expansion and reads one float
        // every 4096 bytes. A head-major workspace bounds that strided working
        // set to one head. Materialize it before fusion; no FP arithmetic changes.
        SmallVector<Value> extents;
        if (resultType.isDynamicDim(2)) extents.push_back(total);
        Value resident = result;
        auto packed = tensor::GenerateOp::create(b, loc, resultType, extents,
            [&](OpBuilder &builder, Location location, ValueRange indices) {
              Value value = tensor::ExtractOp::create(builder, location, resident, indices);
              tensor::YieldOp::create(builder, location, value);
            });
        packed->setAttr("hexagon.resident_key_head_major", b.getUnitAttr());
        result = packed.getResult();
      }
      b.replaceOp(cat, result);
    }
    for (auto &entry : deltaPlanes) {
      auto f = cast<func::FuncOp>(entry.first);
      auto ret = cast<func::ReturnOp>(f.front().getTerminator());
      f.setType(b.getFunctionType(f.getArgumentTypes(), ret.getOperandTypes()));
      f->setAttr("hexagon.kv_delta_planes", b.getDenseI64ArrayAttr(entry.second));
    }
    llvm::errs() << "resident KV appends: " << candidates.size() << "\n";
  }
};
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect,
      bufferization::BufferizationDialect, cf::ControlFlowDialect,
      func::FuncDialect, linalg::LinalgDialect, math::MathDialect,
      memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
      vector::VectorDialect>();
  registerTransformsPasses();
  PassRegistration<ResidentKVPass>();
  return failed(MlirOptMain(argc, argv, "Resident KV ABI adapter\n", registry));
}
