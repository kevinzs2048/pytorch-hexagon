// SPDX-License-Identifier: BSD-3-Clause
// Isolated, opt-in integer contraction matcher; no floating point rewrites.
#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace {
struct PreservedOperation {
  Operation *operation;
  DictionaryAttr attributes;
  Attribute properties;
  SmallVector<Value> operands;
  SmallVector<Type> resultTypes;
  SmallVector<Block *> successors;
  SmallVector<Value> blockArguments;
};
bool admitted(linalg::MatmulOp op, bool multiRow) {
  if (!op.hasPureTensorSemantics() || op->hasAttr("library_call") ||
      op.getNumDpsInputs()!=2 || op.getNumDpsInits()!=1 || op.getNumResults()!=1)
    return false;
  for (NamedAttribute attr:op->getAttrs())
    if (attr.getName()!="cast" && attr.getName()!="indexing_maps" &&
        attr.getName()!="operandSegmentSizes") return false;
  if (auto castAttr=op->getAttrOfType<linalg::TypeFnAttr>("cast"))
    if (castAttr.getValue()!=linalg::TypeFn::cast_signed) return false;
  auto a=dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto b=dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
  auto c=dyn_cast<RankedTensorType>(op.getDpsInits()[0].getType());
  if (!a || !b || !c || a.getRank()!=2 || b.getRank()!=2 || c.getRank()!=2 ||
      !b.hasStaticShape() ||
      a.getEncoding() || b.getEncoding() || c.getEncoding() ||
      !a.getElementType().isSignlessInteger(8) || !b.getElementType().isSignlessInteger(8) ||
      !c.getElementType().isSignlessInteger(32) || op.getResult(0).getType()!=c ||
      a.getDimSize(1)<1 || a.getDimSize(1)>131071 ||
      a.getDimSize(1)!=b.getDimSize(0) || b.getDimSize(1)!=c.getDimSize(1) ||
      c.getDimSize(1)<128 || c.getDimSize(1)>151936 || c.getDimSize(1)%128!=0)
    return false;
  if (!multiRow) {
    if (!a.hasStaticShape() || !c.hasStaticShape() ||
        a.getDimSize(0)!=1 || c.getDimSize(0)!=1) return false;
  } else {
    // Only M may be dynamic. Existing runtime contract checks M=1..32 and
    // matching A/C rows before any memory access; K and N remain static.
    int64_t rows=a.getDimSize(0);
    if (rows!=c.getDimSize(0) || (rows!=ShapedType::kDynamic && (rows<1 || rows>32)))
      return false;
  }
  auto fill=op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || fill->hasAttr("library_call") || !fill.hasPureTensorSemantics() ||
      !matchPattern(fill.getDpsInputs()[0],m_Zero())) return false;
  auto fillYield=dyn_cast<linalg::YieldOp>(fill.getBody()->getTerminator());
  if (fill.getBody()->getOperations().size()!=1 || !fillYield ||
      fillYield.getNumOperands()!=1 || fillYield.getOperand(0)!=fill.getBody()->getArgument(0)) return false;
  auto m=getAffineDimExpr(0,op.getContext()),n=getAffineDimExpr(1,op.getContext()),
       k=getAffineDimExpr(2,op.getContext());
  SmallVector<AffineMap> maps{AffineMap::get(3,0,{m,k},op.getContext()),
    AffineMap::get(3,0,{k,n},op.getContext()),AffineMap::get(3,0,{m,n},op.getContext())};
  if (op.getIndexingMapsArray()!=maps) return false;
  Block *body=op.getBody();
  if (body->getNumArguments()!=3 || body->getOperations().size()!=5) return false;
  auto yield=dyn_cast<linalg::YieldOp>(body->getTerminator());
  auto add=yield && yield.getNumOperands()==1 ?
      yield.getOperand(0).getDefiningOp<arith::AddIOp>() : arith::AddIOp();
  if (!add || add.getOverflowFlags()!=arith::IntegerOverflowFlags::none ||
      (add.getLhs()!=body->getArgument(2) && add.getRhs()!=body->getArgument(2))) return false;
  Value product=add.getLhs()==body->getArgument(2) ? add.getRhs() : add.getLhs();
  auto mul=product.getDefiningOp<arith::MulIOp>();
  if (!mul || mul.getOverflowFlags()!=arith::IntegerOverflowFlags::none) return false;
  auto left=mul.getLhs().getDefiningOp<arith::ExtSIOp>();
  auto right=mul.getRhs().getDefiningOp<arith::ExtSIOp>();
  return left && right && left.getIn()==body->getArgument(0) &&
         right.getIn()==body->getArgument(1);
}

struct ExactInt8MatmulPass : PassWrapper<ExactInt8MatmulPass,OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExactInt8MatmulPass)
  ExactInt8MatmulPass()=default;
  ExactInt8MatmulPass(const ExactInt8MatmulPass &other):PassWrapper(other) {}
  Option<unsigned> expectedCount{*this,"expected-count",llvm::cl::desc("Fail unless this many complete integer contractions match; zero permits partial experiments"),llvm::cl::init(0)};
  Option<bool> multiRow{*this,"multi-row",llvm::cl::desc("Opt-in existing bounded dynamic-row integer runtime; default retains original q1 admission"),llvm::cl::init(false)};
  StringRef getArgument() const final { return "qwen-packed-i8-matmul"; }
  StringRef getDescription() const final { return "Opt-in exact signed INT8 explicit calls; all surviving operations preserved"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect,memref::MemRefDialect,func::FuncDialect>();
  }
  void runOnOperation() final {
    auto module=getOperation();
    StringRef symbol=multiRow ? "__qwen_i8_multi_matmul" : "__qwen_i8_matmul";
    if (module.lookupSymbol(symbol)) {
      module.emitError("reserved packed integer runtime symbol already exists");
      return signalPassFailure();
    }
    SmallVector<linalg::MatmulOp> work;
    module.walk([&](linalg::MatmulOp op) {if (admitted(op,multiRow)) work.push_back(op);});
    if (expectedCount && work.size()!=expectedCount) {
      module.emitError()<<"packed integer coverage differs: expected "<<expectedCount.getValue()<<", admitted "<<work.size();
      return signalPassFailure();
    }
    // Preserve exact SSA dependencies and attributes of EVERY surviving op,
    // including integer quantization and all floating operations. Only the
    // selected contraction result may change to the equivalent new tensor.
    llvm::DenseSet<Operation *> removed;
    for (auto op:work) op->walk([&](Operation *nested) {removed.insert(nested);});
    SmallVector<PreservedOperation,0> original;
    module.walk([&](Operation *op) {
      if (removed.contains(op)) return;
      PreservedOperation snapshot{op,op->getAttrDictionary(),op->getPropertiesAsAttribute(),
        SmallVector<Value>(op->getOperands()),SmallVector<Type>(op->getResultTypes()),
        SmallVector<Block *>(op->getSuccessors()),{}};
      for (auto &region:op->getRegions())
        for (auto &block:region) llvm::append_range(snapshot.blockArguments,block.getArguments());
      original.push_back(std::move(snapshot));
    });
    llvm::DenseMap<Value,Value> replacement;
    IRRewriter rewriter(module.getContext());
    MemRefType da=MemRefType::get({ShapedType::kDynamic,ShapedType::kDynamic},rewriter.getI8Type(),
      StridedLayoutAttr::get(module.getContext(),ShapedType::kDynamic,
                            {ShapedType::kDynamic,ShapedType::kDynamic}));
    MemRefType dc=MemRefType::get({ShapedType::kDynamic,ShapedType::kDynamic},rewriter.getI32Type(),
      StridedLayoutAttr::get(module.getContext(),ShapedType::kDynamic,
                            {ShapedType::kDynamic,ShapedType::kDynamic}));
    if (!work.empty()) {
      rewriter.setInsertionPointToStart(module.getBody());
      auto callee=rewriter.create<func::FuncOp>(module.getLoc(),symbol,
        rewriter.getFunctionType({da,da,dc},{}));
      callee.setPrivate();
      callee->setAttr("llvm.emit_c_interface",rewriter.getUnitAttr());
    }
    for (auto op:work) {
      rewriter.setInsertionPoint(op);
      auto loc=op.getLoc();
      SmallVector<Value> buffers;
      for (Value input:op.getDpsInputs()) {
        auto tensor=cast<RankedTensorType>(input.getType());
        auto memref=MemRefType::get(tensor.getShape(),tensor.getElementType());
        buffers.push_back(rewriter.create<bufferization::ToBufferOp>(loc,memref,input,true));
      }
      auto tensor=cast<RankedTensorType>(op.getResult(0).getType());
      auto memref=MemRefType::get(tensor.getShape(),tensor.getElementType());
      // Never mutate the possibly shared original initializer tensor.
      SmallVector<Value> dimensions;
      if (tensor.isDynamicDim(0))
        dimensions.push_back(rewriter.create<tensor::DimOp>(loc,op.getDpsInits()[0],0));
      Value output=rewriter.create<memref::AllocOp>(loc,memref,dimensions);
      auto fill=op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
      rewriter.create<linalg::FillOp>(loc,fill.getDpsInputs(),ValueRange{output});
      Value left=rewriter.create<memref::CastOp>(loc,da,buffers[0]);
      Value right=rewriter.create<memref::CastOp>(loc,da,buffers[1]);
      Value resultBuffer=rewriter.create<memref::CastOp>(loc,dc,output);
      rewriter.create<func::CallOp>(loc,symbol,TypeRange{},ValueRange{left,right,resultBuffer});
      Value result=rewriter.create<bufferization::ToTensorOp>(loc,tensor,output,true,true);
      replacement[op.getResult(0)]=result;
      rewriter.replaceOp(op,result);
    }
    llvm::DenseSet<Operation *> preserved;
    for (auto &item:original) preserved.insert(item.operation);
    SmallVector<Operation *> actual;
    module.walk([&](Operation *op) {if (preserved.contains(op)) actual.push_back(op);});
    if (actual.size()!=original.size()) {
      module.emitError("unrelated original operation was erased");return signalPassFailure();
    }
    for (auto [index,item]:llvm::enumerate(original)) {
      Operation *op=actual[index];
      SmallVector<Value> expected;
      for (Value operand:item.operands) {
        auto found=replacement.find(operand);
        expected.push_back(found==replacement.end() ? operand : found->second);
      }
      SmallVector<Value> arguments;
      for (auto &region:op->getRegions())
        for (auto &block:region) llvm::append_range(arguments,block.getArguments());
      if (op!=item.operation || op->getAttrDictionary()!=item.attributes ||
          op->getPropertiesAsAttribute()!=item.properties ||
          SmallVector<Value>(op->getOperands())!=expected ||
          SmallVector<Type>(op->getResultTypes())!=item.resultTypes ||
          SmallVector<Block *>(op->getSuccessors())!=item.successors || arguments!=item.blockArguments) {
        module.emitError("unrelated original operation, SSA dependency or attribute changed");
        return signalPassFailure();
      }
    }
    llvm::errs()<<"qwen packed integer contractions: "<<work.size()<<"\n";
    llvm::errs()<<"qwen preserved original operations: "<<original.size()<<"\n";
  }
};
}

int main(int argc,char **argv) {
  DialectRegistry registry;
  registry.insert<affine::AffineDialect,arith::ArithDialect,cf::ControlFlowDialect,
      func::FuncDialect,linalg::LinalgDialect,LLVM::LLVMDialect,math::MathDialect,bufferization::BufferizationDialect,
      memref::MemRefDialect,scf::SCFDialect,tensor::TensorDialect,
      vector::VectorDialect,tm_tensor::TmTensorDialect>();
  registerTransformsPasses();
  PassRegistration<ExactInt8MatmulPass>();
  return asMainReturnCode(MlirOptMain(argc,argv,"Isolated Qwen exact integer optimizer\n",registry));
}
