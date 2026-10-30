// Only duplicate columns whose complete original FMA operands are identical.
// The original register rewrite and its ascending128-component recurrence stay
// untouched. Unknown extents/layouts/broadcast retain its complete old path.
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"

namespace {
mlir::FailureOr<mlir::scf::ForOp> rewriteIdenticalQKTail(mlir::ModuleOp module,Candidate c) {
  using namespace mlir;
  Value seed=c.head.getInitArgs()[0],keys=c.key.getSource(),flag=c.select.getCondition();
  auto score=dyn_cast<RankedTensorType>(seed.getType());
  if(!score || score.getShape()!=ArrayRef<int64_t>({16,ShapedType::kDynamic,ShapedType::kDynamic})
      || !score.getElementType().isF32() || !flag.getType().isInteger(1))return failure();
  DominanceInfo dominance(module);
  for(Value value:{keys,seed,flag,c.n.getUpperBound(),c.m.getUpperBound()})
    if(!dominance.properlyDominates(value,c.head))return failure();
  auto loc=c.head.getLoc();OpBuilder b(c.head);
  auto keyType=MemRefType::get({16,128,ShapedType::kDynamic},b.getF32Type());
  auto seedType=MemRefType::get({16,ShapedType::kDynamic,ShapedType::kDynamic},b.getF32Type());
  auto signature=b.getFunctionType({keyType,seedType,b.getI1Type()},{b.getIndexType()});
  auto helper=module.lookupSymbol<func::FuncOp>("qwen_q2_qk_positive_zero_extent");
  if(helper && (!helper.isDeclaration() || helper.getFunctionType()!=signature))return failure();
  if(!helper) {
    OpBuilder decl(module.getBodyRegion());
    helper=func::FuncOp::create(decl,loc,"qwen_q2_qk_positive_zero_extent",signature);
    helper.setPrivate();helper->setAttr("llvm.emit_c_interface",b.getUnitAttr());
  }
  auto keyBuffer=bufferization::ToBufferOp::create(b,loc,keyType,keys,true);
  auto seedBuffer=bufferization::ToBufferOp::create(b,loc,seedType,seed,true);
  auto extent=func::CallOp::create(b,loc,helper,ValueRange{keyBuffer,seedBuffer,flag}).getResult(0);
  auto eight=arith::ConstantIndexOp::create(b,loc,8);
  auto maximum=arith::ConstantIndexOp::create(b,loc,1024);
  auto scoreRows=tensor::DimOp::create(b,loc,seed,1);
  auto scoreColumns=tensor::DimOp::create(b,loc,seed,2);
  Value cap=c.n.getUpperBound();
  Value canonical=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,cap,maximum);
  auto rowsMatch=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,c.m.getUpperBound(),scoreRows);
  auto columnsMatch=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,cap,scoreColumns);
  canonical=arith::AndIOp::create(b,loc,canonical,rowsMatch);
  canonical=arith::AndIOp::create(b,loc,canonical,columnsMatch);
  auto shortPrefix=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ult,extent,cap);
  auto skip=arith::AndIOp::create(b,loc,canonical,shortPrefix);
  auto represented=arith::AddIOp::create(b,loc,extent,eight);
  auto upper=arith::SelectOp::create(b,loc,skip,represented,cap);
  auto head=rewrite(c);
  auto m=cast<scf::ForOp>(head.getBody()->front());
  auto n=cast<scf::ForOp>(m.getBody()->front());
  n.setUpperBound(upper);
  b.setInsertionPointAfter(n);
  auto choice=scf::IfOp::create(b,loc,n.getResultTypes(),skip,true);
  b.setInsertionPointToStart(&choice.getThenRegion().front());
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(1),b.getIndexAttr(1),b.getIndexAttr(8)};
  SmallVector<OpFoldResult> strides{b.getIndexAttr(1),b.getIndexAttr(1),b.getIndexAttr(1)};
  auto tile=tensor::ExtractSliceOp::create(b,loc,RankedTensorType::get({8},b.getF32Type()),
      n.getResult(0),SmallVector<OpFoldResult>{head.getInductionVar(),m.getInductionVar(),extent},sizes,strides);
  auto copies=scf::ForOp::create(b,loc,upper,cap,eight,n.getResults());
  b.setInsertionPointToStart(copies.getBody());
  auto inserted=tensor::InsertSliceOp::create(b,loc,tile,copies.getRegionIterArgs()[0],
      SmallVector<OpFoldResult>{head.getInductionVar(),m.getInductionVar(),copies.getInductionVar()},sizes,strides);
  scf::YieldOp::create(b,loc,inserted.getResult());
  b.setInsertionPointAfter(copies);scf::YieldOp::create(b,loc,copies.getResult(0));
  b.setInsertionPointToStart(&choice.getElseRegion().front());scf::YieldOp::create(b,loc,n.getResult(0));
  cast<scf::YieldOp>(m.getBody()->getTerminator()).setOperand(0,choice.getResult(0));
  // dispatch() completes the enclosing fallback region and then verifies the
  // whole module. Its else block is intentionally still empty at this point.
  return head;
}
}
