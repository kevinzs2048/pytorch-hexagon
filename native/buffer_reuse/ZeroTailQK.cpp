// SPDX-License-Identifier: BSD-3-Clause
// Preserve every original query operation and component FMA. Only duplicate
// completed columns after an integer proof of identical +0 keys AND seeds.
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace {
using namespace mlir;
bool constant(Value v, int64_t n) {
  auto c=v.getDefiningOp<arith::ConstantIndexOp>(); return c && c.value()==n;
}
bool loop(scf::ForOp v, int64_t end, int64_t step) {
  return v && constant(v.getLowerBound(),0) && constant(v.getUpperBound(),end)
      && constant(v.getStep(),step) && v.getNumRegionIterArgs()==1;
}
struct Proven { scf::ForOp component, columns; Value keys, flag, cap; };
std::optional<Proven> prove(scf::ForOp head) {
  if(!loop(head,16,1)||head.getBody()->getOperations().size()!=2) return {};
  auto score=dyn_cast<RankedTensorType>(head.getResult(0).getType());
  if(!score||score.getShape()!=ArrayRef<int64_t>({16,ShapedType::kDynamic})||!score.getElementType().isF32()) return {};
  auto component=dyn_cast<scf::ForOp>(head.getBody()->front());
  if(!loop(component,128,1)||component.getInitArgs()[0]!=head.getRegionIterArgs()[0]
      ||cast<scf::YieldOp>(head.getBody()->getTerminator()).getOperand(0)!=component.getResult(0)) return {};
  scf::ForOp columns;
  for(Operation &op:component.getBody()->without_terminator()) {
    if(auto found=dyn_cast<scf::ForOp>(op)) { if(columns) return {}; columns=found; }
    else if(!isa<tensor::ExtractSliceOp,tensor::ExtractOp,tensor::FromElementsOp,
                 arith::MulFOp,arith::AddFOp>(op)) return {};
  }
  if(!columns||!constant(columns.getLowerBound(),0)||!constant(columns.getStep(),128)
      ||columns.getNumRegionIterArgs()!=1||columns.getInitArgs()[0]!=component.getRegionIterArgs()[0]
      ||cast<scf::YieldOp>(component.getBody()->getTerminator()).getOperand(0)!=columns.getResult(0)
      ||columns.getBody()->getOperations().size()!=5) return {};
  auto cap=columns.getUpperBound().getDefiningOp<tensor::DimOp>();
  if(!cap||cap.getSource()!=head.getInitArgs()[0]||!constant(cap.getIndex(),1)) return {};
  affine::AffineMinOp width; tensor::ExtractSliceOp slice; linalg::GenericOp generic; tensor::InsertSliceOp insert;
  for(Operation &op:columns.getBody()->without_terminator()) {
    if(auto v=dyn_cast<affine::AffineMinOp>(op)) width=v;
    else if(auto v=dyn_cast<tensor::ExtractSliceOp>(op)) slice=v;
    else if(auto v=dyn_cast<linalg::GenericOp>(op)) generic=v;
    else if(auto v=dyn_cast<tensor::InsertSliceOp>(op)) insert=v;
    else return {};
  }
  if(!width||!slice||!generic||!insert) return {};
  auto context=head.getContext(); auto d=getAffineDimExpr(0,context), s=getAffineSymbolExpr(0,context);
  if(width.getAffineMap()!=AffineMap::get(1,1,{s-d,getAffineConstantExpr(128,context)},context)
      ||width.getOperands()!=ValueRange({columns.getInductionVar(),cap.getResult()})) return {};
  if(slice.getSource()!=columns.getRegionIterArgs()[0]
      ||slice.getOffsets()!=ValueRange({head.getInductionVar(),columns.getInductionVar()})
      ||slice.getSizes()!=ValueRange({width.getResult()})
      ||slice.getStaticSizes()!=ArrayRef<int64_t>({1,ShapedType::kDynamic})
      ||slice.getStaticStrides()!=ArrayRef<int64_t>({1,1})) return {};
  if(insert.getSource()!=generic.getResult(0)||insert.getDest()!=columns.getRegionIterArgs()[0]
      ||insert.getOffsets()!=slice.getOffsets()||insert.getSizes()!=slice.getSizes()
      ||insert.getStaticSizes()!=slice.getStaticSizes()||insert.getStaticStrides()!=slice.getStaticStrides()
      ||cast<scf::YieldOp>(columns.getBody()->getTerminator()).getOperand(0)!=insert.getResult()) return {};
  if(generic.getNumDpsInputs()!=3||generic.getNumDpsInits()!=1||generic.getNumResults()!=1
      ||generic.getDpsInputs()[0]!=slice.getResult()||generic.getDpsInits()[0]!=slice.getResult()
      ||generic.getIteratorTypesArray()!=SmallVector<utils::IteratorType>{utils::IteratorType::parallel}) return {};
  auto identity=AffineMap::get(1,0,d,context), scalar=AffineMap::get(1,0,{},context);
  if(generic.getIndexingMapsArray()!=SmallVector<AffineMap>{identity,scalar,scalar,identity}) return {};
  auto flag=generic.getDpsInputs()[1]; auto flagType=dyn_cast<RankedTensorType>(flag.getType());
  auto queryType=dyn_cast<RankedTensorType>(generic.getDpsInputs()[2].getType());
  if(!flagType||flagType.getRank()!=0||!flagType.getElementType().isInteger(1)
      ||!queryType||queryType.getRank()!=0||!queryType.getElementType().isF32()) return {};
  // The query is defined before the column loop; no column can change it.
  auto queryDef=generic.getDpsInputs()[2].getDefiningOp();
  if(!queryDef||queryDef->getBlock()!=component.getBody()||!queryDef->isBeforeInBlock(columns)) return {};
  math::FmaOp fma; tensor::ExtractOp key;
  for(Operation &op:generic.getBody()->without_terminator()) {
    if(auto v=dyn_cast<math::FmaOp>(op)) { if(fma) return {}; fma=v; }
    else if(auto v=dyn_cast<tensor::ExtractOp>(op)) { if(key) return {}; key=v; }
    else if(!isa<linalg::IndexOp,affine::AffineApplyOp,arith::SelectOp,
                 arith::DivUIOp,arith::MulIOp,arith::AddIOp>(op)) return {};
  }
  if(!fma||!key||fma.getFastmath()!=arith::FastMathFlags::none
      ||fma.getOperand(0)!=generic.getBody()->getArgument(2)||fma.getOperand(1)!=key.getResult()
      ||fma.getOperand(2)!=generic.getBody()->getArgument(0)
      ||cast<linalg::YieldOp>(generic.getBody()->getTerminator()).getOperand(0)!=fma.getResult()) return {};
  auto kt=dyn_cast<RankedTensorType>(key.getTensor().getType());
  if(!kt||kt.getShape()!=ArrayRef<int64_t>({8,128,ShapedType::kDynamic})||!kt.getElementType().isF32()
      ||key.getIndices().size()!=3||key.getIndices()[1]!=component.getInductionVar()) return {};
  auto add=key.getIndices()[0].getDefiningOp<arith::AddIOp>();
  auto mul=add?add.getLhs().getDefiningOp<arith::MulIOp>():arith::MulIOp();
  auto div=add?add.getRhs().getDefiningOp<arith::DivUIOp>():arith::DivUIOp();
  if(!mul||!constant(mul.getLhs(),0)||!constant(mul.getRhs(),8)
      ||!div||div.getLhs()!=head.getInductionVar()||!constant(div.getRhs(),2)) return {};
  auto select=key.getIndices()[2].getDefiningOp<arith::SelectOp>();
  auto column=select?select.getFalseValue().getDefiningOp<affine::AffineApplyOp>():affine::AffineApplyOp();
  if(!select||select.getCondition()!=generic.getBody()->getArgument(1)||!constant(select.getTrueValue(),0)
      ||!column||column.getAffineMap()!=AffineMap::get(1,1,d+s,context)
      ||column.getOperands().size()!=2||column.getOperands()[0]!=columns.getInductionVar()) return {};
  auto lane=column.getOperands()[1].getDefiningOp<linalg::IndexOp>();
  if(!lane||lane.getDim()!=0) return {};
  return Proven{component,columns,key.getTensor(),flag,cap.getResult()};
}
}

mlir::LogicalResult guardQwenPhysicalQK(mlir::ModuleOp module,mlir::scf::ForOp head) {
  using namespace mlir; auto proof=prove(head);
  if(!proof) { head.emitError("Original QK recurrence/address proof failed"); return failure(); }
  auto &p=*proof; auto loc=head.getLoc(); OpBuilder b(head);
  auto km=MemRefType::get({8,128,ShapedType::kDynamic},b.getF32Type());
  auto sm=MemRefType::get({16,ShapedType::kDynamic},b.getF32Type());
  auto signature=b.getFunctionType({km,sm,b.getI1Type()},{b.getIndexType()});
  auto helper=module.lookupSymbol<func::FuncOp>("qwen_qk_positive_zero_extent");
  if(helper&&(!helper.isDeclaration()||helper.getFunctionType()!=signature)) return failure();
  if(!helper) { OpBuilder decl(module.getBodyRegion());
    helper=func::FuncOp::create(decl,loc,"qwen_qk_positive_zero_extent",signature);
    helper.setPrivate();helper->setAttr("llvm.emit_c_interface",b.getUnitAttr()); }
  auto keys=bufferization::ToBufferOp::create(b,loc,km,p.keys,true);
  auto seed=bufferization::ToBufferOp::create(b,loc,sm,head.getInitArgs()[0],true);
  auto flag=tensor::ExtractOp::create(b,loc,p.flag,ValueRange{});
  auto extent=func::CallOp::create(b,loc,helper,ValueRange{keys,seed,flag}).getResult(0);
  auto c128=arith::ConstantIndexOp::create(b,loc,128),c1024=arith::ConstantIndexOp::create(b,loc,1024);
  auto canonical=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::eq,p.cap,c1024);
  auto shortPrefix=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ult,extent,p.cap);
  auto skip=arith::AndIOp::create(b,loc,canonical,shortPrefix);
  auto represented=arith::AddIOp::create(b,loc,extent,c128);
  auto upper=arith::SelectOp::create(b,loc,skip,represented,p.cap);
  p.columns.setUpperBound(upper);
  b.setInsertionPointAfter(p.component);
  auto answer=scf::IfOp::create(b,loc,head.getResultTypes(),skip,true);
  b.setInsertionPointToStart(&answer.getThenRegion().front());
  SmallVector<OpFoldResult> offsets{head.getInductionVar(),extent};
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(1),b.getIndexAttr(128)};
  SmallVector<OpFoldResult> strides{b.getIndexAttr(1),b.getIndexAttr(1)};
  auto tile=tensor::ExtractSliceOp::create(b,loc,RankedTensorType::get({128},b.getF32Type()),
      p.component.getResult(0),offsets,sizes,strides);
  auto copies=scf::ForOp::create(b,loc,upper,p.cap,c128,ValueRange{p.component.getResult(0)});
  b.setInsertionPointToStart(copies.getBody());
  auto inserted=tensor::InsertSliceOp::create(b,loc,tile,copies.getRegionIterArgs()[0],
      SmallVector<OpFoldResult>{head.getInductionVar(),copies.getInductionVar()},sizes,strides);
  scf::YieldOp::create(b,loc,inserted.getResult());
  b.setInsertionPointAfter(copies);scf::YieldOp::create(b,loc,copies.getResult(0));
  b.setInsertionPointToStart(&answer.getElseRegion().front());
  scf::YieldOp::create(b,loc,p.component.getResult(0));
  cast<scf::YieldOp>(head.getBody()->getTerminator()).setOperand(0,answer.getResult(0));
  return verify(module);
}

extern "C" __attribute__((visibility("default"))) int qwenQKPhysicalPreflight(const char *input,const char *output) {
  using namespace mlir; DialectRegistry registry;
  registry.insert<affine::AffineDialect,arith::ArithDialect,bufferization::BufferizationDialect,
      func::FuncDialect,linalg::LinalgDialect,math::MathDialect,memref::MemRefDialect,
      scf::SCFDialect,tensor::TensorDialect>();
  MLIRContext context(registry);context.loadAllAvailableDialects();
  auto module=parseSourceFile<ModuleOp>(input,&context);if(!module||failed(verify(*module))) return 2;
  auto function=module->lookupSymbol<func::FuncOp>("probe");
  if(!function||function.getNumArguments()!=7||function.getNumResults()!=1) return 3;
  SmallVector<scf::ForOp> heads;
  for(Operation &op:function.front()) if(auto loop=dyn_cast<scf::ForOp>(op)) heads.push_back(loop);
  if(heads.size()!=1||failed(guardQwenPhysicalQK(*module,heads[0]))) return 3;
  function.setSymName("probe_guarded");
  std::error_code error;llvm::raw_fd_ostream stream(output,error);if(error) return 4;
  module->print(stream,OpPrintingFlags().printGenericOpForm());return 0;
}
