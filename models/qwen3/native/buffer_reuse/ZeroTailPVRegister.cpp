// SPDX-License-Identifier: BSD-3-Clause
// Integer runtime extent, exact original FMA/address clones, original fallback.
// This pass is admitted only for the proved physical q1 register8 recurrence.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <optional>
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
bool constant(Value value,int64_t expected){
  auto c=value.getDefiningOp<arith::ConstantIndexOp>();return c&&c.value()==expected;
}
bool loop(scf::ForOp op,int64_t upper,int64_t step){
  return op&&constant(op.getLowerBound(),0)&&constant(op.getUpperBound(),upper)
      &&constant(op.getStep(),step)&&op.getNumRegionIterArgs()==1;
}
bool plusZero(Value value,Value variable){
  auto add=value.getDefiningOp<arith::AddIOp>();
  return add&&constant(add.getLhs(),0)&&add.getRhs()==variable;
}
bool headAddress(Value value,Value head){
  auto add=value.getDefiningOp<arith::AddIOp>();if(!add)return false;
  auto mul=add.getLhs().getDefiningOp<arith::MulIOp>();
  auto div=add.getRhs().getDefiningOp<arith::DivUIOp>();
  if(!mul||!div||!constant(mul.getRhs(),8)||!constant(div.getRhs(),2))return false;
  auto outer=mul.getLhs().getDefiningOp<arith::DivUIOp>();
  auto remainder=div.getLhs().getDefiningOp<arith::RemUIOp>();
  return outer&&remainder&&constant(outer.getRhs(),16)&&constant(remainder.getRhs(),16)
      &&outer.getLhs()==remainder.getLhs()&&plusZero(outer.getLhs(),head);
}
struct Proven {scf::ForOp n,k;Value p,v,flag;};
std::optional<Proven> prove(scf::ForOp head){
  if(!loop(head,16,1)||head.getBody()->getOperations().size()!=2)return {};
  auto n=dyn_cast<scf::ForOp>(head.getBody()->front());
  if(!loop(n,128,8)||n.getBody()->getOperations().size()!=6)return {};
  scf::ForOp k;for(Operation &op:n.getBody()->without_terminator())
    if(auto candidate=dyn_cast<scf::ForOp>(op)){if(k)return {};k=candidate;}
  if(!loop(k,1024,1)||k.getBody()->getOperations().size()!=17)return {};
  auto type=dyn_cast<VectorType>(k.getRegionIterArgs()[0].getType());
  if(!type||type.getShape()!=ArrayRef<int64_t>({8})||!type.getElementType().isF32())return {};
  math::FmaOp fma;
  for(Operation &op:k.getBody()->without_terminator()){
    if(auto f=dyn_cast<math::FmaOp>(op)){if(fma)return {};fma=f;}
    else if(!isa<arith::AddIOp,arith::DivUIOp,arith::RemUIOp,arith::MulIOp,arith::SelectOp,
             tensor::ExtractOp,tensor::ExtractSliceOp,vector::TransferReadOp,vector::BroadcastOp>(op))return {};
  }
  if(!fma||fma.getFastmath()!=arith::FastMathFlags::none
      ||fma.getOperand(2)!=k.getRegionIterArgs()[0]
      ||cast<scf::YieldOp>(k.getBody()->getTerminator()).getOperand(0)!=fma.getResult())return {};
  auto broadcast=fma.getOperand(0).getDefiningOp<vector::BroadcastOp>();
  auto vr=fma.getOperand(1).getDefiningOp<vector::TransferReadOp>();
  if(!broadcast||!vr)return {};
  auto pr=broadcast.getSource().getDefiningOp<vector::TransferReadOp>();
  if(!pr)return {};
  auto ps=pr.getBase().getDefiningOp<tensor::ExtractSliceOp>();
  auto vs=vr.getBase().getDefiningOp<tensor::ExtractSliceOp>();
  if(!ps||!vs||ps.getOffsets()!=ValueRange({head.getInductionVar(),k.getInductionVar()})
      ||ps.getStaticSizes()!=ArrayRef<int64_t>({1,1})||ps.getStaticStrides()!=ArrayRef<int64_t>({1,1})
      ||vs.getStaticSizes()!=ArrayRef<int64_t>({1,1,1,1,8})
      ||vs.getStaticStrides()!=ArrayRef<int64_t>({1,1,1,1,1}))return {};
  auto pt=dyn_cast<RankedTensorType>(ps.getSource().getType());
  auto vt=dyn_cast<RankedTensorType>(vs.getSource().getType());
  if(!pt||!vt||pt.getShape()!=ArrayRef<int64_t>({16,1024})
      ||vt.getShape()!=ArrayRef<int64_t>({1,8,1,ShapedType::kDynamic,128})
      ||!pt.getElementType().isF32()||!vt.getElementType().isF32())return {};
  auto offsets=vs.getOffsets();if(offsets.size()!=5||!constant(offsets[0],0)
      ||!headAddress(offsets[1],head.getInductionVar())||!constant(offsets[2],0)
      ||!plusZero(offsets[4],n.getInductionVar()))return {};
  auto select=offsets[3].getDefiningOp<arith::SelectOp>();
  if(!select||!constant(select.getTrueValue(),0)||!plusZero(select.getFalseValue(),k.getInductionVar()))return {};
  auto flag=select.getCondition().getDefiningOp<tensor::ExtractOp>();
  auto ft=flag?dyn_cast<RankedTensorType>(flag.getTensor().getType()):RankedTensorType();
  if(!flag||!ft||ft.getRank()!=0||!ft.getElementType().isInteger(1))return {};
  return Proven{n,k,ps.getSource(),vs.getSource(),flag.getTensor()};
}
void cloneBody(OpBuilder &b,scf::ForOp original,Value index,Value accumulator){
  IRMapping map;map.map(original.getInductionVar(),index);
  map.map(original.getRegionIterArgs()[0],accumulator);
  for(Operation &op:original.getBody()->without_terminator())b.clone(op,map);
  auto yielded=cast<scf::YieldOp>(original.getBody()->getTerminator()).getOperand(0);
  scf::YieldOp::create(b,original.getLoc(),map.lookup(yielded));
}
}
mlir::LogicalResult guardQwenPhysicalPVRegisterWithCounters(mlir::ModuleOp module,mlir::scf::ForOp head,mlir::Value counters){
  using namespace mlir;auto proof=prove(head);
  if(!proof){head.emitError("zero-tail physical P/V address and strict register8 recurrence proof failed");return failure();}
  auto loc=head.getLoc();OpBuilder b(head);auto &c=*proof;
  auto pm=MemRefType::get({16,1024},b.getF32Type());
  auto vm=MemRefType::get({1,8,1,ShapedType::kDynamic,128},b.getF32Type());
  auto signature=b.getFunctionType({pm,vm,b.getI1Type()},{b.getIndexType()});
  StringRef name="qwen_hvx_positive_zero_extent";
  auto helper=module.lookupSymbol<func::FuncOp>(name);
  if(helper&&(!helper.isDeclaration()||helper.getFunctionType()!=signature))return failure();
  if(!helper){OpBuilder decl(module.getBodyRegion());helper=func::FuncOp::create(decl,loc,name,signature);
    helper.setPrivate();helper->setAttr("llvm.emit_c_interface",b.getUnitAttr());}
  auto pb=bufferization::ToBufferOp::create(b,loc,pm,c.p,true);
  auto vb=bufferization::ToBufferOp::create(b,loc,vm,c.v,true);
  auto flag=tensor::ExtractOp::create(b,loc,c.flag,ValueRange{});
  auto extent=func::CallOp::create(b,loc,helper,ValueRange{pb,vb,flag}).getResult(0);
  auto hasTail=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ult,extent,c.k.getUpperBound());
  SmallVector<OpOperand *> originalUsers;for(OpOperand &use:c.k.getResult(0).getUses())originalUsers.push_back(&use);
  c.k.setUpperBound(extent);b.setInsertionPointAfter(c.k);
  auto iv=VectorType::get({8},b.getI32Type());
  auto exp=arith::ConstantOp::create(b,loc,iv,DenseElementsAttr::get(iv,b.getI32IntegerAttr(0x7f800000)));
  auto bits=arith::BitcastOp::create(b,loc,iv,c.k.getResult(0));
  auto masked=arith::AndIOp::create(b,loc,bits,exp);
  auto finite=arith::CmpIOp::create(b,loc,arith::CmpIPredicate::ne,masked,exp);
  auto allFinite=vector::ReductionOp::create(b,loc,vector::CombiningKind::AND,finite);
  auto skip=arith::AndIOp::create(b,loc,hasTail,allFinite);
  auto answer=scf::IfOp::create(b,loc,TypeRange{c.k.getResult(0).getType()},skip,true);
  b.setInsertionPointToStart(&answer.getThenRegion().front());
  cloneBody(b,c.k,extent,c.k.getResult(0));
  b.setInsertionPointToStart(&answer.getElseRegion().front());
  // The original upper bound is 1024; extent can be 1024 to bypass the guard.
  auto upper=arith::ConstantIndexOp::create(b,loc,1024);
  auto remainder=scf::ForOp::create(b,loc,extent,upper,c.k.getStep(),ValueRange{c.k.getResult(0)});
  b.setInsertionPointToStart(remainder.getBody());cloneBody(b,c.k,remainder.getInductionVar(),remainder.getRegionIterArgs()[0]);
  b.setInsertionPointAfter(remainder);scf::YieldOp::create(b,loc,remainder.getResult(0));
  if(counters){
    b.setInsertionPointAfter(answer);
    auto one=arith::ConstantIndexOp::create(b,loc,1);
    auto full=arith::ConstantIndexOp::create(b,loc,1024);
    auto steps=arith::AddIOp::create(b,loc,extent,one);
    auto actual=arith::SelectOp::create(b,loc,skip,steps,full);
    auto count=arith::IndexCastOp::create(b,loc,b.getI32Type(),actual);
    auto guardbit=arith::ExtUIOp::create(b,loc,b.getI32Type(),skip);
    auto shift=arith::ConstantIntOp::create(b,loc,16,32);
    auto shifted=arith::ShLIOp::create(b,loc,guardbit,shift);
    auto record=arith::OrIOp::create(b,loc,count,shifted);
    auto eight=arith::ConstantIndexOp::create(b,loc,8);
    auto column=arith::DivUIOp::create(b,loc,c.n.getInductionVar(),eight);
    memref::StoreOp::create(b,loc,record,counters,ValueRange{head.getInductionVar(),column});
  }
  for(OpOperand *use:originalUsers)use->set(answer.getResult(0));
  return verify(module);
}
mlir::LogicalResult guardQwenPhysicalPVRegister(mlir::ModuleOp module,mlir::scf::ForOp head){
  return guardQwenPhysicalPVRegisterWithCounters(module,head,mlir::Value{});
}
// The retained original physical excerpt exercises the SAME C++ transform.
// No model compilation or target execution occurs through this entry.
extern "C" __attribute__((visibility("default"))) int qwenZeroTailPhysicalPreflight(const char *input,const char *output){
  using namespace mlir;DialectRegistry registry;
  registry.insert<arith::ArithDialect,bufferization::BufferizationDialect,func::FuncDialect,
      math::MathDialect,memref::MemRefDialect,scf::SCFDialect,tensor::TensorDialect,vector::VectorDialect>();
  MLIRContext context(registry);context.loadAllAvailableDialects();
  auto module=parseSourceFile<ModuleOp>(input,&context);if(!module||failed(verify(*module)))return 2;
  SmallVector<scf::ForOp> heads;module->walk([&](scf::ForOp op){if(loop(op,16,1))heads.push_back(op);});
  if(heads.size()!=1)return 3;
  auto function=heads[0]->getParentOfType<func::FuncOp>();
  if(!function||function.getNumArguments()!=4||function.getSymName()!="probe")return 3;
  auto loc=function.getLoc();auto hint=RankedTensorType::get({},IntegerType::get(&context,64));
  auto counters=MemRefType::get({16,16},IntegerType::get(&context,32));
  if(failed(function.insertArgument(4,hint,DictionaryAttr{},loc)))return 3;
  if(failed(function.insertArgument(5,counters,DictionaryAttr{},loc)))return 3;
  function.setSymName("probe_guarded");
  if(failed(guardQwenPhysicalPVRegisterWithCounters(*module,heads[0],function.getArgument(5))))return 3;
  std::error_code error;llvm::raw_fd_ostream stream(output,error);if(error)return 4;
  module->print(stream,OpPrintingFlags().printGenericOpForm());return 0;
}
