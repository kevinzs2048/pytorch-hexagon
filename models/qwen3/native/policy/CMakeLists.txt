cmake_minimum_required(VERSION 3.20)
project(QwenNumericalPolicy LANGUAGES C CXX)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(HEXAGON_MLIR_SOURCE_DIR "" CACHE PATH "Hexagon-MLIR compiler source checkout")
if(NOT EXISTS "${HEXAGON_MLIR_SOURCE_DIR}/qcom_hexagon_backend/CMakeLists.txt")
  message(FATAL_ERROR "HEXAGON_MLIR_SOURCE_DIR must identify the compiler checkout")
endif()
set(QWEN_BACKEND "${HEXAGON_MLIR_SOURCE_DIR}/qcom_hexagon_backend")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_BINARY_DIR}/include"
                    "${QWEN_BACKEND}/include")
add_subdirectory(tablegen)
add_library(QwenTmTensor STATIC
  "${QWEN_BACKEND}/lib/Dialect/TmTensor/IR/TmTensorAttrs.cpp"
  "${QWEN_BACKEND}/lib/Dialect/TmTensor/IR/TmTensorTypes.cpp"
  "${QWEN_BACKEND}/lib/Dialect/TmTensor/IR/TmTensorOps.cpp"
  "${QWEN_BACKEND}/lib/Dialect/TmTensor/IR/TmTensorDialect.cpp")
add_dependencies(QwenTmTensor QwenTmTensorTableGen)
target_link_libraries(QwenTmTensor PUBLIC MLIRIR MLIRLinalgDialect MLIRLinalgUtils
  MLIRDestinationStyleOpInterface MLIRArithDialect MLIRMathDialect
  MLIRTensorDialect MLIRBufferizationDialect)
foreach(stage first sums)
  if(stage STREQUAL "first")
    set(pass_source "${CMAKE_CURRENT_SOURCE_DIR}/first/StabilizeFP32SumPass.cpp")
  else()
    set(pass_source "${CMAKE_CURRENT_SOURCE_DIR}/sums/StabilizeFP32SumPass.cpp")
  endif()
  add_executable(qwen-policy-${stage}-opt qwen-policy-opt.cpp "${pass_source}")
  target_include_directories(qwen-policy-${stage}-opt BEFORE PRIVATE
                             "${CMAKE_CURRENT_SOURCE_DIR}/${stage}/include")
  add_dependencies(qwen-policy-${stage}-opt QwenPolicyPassIncGen QwenTmTensorTableGen)
  target_compile_options(qwen-policy-${stage}-opt PRIVATE -g0 -ffunction-sections -fdata-sections)
  target_link_options(qwen-policy-${stage}-opt PRIVATE -Wl,--gc-sections -Wl,--strip-debug)
  target_link_libraries(qwen-policy-${stage}-opt PRIVATE QwenTmTensor MLIROptLib
    MLIRTransforms MLIRAffineDialect MLIRArithDialect MLIRControlFlowDialect
    MLIRFuncDialect MLIRLinalgDialect MLIRLinalgTransforms MLIRMathDialect
    MLIRSCFDialect MLIRTensorDialect)
endforeach()

# Explicit isolated experiment. The two accepted numerical targets are unchanged.
option(QWEN_BUILD_PACKED_INT8_EXPERIMENT "Build the opt-in exact integer matcher" OFF)
if(QWEN_BUILD_PACKED_INT8_EXPERIMENT)
  add_executable(qwen-packed-i8-opt packed/ExactInt8MatmulOpt.cpp)
  target_compile_options(qwen-packed-i8-opt PRIVATE -g0 -ffunction-sections -fdata-sections)
  target_link_options(qwen-packed-i8-opt PRIVATE -Wl,--gc-sections -Wl,--strip-debug)
  target_link_libraries(qwen-packed-i8-opt PRIVATE QwenTmTensor MLIROptLib MLIRTransforms
    MLIRAffineDialect MLIRArithDialect MLIRControlFlowDialect MLIRFuncDialect
    MLIRLinalgDialect MLIRMathDialect MLIRSCFDialect MLIRTensorDialect
    MLIRLLVMDialect MLIRMemRefDialect MLIRVectorDialect MLIRBufferizationDialect)
endif()

option(QWEN_BUILD_SILU_EFFECTS_EXPERIMENT "Build the scoped scalar effect contract tool" OFF)
if(QWEN_BUILD_SILU_EFFECTS_EXPERIMENT)
  add_executable(qwen-silu-effects-opt effects/ScalarHelperEffectsOpt.cpp)
  target_compile_options(qwen-silu-effects-opt PRIVATE -g0 -ffunction-sections -fdata-sections)
  target_link_options(qwen-silu-effects-opt PRIVATE -Wl,--gc-sections -Wl,--strip-debug)
  target_link_libraries(qwen-silu-effects-opt PRIVATE QwenTmTensor MLIROptLib MLIRTransforms
    MLIRAffineDialect MLIRArithDialect MLIRControlFlowDialect MLIRFuncDialect
    MLIRLinalgDialect MLIRMathDialect MLIRSCFDialect MLIRTensorDialect
    MLIRLLVMDialect MLIRMemRefDialect MLIRVectorDialect MLIRBufferizationDialect)
endif()
