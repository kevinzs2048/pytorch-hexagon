# Source-built versions of the two qualified Qwen compiler configurations.
# These targets share the Triton/MLIR libraries with the ordinary compiler.
# Link-time wrappers preserve the qualified pass order while eliminating the
# old recipe's dependency on object files from unrelated build directories.

function(qwen_add_compiler_profiles)
  if(NOT TARGET triton)
    message(FATAL_ERROR "Qwen compiler profiles require TRITON_BUILD_PYTHON_MODULE")
  endif()
  get_property(qwen_sources GLOBAL PROPERTY QWEN_PROFILE_SOURCE_DIR)
  get_target_property(binding_sources triton SOURCES)
  foreach(profile IN ITEMS q1 q2)
    set(target qwen-compiler-${profile})
    add_library(${target} SHARED ${binding_sources})
    foreach(property IN ITEMS INCLUDE_DIRECTORIES COMPILE_DEFINITIONS
        COMPILE_OPTIONS LINK_OPTIONS LINK_DIRECTORIES LINK_LIBRARIES)
      get_target_property(value triton ${property})
      if(value)
        set_property(TARGET ${target} PROPERTY ${property} "${value}")
      endif()
    endforeach()
    set_target_properties(${target} PROPERTIES OUTPUT_NAME triton
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/profiles/${profile}")
    target_link_options(${target} PRIVATE
      "-Wl,--wrap=_ZN4mlir13bufferization32createEmptyTensorEliminationPassEv")
  endforeach()

  target_sources(qwen-compiler-q1 PRIVATE
    ${qwen_sources}/ZeroTailPVRegister.cpp
    ${qwen_sources}/OrderedPVRegister.cpp
    ${qwen_sources}/OrderedPVLoopPreflight.cpp
    ${qwen_sources}/GQAKeyCompact.cpp
    ${qwen_sources}/GQARepeatSliceForward.cpp
    ${qwen_sources}/GQAValueForward.cpp
    ${qwen_sources}/GQAValueView.cpp
    ${qwen_sources}/EmptyTensorCSE.cpp
    ${qwen_sources}/ZeroTailQK.cpp
    ${qwen_sources}/ZeroTailQKModelHook.cpp)
  target_compile_definitions(qwen-compiler-q1 PRIVATE
    QWEN_ORDERED_PV_REGISTER_MODEL=1
    QWEN_ORDERED_PV_REGISTER_PREFLIGHT=1
    QWEN_ORDERED_PV_LOOP_PREFLIGHT=1
    QWEN_GQA_VALUE_VIEW=1 QWEN_GQA_VALUE_FORWARD=1 QWEN_GQA_KEY_COMPACT=1)
  set_property(SOURCE ${qwen_sources}/OrderedPVRegister.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS QWEN_Q1_ZERO_TAIL_MODEL_CANDIDATE=1)
  set_property(SOURCE ${qwen_sources}/GQAKeyCompact.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS QWEN_GQA_KEY_TILE_N32=1)
  set_property(SOURCE ${qwen_sources}/GQARepeatSliceForward.cpp
    ${qwen_sources}/GQAValueForward.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS QWEN_GQA_DEAD_REPEAT=1)
  target_link_options(qwen-compiler-q1 PRIVATE
    "-Wl,--wrap=_ZN4mlir13createCSEPassEv"
    "-Wl,--wrap=_Z25inspectQwenOrderedPVLoopsN4mlir8ModuleOpE")

  target_sources(qwen-compiler-q2 PRIVATE
    ${qwen_sources}/DynamicQKInvariantVector.cpp
    ${qwen_sources}/DynamicQKRegisterDispatch.cpp
    ${qwen_sources}/DynamicPVRegisterPreflight.cpp
    ${qwen_sources}/EmptyTensorCSE.cpp)
  target_compile_definitions(qwen-compiler-q2 PRIVATE
    QWEN_DIAGNOSTIC_LOOP_PROFILE=1 QWEN_DIAGNOSTIC_DYNAMIC_LOOP_PROFILE=1
    QWEN_DYNAMIC_QK_INVARIANT_VECTOR=1)
  set_property(SOURCE ${qwen_sources}/DynamicQKRegisterDispatch.cpp APPEND
    PROPERTY COMPILE_DEFINITIONS QWEN_Q2_IDENTICAL_TAIL=1)
  set_property(SOURCE ${qwen_sources}/DynamicPVRegisterPreflight.cpp APPEND
    PROPERTY COMPILE_DEFINITIONS QWEN_DYNAMIC_PV_REGISTER_MODEL=1
    instrumentQwenLoopRegions=retainOriginalDynamicPV)
endfunction()

set_property(GLOBAL PROPERTY QWEN_PROFILE_SOURCE_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../buffer_reuse")
# Triton's Python binding target is declared after its backend plugins.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL qwen_add_compiler_profiles)
