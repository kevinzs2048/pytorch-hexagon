cmake_minimum_required(VERSION 3.20)
project(QwenDSPRuntime LANGUAGES NONE)

set(HEXAGON_TOOLS "" CACHE PATH "Hexagon Tools installation containing bin/")
set(HEXAGON_SDK "" CACHE PATH "Hexagon SDK installation")
set(HEXAGON_LLD "" CACHE FILEPATH "Qualified Hexagon ld.lld")
find_program(HEXAGON_CC hexagon-clang HINTS "${HEXAGON_TOOLS}/bin" NO_DEFAULT_PATH REQUIRED)
find_program(HEXAGON_CXX hexagon-clang++ HINTS "${HEXAGON_TOOLS}/bin" NO_DEFAULT_PATH REQUIRED)
find_program(HEXAGON_OBJCOPY hexagon-llvm-objcopy HINTS "${HEXAGON_TOOLS}/bin" NO_DEFAULT_PATH REQUIRED)
if(NOT EXISTS "${HEXAGON_LLD}")
  message(FATAL_ERROR "HEXAGON_LLD must identify the qualified Hexagon linker")
endif()

# Compile only production helpers. Diagnostic counters and probe kernels are
# separate tests and never enter the model's runtime library.
set(common -O2 -Wall -Wextra -Werror -fPIC -mv73 -mhvx -G0)
set(cxx ${common} -nostdlib++ -fno-exceptions -fno-rtti)
set(sdk_includes -I "${HEXAGON_SDK}/incs" -I "${HEXAGON_SDK}/incs/stddef")
function(qwen_runtime_object name source compiler)
  get_filename_component(source "${source}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(object "${CMAKE_CURRENT_BINARY_DIR}/${name}.o")
  add_custom_command(OUTPUT "${object}"
    COMMAND "${compiler}" ${ARGN} -MMD -MF "${object}.d"
      -c "${source}" -o "${object}"
    DEPENDS "${source}"
    DEPFILE "${object}.d" VERBATIM)
  set_property(GLOBAL APPEND PROPERTY QWEN_RUNTIME_OBJECTS "${object}")
endfunction()

qwen_runtime_object(packed-int8-runtime profile/packed_int8_matmul_runtime.c
  "${HEXAGON_CC}" ${common} -mhvx-length=128B ${sdk_includes})
qwen_runtime_object(packed-multi-i8-runtime profile/packed_int8_multi_matmul_runtime.c
  "${HEXAGON_CC}" ${common} -mhvx-length=128B ${sdk_includes})
qwen_runtime_object(guarded-copy-runtime profile/guarded_memref_copy.cpp
  "${HEXAGON_CXX}" ${cxx})
qwen_runtime_object(fixed512-memcpy profile/fixed512_memcpy.cpp
  "${HEXAGON_CXX}" ${cxx})
qwen_runtime_object(caller-byte-copy profile/caller_hvx_byte_copy.c
  "${HEXAGON_CC}" ${common} -fno-builtin -fno-strict-aliasing)
qwen_runtime_object(global-hvx-copy profile/global_hvx_byte_copy.cpp
  "${HEXAGON_CC}" -O2 -fPIC -fno-builtin -fno-strict-aliasing -mv73 -mhvx -G0)
qwen_runtime_object(q1-pv-guard profile/zero_tail_hvx_guard.c
  "${HEXAGON_CC}" -std=gnu11 ${common})
qwen_runtime_object(q1-qk-guard profile/qk_positive_zero_extent.c
  "${HEXAGON_CC}" -std=gnu11 ${common})
qwen_runtime_object(q2-qk-guard profile/q2_qk_positive_zero_extent_row_major.c
  "${HEXAGON_CC}" ${common} -fvisibility=hidden -fno-builtin -mhvx-length=128B ${sdk_includes})
qwen_runtime_object(q2-zero-guard profile/zero_tail_hvx_guard.c
  "${HEXAGON_CC}" ${common} -fvisibility=hidden -fno-builtin -mhvx-length=128B ${sdk_includes})

include("${CMAKE_CURRENT_SOURCE_DIR}/../arm_rope_trig_runtime/Sleef.cmake")
qwen_prepare_sleef(sleef_sources)
set(trig_options -O2 -ffp-contract=off -fPIC -mv73 -mhvx
  -Wno-unknown-pragmas -Wno-gcc-compat -I "${sleef_sources}")
qwen_runtime_object(trig_lane ../arm_rope_trig_runtime/trig_lane.c
  "${HEXAGON_CC}" ${trig_options})
qwen_runtime_object(rempitab "${sleef_sources}/rempitab.c"
  "${HEXAGON_CC}" ${trig_options})
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/trig_runtime.o"
  COMMAND "${HEXAGON_LLD}" -r "${CMAKE_CURRENT_BINARY_DIR}/trig_lane.o"
    "${CMAKE_CURRENT_BINARY_DIR}/rempitab.o" -o "${CMAKE_CURRENT_BINARY_DIR}/trig_runtime.o"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/trig_lane.o"
    "${CMAKE_CURRENT_BINARY_DIR}/rempitab.o" VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/preserved-memcpy.o"
  COMMAND "${HEXAGON_OBJCOPY}"
    --redefine-sym __wrap_memcpy=qwen_global_copy_preserved_memcpy
    "${CMAKE_CURRENT_BINARY_DIR}/fixed512-memcpy.o"
    "${CMAKE_CURRENT_BINARY_DIR}/preserved-memcpy.o"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/fixed512-memcpy.o" VERBATIM)
get_property(objects GLOBAL PROPERTY QWEN_RUNTIME_OBJECTS)
add_custom_target(qwen-dsp-runtime ALL DEPENDS ${objects}
  "${CMAKE_CURRENT_BINARY_DIR}/trig_runtime.o"
  "${CMAKE_CURRENT_BINARY_DIR}/preserved-memcpy.o")
