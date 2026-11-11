cmake_minimum_required(VERSION 3.20)
project(QwenExactInt8HMX LANGUAGES NONE)
set(HEXAGON_SDK "" CACHE PATH "Hexagon SDK")
set(HEXAGON_TOOLS "" CACHE PATH "Hexagon Tools")
set(HEXKL_DIR "" CACHE PATH "HexKL addon")
set(LLAMA_SOURCE "" CACHE PATH "Pinned llama.cpp source checkout")
set(TRITON_PACK_OBJECT "" CACHE FILEPATH "Object from strict-int8-pack")
set(TRITON_ACCUMULATE_OBJECT "" CACHE FILEPATH "Object from strict-int8-accumulate")
set(TRITON_DEQUANT_OBJECT "" CACHE FILEPATH "Optional object from strict-int8-dequant")
set(TRITON_DEQUANT_RESIDUAL_OBJECT "" CACHE FILEPATH
  "Optional object from strict-int8-dequant-residual")
set(TRITON_DEQUANT_PRODUCT_OBJECT "" CACHE FILEPATH
  "Optional object from strict-int8-dequant-product")
option(SHARE_HMX_RUNTIME "Share generated attention's HMX dispatcher and VTCM owner" OFF)
set(pin f818065d753700341b5355ce36ca676ecfb5d139)
execute_process(COMMAND git -C "${LLAMA_SOURCE}" rev-parse HEAD
  OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${LLAMA_SOURCE}" diff --quiet HEAD -- ggml/src ggml/include RESULT_VARIABLE dirty)
if(NOT revision STREQUAL pin OR NOT dirty EQUAL 0)
  message(FATAL_ERROR "INT8 HMX DMA requires unchanged llama.cpp ${pin}")
endif()
if(NOT EXISTS "${HEXKL_DIR}/include/hexkl_micro.h" OR
   NOT EXISTS "${HEXKL_DIR}/lib/hexagon_toolv19_v73/libhexkl_micro.a")
  message(FATAL_ERROR "HexKL v73 micro headers and library are required")
endif()
set(flags -O3 -fPIC -G0 -mv73 -mhvx -mhmx -mhvx-length=128B
  -include hmx_hexagon_protos.h
  -I${LLAMA_SOURCE}/ggml/src/ggml-hexagon/htp
  -I${HEXAGON_SDK}/incs -I${HEXAGON_SDK}/incs/stddef
  -I${HEXAGON_SDK}/rtos/qurt/computev73/include/qurt
  -I${HEXAGON_SDK}/rtos/qurt/computev73/include/posix
  -I${HEXKL_DIR}/include)
if(SHARE_HMX_RUNTIME)
  list(APPEND flags -DQWEN_SHARED_HMX_RUNTIME)
endif()
set(kinds PACK ACCUMULATE)
# Fused dequantizing output stages are optional and come as a pair; without
# them the library exposes only the original INT32 entry.
if(TRITON_DEQUANT_OBJECT OR TRITON_DEQUANT_RESIDUAL_OBJECT OR TRITON_DEQUANT_PRODUCT_OBJECT)
  if(NOT TRITON_DEQUANT_OBJECT OR NOT TRITON_DEQUANT_RESIDUAL_OBJECT OR
     NOT TRITON_DEQUANT_PRODUCT_OBJECT)
    message(FATAL_ERROR "All three fused INT8 dequant objects are required together")
  endif()
  list(APPEND kinds DEQUANT DEQUANT_RESIDUAL DEQUANT_PRODUCT)
  list(APPEND flags -DQWEN_INT8_FUSED_DEQUANT)
endif()
foreach(kind IN LISTS kinds)
  string(TOLOWER "${kind}" lower)
  string(REPLACE "_" "-" lower "${lower}")
  set(input "${TRITON_${kind}_OBJECT}")
  get_filename_component(directory "${input}" DIRECTORY)
  set(manifest "${directory}/manifest.json")
  file(READ "${manifest}" receipt)
  string(JSON profile GET "${receipt}" profile)
  string(JSON expected_hash GET "${receipt}" object_sha256)
  string(JSON lwp GET "${receipt}" lwp enabled)
  string(JSON row_count LENGTH "${receipt}" contract query_rows)
  string(JSON row_min GET "${receipt}" contract query_rows 0)
  string(JSON row_max GET "${receipt}" contract query_rows 1)
  file(SHA256 "${input}" actual_hash)
  if(NOT profile STREQUAL "strict-int8-${lower}" OR NOT expected_hash STREQUAL actual_hash OR lwp OR
     NOT row_count EQUAL 2 OR NOT row_min EQUAL 1 OR NOT row_max EQUAL 32)
    message(FATAL_ERROR "Invalid INT8 ${lower} object/manifest or instrumented kernel")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}" "${manifest}")
  set(object "${CMAKE_CURRENT_BINARY_DIR}/${lower}.o")
  add_custom_command(OUTPUT "${object}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-llvm-objcopy"
      --localize-symbol=memrefCopy --localize-symbol=__hexagon_mlir_expf
      --localize-symbol=_Assert --localize-symbol=aligned_alloc
      --localize-symbol=roundevenf "${input}" "${object}"
    DEPENDS "${input}" "${manifest}" VERBATIM)
  list(APPEND objects "${object}")
endforeach()
foreach(source IN ITEMS hmx_prepacked.cpp hmx_dma.c ../profile/prepacked_int8_matmul_runtime.c)
  get_filename_component(name "${source}" NAME_WE)
  set(extra)
  set(compiler "${HEXAGON_TOOLS}/bin/hexagon-clang")
  if(source MATCHES "\\.cpp$")
    set(compiler "${HEXAGON_TOOLS}/bin/hexagon-clang++")
    list(APPEND extra -std=c++17 -fno-exceptions -fno-rtti)
  elseif(name STREQUAL "prepacked_int8_matmul_runtime")
    list(APPEND extra -DHEXAGON_PREPACK_LLAMA_WORKERS
      -D_mlir_ciface___hexagon_i8_prepacked_v1_matmul=qwen_hvx_matmul)
  endif()
  set(object "${CMAKE_CURRENT_BINARY_DIR}/${name}.o")
  add_custom_command(OUTPUT "${object}"
    COMMAND "${compiler}" ${flags} ${extra} -MMD -MF "${object}.d"
      -c "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}" DEPFILE "${object}.d" VERBATIM)
  list(APPEND objects "${object}")
endforeach()
# worker-pool.c is shared with the existing model runtime; do not duplicate it.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libqwen_int8_prefill.a"
  COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs
    "${CMAKE_CURRENT_BINARY_DIR}/libqwen_int8_prefill.a" ${objects}
  DEPENDS ${objects} VERBATIM)
add_custom_target(qwen-int8-prefill-static
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libqwen_int8_prefill.a")
