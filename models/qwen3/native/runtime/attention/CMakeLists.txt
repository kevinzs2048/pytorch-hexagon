cmake_minimum_required(VERSION 3.20)
project(QwenOrderedAttention LANGUAGES NONE)
get_filename_component(common_native
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../hexagon_runtime/native" ABSOLUTE)
set(LLAMA_SOURCE "" CACHE PATH "Pinned llama.cpp source checkout")
set(HEXAGON_SDK "" CACHE PATH "Hexagon SDK")
set(HEXAGON_TOOLS "" CACHE PATH "Hexagon Tools")
set(pin f818065d753700341b5355ce36ca676ecfb5d139)
execute_process(COMMAND git -C "${LLAMA_SOURCE}" rev-parse HEAD OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT revision STREQUAL pin)
  message(FATAL_ERROR "Attention requires llama.cpp ${pin}")
endif()

execute_process(COMMAND git -C "${LLAMA_SOURCE}" diff --quiet HEAD -- ggml/src ggml/include RESULT_VARIABLE dirty)
if(NOT dirty EQUAL 0)
  message(FATAL_ERROR "The external kernel sources must match the pinned revision")
endif()
set(htp "${LLAMA_SOURCE}/ggml/src/ggml-hexagon/htp")
set(flags -O3 -fPIC -G0 -mv73 -mhvx -mhmx -mhvx-length=128B -fno-math-errno
  -include hmx_hexagon_protos.h
  -I${htp} -I${LLAMA_SOURCE}/ggml/src -I${LLAMA_SOURCE}/ggml/include
  -I${HEXAGON_SDK}/incs -I${HEXAGON_SDK}/incs/stddef
  -I${HEXAGON_SDK}/rtos/qurt/computev73/include/qurt
  -I${HEXAGON_SDK}/rtos/qurt/computev73/include/posix)
list(APPEND flags "-I${common_native}")
set(sources "${htp}/worker-pool.c")
foreach(source IN LISTS sources)
  get_filename_component(name "${source}" NAME_WE)
  set(obj "${CMAKE_CURRENT_BINARY_DIR}/${name}.o")
  add_custom_command(OUTPUT "${obj}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang" ${flags} -MMD -MF "${obj}.d"
      -c "${source}" -o "${obj}"
    DEPENDS "${source}" DEPFILE "${obj}.d" VERBATIM)
endforeach()
# Integer workers can be used with the existing FP32 attention graphs.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libqwen_workers.a"
  COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs
    "${CMAKE_CURRENT_BINARY_DIR}/libqwen_workers.a"
    "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o" VERBATIM)
add_custom_target(qwen-workers-static
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libqwen_workers.a")

# Strict FP32 attention uses only the worker pool, with no HMX/FP16 kernels.
set(ordered_object "${CMAKE_CURRENT_BINARY_DIR}/ordered_fp32_attention.o")
add_custom_command(OUTPUT "${ordered_object}"
  COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags}
    -std=c++17 -fno-vectorize -fno-slp-vectorize -ffp-contract=off
    -MMD -MF "${ordered_object}.d" -c
    "${CMAKE_CURRENT_SOURCE_DIR}/ordered_fp32_attention.cpp" -o "${ordered_object}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ordered_fp32_attention.cpp"
  DEPFILE "${ordered_object}.d" VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libqwen_ordered_fp32_attention.a"
  COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs
    "${CMAKE_CURRENT_BINARY_DIR}/libqwen_ordered_fp32_attention.a"
    "${ordered_object}" "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o"
  DEPENDS "${ordered_object}" "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o" VERBATIM)
add_custom_target(qwen-ordered-fp32-attention-static
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libqwen_ordered_fp32_attention.a")

# Explicit experimental build: Triton strict q1, accepted C FP32 prefill.
# No source compiler, SDK path or temporary benchmark object is inferred.
set(TRITON_DECODE_OBJECT "" CACHE FILEPATH "Object from kernels/compile.py strict-decode or qfloat-decode")
set(strict_prefill_objects)
set(strict_prefill_enabled FALSE)
foreach(stage IN ITEMS PACK QK SOFTMAX PV)
  set(TRITON_STRICT_PREFILL_${stage}_OBJECT "" CACHE FILEPATH "Object from strict-prefill stage")
  if(TRITON_STRICT_PREFILL_${stage}_OBJECT)
    set(strict_prefill_enabled TRUE)
  endif()
endforeach()
if(strict_prefill_enabled)
  if(NOT TRITON_DECODE_OBJECT OR TRITON_PREFILL_OBJECT)
    message(FATAL_ERROR "Strict prefill requires strict decode and excludes experimental FP16 prefill")
  endif()
  foreach(stage IN ITEMS PACK QK SOFTMAX PV)
    string(TOLOWER "${stage}" lower)
    set(input "${TRITON_STRICT_PREFILL_${stage}_OBJECT}")
    if(NOT EXISTS "${input}")
      message(FATAL_ERROR "All four strict prefill stages are required")
    endif()
    get_filename_component(directory "${input}" DIRECTORY)
    set(manifest "${directory}/manifest.json")
    file(READ "${manifest}" receipt)
    string(JSON profile GET "${receipt}" profile)
    string(JSON expected_hash GET "${receipt}" object_sha256)
    string(JSON lwp GET "${receipt}" lwp enabled)
    string(JSON scratch GET "${receipt}" scratch_fp32_elements)
    string(JSON query_min GET "${receipt}" contract query 0)
    string(JSON query_max GET "${receipt}" contract query 1)
    string(JSON kv_max GET "${receipt}" contract kv_length 1)
    string(JSON softmax_block GET "${receipt}" constants SOFTMAX_BLOCK)
    file(SHA256 "${input}" actual_hash)
    if(NOT profile STREQUAL "strict-prefill-${lower}" OR
        NOT expected_hash STREQUAL actual_hash OR lwp OR
        NOT scratch EQUAL 131104 OR NOT query_min EQUAL 16 OR
        NOT query_max EQUAL 32 OR NOT kv_max EQUAL 256 OR NOT softmax_block EQUAL 32)
      message(FATAL_ERROR "Strict prefill ${lower} identity/ABI mismatch or instrumented kernel")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}" "${manifest}")
    set(object "${CMAKE_CURRENT_BINARY_DIR}/strict_prefill_${lower}.o")
    add_custom_command(OUTPUT "${object}"
      COMMAND "${HEXAGON_TOOLS}/bin/hexagon-llvm-objcopy"
        --localize-symbol=memrefCopy --localize-symbol=__hexagon_mlir_expf
        --localize-symbol=_Assert --localize-symbol=aligned_alloc
        --localize-symbol=roundevenf "${input}" "${object}"
      DEPENDS "${input}" "${manifest}" VERBATIM)
    list(APPEND strict_prefill_objects "${object}")
  endforeach()
  set(object "${CMAKE_CURRENT_BINARY_DIR}/strict_prefill_adapter.o")
  add_custom_command(OUTPUT "${object}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags} -std=c++17
      -MMD -MF "${object}.d" -c "${CMAKE_CURRENT_SOURCE_DIR}/triton_strict_prefill.cpp" -o "${object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/triton_strict_prefill.cpp"
    DEPFILE "${object}.d" VERBATIM)
  list(APPEND strict_prefill_objects "${object}")
endif()
if(TRITON_DECODE_OBJECT)
  if(NOT EXISTS "${TRITON_DECODE_OBJECT}")
    message(FATAL_ERROR "TRITON_DECODE_OBJECT does not exist")
  endif()
  get_filename_component(decode_directory "${TRITON_DECODE_OBJECT}" DIRECTORY)
  set(decode_manifest "${decode_directory}/manifest.json")
  file(READ "${decode_manifest}" decode_receipt)
  string(JSON decode_profile GET "${decode_receipt}" profile)
  string(JSON decode_hash GET "${decode_receipt}" object_sha256)
  string(JSON decode_lwp GET "${decode_receipt}" lwp enabled)
  string(JSON decode_scratch GET "${decode_receipt}" scratch_fp32_elements)
  string(JSON decode_grid_size LENGTH "${decode_receipt}" grid)
  string(JSON decode_grid_x GET "${decode_receipt}" grid 0)
  string(JSON decode_grid_y GET "${decode_receipt}" grid 1)
  string(JSON decode_heads GET "${decode_receipt}" constants HEADS_PER_PROGRAM)
  string(JSON decode_approx_exp GET "${decode_receipt}" constants APPROX_EXP)
  file(SHA256 "${TRITON_DECODE_OBJECT}" actual_decode_hash)
  if((NOT decode_profile STREQUAL "strict-decode" AND NOT decode_profile STREQUAL "qfloat-decode") OR
      NOT decode_hash STREQUAL actual_decode_hash OR decode_lwp OR
      NOT decode_grid_size EQUAL 2 OR NOT decode_grid_x EQUAL 1 OR
      NOT decode_grid_y EQUAL 4 OR NOT decode_heads EQUAL 4 OR decode_approx_exp OR
      NOT decode_scratch MATCHES "^[1-9][0-9]*$" OR
      decode_scratch LESS 16384 OR decode_scratch GREATER 1064960)
    message(FATAL_ERROR "Decode object/manifest mismatch, unsupported grid/scratch, or multiworker LWP object")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${TRITON_DECODE_OBJECT}" "${decode_manifest}")
  set(triton_object "${CMAKE_CURRENT_BINARY_DIR}/triton_decode_kernel.o")
  add_custom_command(OUTPUT "${triton_object}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-llvm-objcopy"
      --localize-symbol=memrefCopy --localize-symbol=__hexagon_mlir_expf
      --localize-symbol=_Assert --localize-symbol=aligned_alloc
      --localize-symbol=roundevenf "${TRITON_DECODE_OBJECT}" "${triton_object}"
    DEPENDS "${TRITON_DECODE_OBJECT}" VERBATIM)
  set(prefill_object "${CMAKE_CURRENT_BINARY_DIR}/strict_prefill.o")
  add_custom_command(OUTPUT "${prefill_object}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags}
      -std=c++17 -fno-vectorize -fno-slp-vectorize -ffp-contract=off
      -D_mlir_ciface_qwen_ordered_fp32_attention_v1=qwen_strict_prefill
      -MMD -MF "${prefill_object}.d" -c
      "${CMAKE_CURRENT_SOURCE_DIR}/ordered_fp32_attention.cpp" -o "${prefill_object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ordered_fp32_attention.cpp"
    DEPFILE "${prefill_object}.d" VERBATIM)
  set(adapter_object "${CMAKE_CURRENT_BINARY_DIR}/triton_decode_adapter.o")
  add_custom_command(OUTPUT "${adapter_object}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags} -std=c++17
      -DQWEN_DECODE_SCRATCH_ELEMENTS=${decode_scratch}
      $<$<BOOL:${strict_prefill_enabled}>:-DQWEN_TRITON_PREFILL>
      -MMD -MF "${adapter_object}.d" -c
      "${CMAKE_CURRENT_SOURCE_DIR}/triton_decode.cpp" -o "${adapter_object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/triton_decode.cpp"
    DEPFILE "${adapter_object}.d" VERBATIM)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_strict_attention.a"
    COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs
      "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_strict_attention.a"
      "${adapter_object}" "${triton_object}" "${prefill_object}" ${strict_prefill_objects}
      "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o"
    DEPENDS "${adapter_object}" "${triton_object}" "${prefill_object}" ${strict_prefill_objects}
      "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o" VERBATIM)
  add_custom_target(qwen-triton-strict-attention-static
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_strict_attention.a")
endif()

# Explicit FP16 HMX attention. q1 defaults to the supplied HVX decode object.
set(TRITON_PREFILL_OBJECT "" CACHE FILEPATH "Object from experimental-prefill-fp16")
set(TRITON_PREFILL_POOL_BYTES "0" CACHE STRING "Temporary bytes per prefill worker, 0..1048576")
option(SHARE_HMX_RUNTIME "Share the HMX owner and VTCM pool with INT8 projections" OFF)
set(TRITON_HMX_DECODE_OBJECT "" CACHE FILEPATH "Object from experimental-decode-fp16; shares prefill runtime")
set(TRITON_HMX_DECODE_MIN_KV "256" CACHE STRING "Minimum effective KV length for optional q1 HMX")
set(hmx_decode_objects)
set(hmx_decode_flags)
if(TRITON_HMX_DECODE_OBJECT)
  if(NOT TRITON_PREFILL_OBJECT OR NOT TRITON_DECODE_OBJECT OR NOT SHARE_HMX_RUNTIME OR
      NOT TRITON_HMX_DECODE_MIN_KV MATCHES "^[1-9][0-9]*$" OR
      TRITON_HMX_DECODE_MIN_KV GREATER 1024)
    message(FATAL_ERROR "HMX decode requires prefill, HVX decode, shared HMX runtime and a threshold in 1..1024")
  endif()
  get_filename_component(directory "${TRITON_HMX_DECODE_OBJECT}" DIRECTORY)
  set(manifest "${directory}/manifest.json")
  file(READ "${manifest}" receipt)
  string(JSON profile GET "${receipt}" profile)
  string(JSON expected_hash GET "${receipt}" object_sha256)
  string(JSON lwp GET "${receipt}" lwp enabled)
  string(JSON entry GET "${receipt}" entry_point)
  string(JSON kv_groups GET "${receipt}" kv_heads_per_program)
  string(JSON slot_bytes GET "${receipt}" prepared_hmx_slot_bytes)
  get_filename_component(prefill_directory "${TRITON_PREFILL_OBJECT}" DIRECTORY)
  file(READ "${prefill_directory}/manifest.json" prefill_receipt)
  string(JSON shared_slot_bytes GET "${prefill_receipt}" prepared_hmx_slot_bytes)
  string(JSON shared GET "${receipt}" shared_runtime)
  string(JSON query_min GET "${receipt}" contract query 0)
  string(JSON query_max GET "${receipt}" contract query 1)
  string(JSON kv_max GET "${receipt}" contract kv_length_max)
  string(JSON block_n GET "${receipt}" constants BLOCK_N)
  string(JSON access GET "${receipt}" raw_buffer_access)
  string(JSON allocator GET "${receipt}" external_allocator)
  string(JSON prepared GET "${receipt}" prepare_hmx_matmul)
  string(JSON reuse GET "${receipt}" reuse_hmx_workspace)
  file(SHA256 "${TRITON_HMX_DECODE_OBJECT}" actual_hash)
  if(NOT profile STREQUAL "experimental-decode-fp16" OR
      NOT expected_hash STREQUAL actual_hash OR lwp OR
      NOT entry STREQUAL "attention_forward_q1_hmx_g2" OR
      NOT shared STREQUAL "experimental-prefill-fp16" OR NOT kv_groups EQUAL 2 OR
      NOT slot_bytes EQUAL 196608 OR NOT shared_slot_bytes EQUAL 196608 OR
      NOT query_min EQUAL 1 OR NOT query_max EQUAL 1 OR NOT kv_max EQUAL 1024 OR
      NOT block_n EQUAL 128 OR NOT access STREQUAL "dsp-load-store" OR
      NOT allocator OR NOT prepared OR NOT reuse)
    message(FATAL_ERROR "HMX decode object/manifest, shape or shared-runtime contract mismatch")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${TRITON_HMX_DECODE_OBJECT}" "${manifest}")
  list(APPEND hmx_decode_objects "${TRITON_HMX_DECODE_OBJECT}")
  list(APPEND hmx_decode_flags -DQWEN_TRITON_HMX_DECODE
    -DQWEN_HMX_DECODE_MIN_KV=${TRITON_HMX_DECODE_MIN_KV})
endif()
if(TRITON_PREFILL_OBJECT)
  if(NOT TRITON_DECODE_OBJECT OR NOT EXISTS "${TRITON_PREFILL_OBJECT}")
    message(FATAL_ERROR "Experimental prefill requires existing prefill and strict decode objects")
  endif()
  if(NOT TRITON_PREFILL_POOL_BYTES MATCHES "^(0|[1-9][0-9]*)$" OR
      TRITON_PREFILL_POOL_BYTES GREATER 1048576)
    message(FATAL_ERROR "TRITON_PREFILL_POOL_BYTES must be an integer in 0..1048576")
  endif()
  get_filename_component(prefill_directory "${TRITON_PREFILL_OBJECT}" DIRECTORY)
  set(prefill_manifest "${prefill_directory}/manifest.json")
  file(READ "${prefill_manifest}" prefill_receipt)
  string(JSON prefill_profile GET "${prefill_receipt}" profile)
  string(JSON prefill_hash GET "${prefill_receipt}" object_sha256)
  string(JSON prefill_hooks GET "${prefill_receipt}" external_allocator)
  string(JSON prefill_lwp GET "${prefill_receipt}" lwp enabled)
  string(JSON prefill_access ERROR_VARIABLE access_error GET "${prefill_receipt}" raw_buffer_access)
  if(access_error OR NOT prefill_access STREQUAL "dsp-load-store")
    message(FATAL_ERROR "Prefill requires the DSP-coherent load/store contract; rebuild the kernel with kernels/compile.py")
  endif()
  file(SHA256 "${TRITON_PREFILL_OBJECT}" actual_prefill_hash)
  if(NOT prefill_profile STREQUAL "experimental-prefill-fp16" OR
      NOT prefill_hash STREQUAL actual_prefill_hash OR prefill_lwp OR
      (TRITON_PREFILL_POOL_BYTES GREATER 0 AND NOT prefill_hooks))
    message(FATAL_ERROR "Prefill object/manifest mismatch, unsupported LWP, or missing allocator hooks")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${TRITON_PREFILL_OBJECT}" "${prefill_manifest}")
  set(prefill_kernel "${CMAKE_CURRENT_BINARY_DIR}/triton_prefill_kernel.o")
  add_custom_command(OUTPUT "${prefill_kernel}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-llvm-objcopy"
      --localize-symbol=memrefCopy --localize-symbol=__hexagon_mlir_expf
      --localize-symbol=_Assert --localize-symbol=aligned_alloc
      --localize-symbol=roundevenf "${TRITON_PREFILL_OBJECT}" "${prefill_kernel}"
    DEPENDS "${TRITON_PREFILL_OBJECT}" VERBATIM)
  set(experimental_objects)
  set(prefill_vtcm_bytes 1048576)
  if(SHARE_HMX_RUNTIME)
    set(prefill_vtcm_bytes 2097152)
  endif()
  foreach(source IN ITEMS "${CMAKE_CURRENT_SOURCE_DIR}/triton_decode.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/triton_prefill.cpp"
      "${common_native}/temporary_buffer_pool.cpp")
    get_filename_component(name "${source}" NAME_WE)
    set(obj "${CMAKE_CURRENT_BINARY_DIR}/experimental_${name}.o")
    add_custom_command(OUTPUT "${obj}"
      COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags} -std=c++17
        -DQWEN_TRITON_PREFILL -DQWEN_PREFILL_POOL_BYTES=${TRITON_PREFILL_POOL_BYTES}
        -DQWEN_PREFILL_VTCM_BYTES=${prefill_vtcm_bytes}
        -DQWEN_DECODE_SCRATCH_ELEMENTS=${decode_scratch} ${hmx_decode_flags}
        -MMD -MF "${obj}.d" -c "${source}" -o "${obj}"
      DEPENDS "${source}" DEPFILE "${obj}.d" VERBATIM)
    list(APPEND experimental_objects "${obj}")
  endforeach()
  set(experimental_archive "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_experimental_attention.a")
  add_custom_command(OUTPUT "${experimental_archive}"
    COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs "${experimental_archive}"
      ${experimental_objects} "${prefill_kernel}" "${triton_object}" ${hmx_decode_objects}
      "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o"
    DEPENDS ${experimental_objects} "${prefill_kernel}" "${triton_object}" ${hmx_decode_objects}
      "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o" VERBATIM)
  add_custom_target(qwen-triton-experimental-attention-static
    DEPENDS "${experimental_archive}")
elseif(NOT TRITON_PREFILL_POOL_BYTES STREQUAL "0")
  message(FATAL_ERROR "TRITON_PREFILL_POOL_BYTES requires TRITON_PREFILL_OBJECT")
endif()

# Strict activation kernels share the same pinned worker-pool dependency.
set(silu_source "${CMAKE_CURRENT_SOURCE_DIR}/../activation/ordered_silu.cpp")
set(silu_object "${CMAKE_CURRENT_BINARY_DIR}/ordered_silu.o")
add_custom_command(OUTPUT "${silu_object}"
  COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags}
    -std=c++17 -fno-vectorize -fno-slp-vectorize -ffp-contract=off
    -MMD -MF "${silu_object}.d" -c "${silu_source}" -o "${silu_object}"
  DEPENDS "${silu_source}" DEPFILE "${silu_object}.d" VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libqwen_ordered_silu.a"
  COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs
    "${CMAKE_CURRENT_BINARY_DIR}/libqwen_ordered_silu.a"
    "${silu_object}" "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o"
  DEPENDS "${silu_object}" "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o" VERBATIM)
add_custom_target(qwen-ordered-silu-static
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libqwen_ordered_silu.a")

# Optional Triton SiLU shares the existing validation, cache and worker adapter.
set(TRITON_SILU_OBJECT "" CACHE FILEPATH "Object from kernels/compile.py strict-silu or experimental-silu-fp32")
if(TRITON_SILU_OBJECT)
  get_filename_component(silu_directory "${TRITON_SILU_OBJECT}" DIRECTORY)
  set(silu_manifest "${silu_directory}/manifest.json")
  file(READ "${silu_manifest}" silu_receipt)
  string(JSON silu_profile GET "${silu_receipt}" profile)
  string(JSON silu_hash GET "${silu_receipt}" object_sha256)
  string(JSON silu_lwp GET "${silu_receipt}" lwp enabled)
  string(JSON silu_grid_size LENGTH "${silu_receipt}" grid)
  string(JSON silu_workers GET "${silu_receipt}" grid 0)
  file(SHA256 "${TRITON_SILU_OBJECT}" actual_silu_hash)
  if((NOT silu_profile STREQUAL "strict-silu" AND NOT silu_profile STREQUAL "experimental-silu-fp32" AND NOT silu_profile STREQUAL "llama-silu-fp32") OR NOT silu_hash STREQUAL actual_silu_hash OR
      silu_lwp OR NOT silu_grid_size EQUAL 1 OR NOT silu_workers EQUAL 4)
    message(FATAL_ERROR "SiLU object/manifest mismatch, unsupported grid or instrumented kernel")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TRITON_SILU_OBJECT}" "${silu_manifest}")
  if(NOT EXISTS "${TRITON_SILU_OBJECT}")
    message(FATAL_ERROR "TRITON_SILU_OBJECT does not exist")
  endif()
  set(silu_kernel "${CMAKE_CURRENT_BINARY_DIR}/triton_silu_kernel.o")
  set(silu_adapter "${CMAKE_CURRENT_BINARY_DIR}/triton_silu_adapter.o")
  set(silu_precision_flags "")
  if(silu_profile STREQUAL "experimental-silu-fp32" OR silu_profile STREQUAL "llama-silu-fp32")
    set(silu_precision_flags -DQWEN_APPROX_SILU)
  endif()
  add_custom_command(OUTPUT "${silu_kernel}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-llvm-objcopy"
      --localize-symbol=memrefCopy --localize-symbol=__hexagon_mlir_expf
      --localize-symbol=_Assert --localize-symbol=aligned_alloc
      --localize-symbol=roundevenf "${TRITON_SILU_OBJECT}" "${silu_kernel}"
    DEPENDS "${TRITON_SILU_OBJECT}" VERBATIM)
  add_custom_command(OUTPUT "${silu_adapter}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags}
      -std=c++17 -ffp-contract=off -DQWEN_TRITON_SILU ${silu_precision_flags}
      -MMD -MF "${silu_adapter}.d" -c "${silu_source}" -o "${silu_adapter}"
    DEPENDS "${silu_source}" DEPFILE "${silu_adapter}.d" VERBATIM)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_silu.a"
    COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs
      "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_silu.a"
      "${silu_adapter}" "${silu_kernel}" "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o"
    DEPENDS "${silu_adapter}" "${silu_kernel}" "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o" VERBATIM)
  add_custom_target(qwen-triton-silu-static
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_silu.a")
endif()

# Experimental quantizer-specific product shares the ordered SiLU owner/pool.
# It is a distinct ABI, never an ordinary SiLU replacement.
set(TRITON_SILU_UP_OBJECT "" CACHE FILEPATH "Object from kernels/compile.py experimental-quantized-silu-up")
if(TRITON_SILU_UP_OBJECT)
  if(NOT TRITON_SILU_OBJECT OR NOT EXISTS "${TRITON_SILU_UP_OBJECT}")
    message(FATAL_ERROR "TRITON_SILU_UP_OBJECT requires an existing object and TRITON_SILU_OBJECT")
  endif()
  foreach(pair IN ITEMS "${TRITON_SILU_UP_OBJECT}|experimental-quantized-silu-up"
                        "${TRITON_SILU_OBJECT}|strict-silu")
    string(REPLACE "|" ";" fields "${pair}")
    list(GET fields 0 object)
    list(GET fields 1 expected_profile)
    get_filename_component(directory "${object}" DIRECTORY)
    set(receipt_path "${directory}/manifest.json")
    file(READ "${receipt_path}" receipt)
    string(JSON profile GET "${receipt}" profile)
    string(JSON expected_hash GET "${receipt}" object_sha256)
    string(JSON instrumented GET "${receipt}" lwp enabled)
    file(SHA256 "${object}" actual_hash)
    if(NOT profile STREQUAL expected_profile OR NOT actual_hash STREQUAL expected_hash OR instrumented)
      message(FATAL_ERROR "SiLU object/manifest mismatch or unsupported multiworker LWP object")
    endif()
    if(expected_profile STREQUAL "experimental-quantized-silu-up")
      string(JSON scale_contract GET "${receipt}" contract scale)
      if(NOT scale_contract STREQUAL "max(max(abs(Y))/127, float32(1e-5))")
        message(FATAL_ERROR "Quantized SiLU requires the model scale floor float32(1e-5); rebuild the kernel")
      endif()
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${object}" "${receipt_path}")
  endforeach()
  set(product_kernel "${CMAKE_CURRENT_BINARY_DIR}/triton_silu_up_kernel.o")
  set(product_adapter "${CMAKE_CURRENT_BINARY_DIR}/triton_silu_up_adapter.o")
  add_custom_command(OUTPUT "${product_kernel}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-llvm-objcopy"
      --localize-symbol=memrefCopy --localize-symbol=__hexagon_mlir_expf
      --localize-symbol=_Assert --localize-symbol=aligned_alloc
      --localize-symbol=roundevenf "${TRITON_SILU_UP_OBJECT}" "${product_kernel}"
    DEPENDS "${TRITON_SILU_UP_OBJECT}" VERBATIM)
  add_custom_command(OUTPUT "${product_adapter}"
    COMMAND "${HEXAGON_TOOLS}/bin/hexagon-clang++" ${flags}
      -std=c++17 -ffp-contract=off -DQWEN_TRITON_SILU -DQWEN_TRITON_SILU_UP
      -MMD -MF "${product_adapter}.d" -c "${silu_source}" -o "${product_adapter}"
    DEPENDS "${silu_source}" DEPFILE "${product_adapter}.d" VERBATIM)
  set(product_archive "${CMAKE_CURRENT_BINARY_DIR}/libqwen_triton_quantized_silu.a")
  add_custom_command(OUTPUT "${product_archive}"
    COMMAND "${HEXAGON_TOOLS}/bin/llvm-ar" rcs "${product_archive}"
      "${product_adapter}" "${product_kernel}" "${silu_kernel}"
      "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o"
    DEPENDS "${product_adapter}" "${product_kernel}" "${silu_kernel}"
      "${CMAKE_CURRENT_BINARY_DIR}/worker-pool.o" VERBATIM)
  add_custom_target(qwen-triton-quantized-silu-static DEPENDS "${product_archive}")
endif()
