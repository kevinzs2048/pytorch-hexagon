# Source-only SLEEF dependency. Never configure or build upstream SLEEF targets.
function(qwen_prepare_sleef output_dir)
  set(QWEN_SLEEF_SOURCE_DIR "$ENV{QWEN_SLEEF_SOURCE_DIR}" CACHE PATH
    "Unmodified SLEEF 3.8 source tree for offline builds")
  if(QWEN_SLEEF_SOURCE_DIR)
    get_filename_component(source "${QWEN_SLEEF_SOURCE_DIR}" ABSOLUTE)
  else()
    include(FetchContent)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Populate(qwen_sleef
      URL https://codeload.github.com/shibatch/sleef/tar.gz/refs/tags/3.8
      URL_HASH SHA256=a12ccd50f57083c530e1c76f10d52865defbd19fc9e2c85b483493065709874a
      TLS_VERIFY TRUE
      TIMEOUT 120)
    set(source "${qwen_sleef_SOURCE_DIR}")
  endif()

  set(files
    "src/libm/sleefsimdsp.c|726d962aaa9cfb0b1696122f4d26765495491c0cbc26c084c899704de46b2f5c"
    "src/libm/rempitab.c|105201cc9c80126a3f18ea96e52540c0d13fda7c91a664599ba4078639f961ad"
    "src/common/quaddef.h|7dd06a0161c384d6823f4554312765cd5af61eb913fd997c5f835c10bf5cf309"
    "src/common/misc.h|516ddeb5cc9501039b104967f1830b92e94cc6053c401fbb6773873dfb1b6178"
    "src/common/df.h|cd3c7de0e47f3751a15dd62878effabdec9e47a892a03fefd75981c26fdaf097"
    "src/common/estrin.h|b7b49955fb5fb538096b454792aa0d0566c247ba0ea293f8f05f0c982a4096ed"
    "src/arch/helperpurec.h|3a32de8591ecf91ee1ed2c0e0dc5e224c45245d5c8dbe11eb3eabf865b9cefe7")
  set(prepared "${CMAKE_CURRENT_BINARY_DIR}/sleef-rope")
  file(MAKE_DIRECTORY "${prepared}" "${prepared}/adaptation")
  foreach(entry IN LISTS files)
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 path)
    list(GET fields 1 expected)
    if(NOT EXISTS "${source}/${path}")
      message(FATAL_ERROR "Missing SLEEF 3.8 dependency: ${source}/${path}")
    endif()
    file(SHA256 "${source}/${path}" actual)
    if(NOT actual STREQUAL expected)
      message(FATAL_ERROR "SLEEF 3.8 SHA256 mismatch: ${path}")
    endif()
    get_filename_component(name "${path}" NAME)
    if(name STREQUAL "helperpurec.h")
      configure_file("${source}/${path}" "${prepared}/adaptation/${name}" COPYONLY)
    else()
      configure_file("${source}/${path}" "${prepared}/${name}" COPYONLY)
    endif()
  endforeach()

  find_program(QWEN_PATCH patch REQUIRED)
  set(adaptation "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/helper-adaptation.patch")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${adaptation}")
  execute_process(COMMAND "${QWEN_PATCH}" --batch --silent
    "${prepared}/adaptation/helperpurec.h" "${adaptation}"
    RESULT_VARIABLE result ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Could not apply SLEEF rounding/FMA adaptation: ${error}")
  endif()
  file(SHA256 "${prepared}/adaptation/helperpurec.h" adapted_hash)
  if(NOT adapted_hash STREQUAL "1dec369567711e9cf902d64d2457a3d9d1b757c2a83db5f14745d8161dabe7ac")
    message(FATAL_ERROR "Adapted SLEEF helper SHA256 mismatch")
  endif()
  configure_file("${prepared}/adaptation/helperpurec.h" "${prepared}/helperpurec.h" COPYONLY)
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/LICENSE.txt" "${prepared}/LICENSE.txt" COPYONLY)
  set(${output_dir} "${prepared}" PARENT_SCOPE)
endfunction()
