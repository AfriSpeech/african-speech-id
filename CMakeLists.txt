cmake_minimum_required(VERSION 3.16)
project(african_speech_id VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ASID_BUILD_CLI   "Build the gsid command line tool" ON)
option(ASID_BUILD_TESTS "Build the self test"              ON)
option(ASID_SHARED      "Build a shared library"           OFF)

# onnxruntime -------------------------------------------------------------------
# Point ONNXRUNTIME_ROOT at an extracted release from
#   https://github.com/microsoft/onnxruntime/releases
# (onnxruntime-linux-x64-*, onnxruntime-android-*.aar, onnxruntime-osx-*, ...).
# On Android the usual route is the AAR's headers plus libonnxruntime.so per ABI.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an onnxruntime C/C++ distribution")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        ${ONNXRUNTIME_ROOT}/headers
  PATH_SUFFIXES onnxruntime core/session)

find_library(ORT_LIBRARY NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64 ${ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI})

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR
    "onnxruntime not found. Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>.\n"
    "  include: ${ORT_INCLUDE_DIR}\n  library: ${ORT_LIBRARY}")
endif()
message(STATUS "onnxruntime headers: ${ORT_INCLUDE_DIR}")
message(STATUS "onnxruntime library: ${ORT_LIBRARY}")

# library -----------------------------------------------------------------------
if(ASID_SHARED)
  add_library(african_speech_id SHARED src/african_speech_id.cpp)
  target_compile_definitions(african_speech_id PRIVATE ASID_BUILDING PUBLIC ASID_SHARED)
else()
  add_library(african_speech_id STATIC src/african_speech_id.cpp)
endif()

target_include_directories(african_speech_id
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(african_speech_id PRIVATE ${ORT_LIBRARY})

if(MSVC)
  target_compile_options(african_speech_id PRIVATE /W4)
else()
  target_compile_options(african_speech_id PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ASID_BUILD_CLI)
  add_executable(gsid src/cli.cpp)
  target_link_libraries(gsid PRIVATE african_speech_id ${ORT_LIBRARY})
  target_include_directories(gsid PRIVATE ${ORT_INCLUDE_DIR})
endif()

if(ASID_BUILD_TESTS)
  enable_testing()
  add_executable(asid_selftest tests/selftest.cpp)
  target_link_libraries(asid_selftest PRIVATE african_speech_id ${ORT_LIBRARY})
  target_include_directories(asid_selftest PRIVATE ${ORT_INCLUDE_DIR})
  # needs a model: ctest -V after exporting one, with ASID_MODEL_DIR set
  add_test(NAME selftest COMMAND asid_selftest)
endif()

include(GNUInstallDirs)
install(TARGETS african_speech_id
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/african_speech_id.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(ASID_BUILD_CLI)
  install(TARGETS gsid RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
