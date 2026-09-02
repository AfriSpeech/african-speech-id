# Android build for the JNI shim. Point at this from app/build.gradle:
#
#   externalNativeBuild { cmake { path "../bindings/android/CMakeLists.txt" } }
#
# ONNXRUNTIME_ROOT should be an extracted onnxruntime AAR: headers under headers/, and one
# libonnxruntime.so per ABI under jni/<abi>/.
cmake_minimum_required(VERSION 3.18)
project(african_speech_id_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASID_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Extracted onnxruntime Android AAR")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/headers ${ONNXRUNTIME_ROOT}/include)
find_library(ORT_LIBRARY onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI} ${ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI})
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "set -DONNXRUNTIME_ROOT to an extracted onnxruntime Android AAR")
endif()

add_library(african_speech_id_jni SHARED
  ${ASID_ROOT}/src/african_speech_id.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jni/asid_jni.cpp)

target_include_directories(african_speech_id_jni PRIVATE ${ASID_ROOT}/include ${ORT_INCLUDE_DIR})
target_link_libraries(african_speech_id_jni PRIVATE ${ORT_LIBRARY} log)
target_compile_options(african_speech_id_jni PRIVATE -Wall -Wextra -Os -fvisibility=hidden)
