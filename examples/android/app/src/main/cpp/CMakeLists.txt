cmake_minimum_required(VERSION 3.20)
project(motif_android_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Include embedded Kompute Android library from Motif third_party
add_subdirectory(../../../../../../third_party/kompute ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../third_party/kompute/include
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
)

# 2. Build Motif JNI shared library for Android
add_library(motif-jni SHARED
    MotifJniNative.cpp
    generated/motif_vecadd.cpp
    generated/motif_matmul.cpp
    generated/motif_logistic.cpp
)

target_link_libraries(motif-jni PRIVATE
    kompute::kompute
    vulkan
    android
    log
)
