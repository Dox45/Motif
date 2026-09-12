cmake_minimum_required(VERSION 3.20)
project(motif_android CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include embedded Android-specific Kompute library
add_subdirectory(third_party/kompute)

# Include generated native dispatch files
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/motif/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kompute/include
)

# Optional standalone test executable (built when NDK or Vulkan host build is run)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/test_runner.cpp")
    add_executable(motif_test_vecadd generated/test_runner.cpp)
    target_link_libraries(motif_test_vecadd PRIVATE kompute::kompute)
endif()
