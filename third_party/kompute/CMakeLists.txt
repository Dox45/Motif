cmake_minimum_required(VERSION 3.20)
project(kompute_motif CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DKOMPUTE_OPT_ANDROID_BUILD=1)
add_definitions(-DKOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS=1)
add_definitions(-DKOMPUTE_OPT_LOG_LEVEL_DISABLED=1)
add_definitions(-DVK_NO_PROTOTYPES=1)
add_definitions(-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR=1)

include(FetchContent)

# 1. Vulkan-Headers
if(NOT DEFINED VULKAN_INCLUDE_DIR OR NOT EXISTS "${VULKAN_INCLUDE_DIR}/vulkan/vulkan.hpp")
    if(EXISTS "${ANDROID_NDK}/sysroot/usr/include/vulkan/vulkan.hpp")
        set(VULKAN_INCLUDE_DIR "${ANDROID_NDK}/sysroot/usr/include")
    else()
        FetchContent_Declare(vulkan_header
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG v1.3.231
            GIT_SHALLOW 1
        )
        FetchContent_MakeAvailable(vulkan_header)
        set(VULKAN_INCLUDE_DIR "${vulkan_header_SOURCE_DIR}/include")
    endif()
endif()

# 2. fmt formatting library
if(NOT TARGET fmt AND NOT TARGET fmt::fmt)
    FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 8.1.1
        GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(fmt)
endif()

# 3. Generate precompiled Shader headers
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/glsl/ShaderOpMult.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/ShaderOpMult.hpp
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/glsl/ShaderLogisticRegression.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/shaders/ShaderLogisticRegression.hpp
    COPYONLY
)

file(GLOB KOMPUTE_SOURCES
    src/*.cpp
    src/logger/*.cpp
)

add_library(kompute STATIC ${KOMPUTE_SOURCES})
add_library(kompute::kompute ALIAS kompute)

target_include_directories(kompute PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/shaders>
    $<BUILD_INTERFACE:${VULKAN_INCLUDE_DIR}>
)

target_link_libraries(kompute PUBLIC fmt::fmt)

if(ANDROID)
    find_library(VULKAN_LIB vulkan)
    find_library(ANDROID_LIB android)
    find_library(LOG_LIB log)
    target_link_libraries(kompute PUBLIC ${VULKAN_LIB} ${ANDROID_LIB} ${LOG_LIB})
else()
    find_package(Vulkan REQUIRED)
    target_link_libraries(kompute PUBLIC Vulkan::Vulkan)
endif()
