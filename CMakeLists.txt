cmake_minimum_required(VERSION 3.25)

project(test_lru_hash_project LANGUAGES CXX)

# ------------------------------------------------------------
# Compiler Verification (Unix Only)
# ------------------------------------------------------------
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    message(FATAL_ERROR "This CMake project supports only GCC or Clang (Linux/macOS). Use MSVC project files for Windows.")
endif()

# ------------------------------------------------------------
# C++ standard
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# Default to Release Build
# ------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ------------------------------------------------------------
# Global Options & Dependencies
# ------------------------------------------------------------
option(USE_EXPONENTIAL_BACKOFF "Use ExponentialBackoffPolicy locking policy for better performance on hybrid CPU arhitecture" OFF)

if(USE_EXPONENTIAL_BACKOFF)
    message(STATUS "Hybrid Optimization Enabled: Compiling with LRU_USE_EXPONENTIAL_BACKOFF")
else()
    message(STATUS "Default: Compiling with AdaptiveSpinPolicy locking policy")
endif()

find_package(Threads REQUIRED)

# ============================================================
# TARGET: test_lru_hash (Linux / macOS Only)
# ============================================================
set(TEST_UM_SOURCES
    test_um/test_lru_hash.cpp
)

add_executable(test_lru_hash ${TEST_UM_SOURCES})

if(USE_EXPONENTIAL_BACKOFF)
    target_compile_definitions(test_lru_hash PRIVATE LRU_USE_EXPONENTIAL_BACKOFF)
endif()

target_include_directories(test_lru_hash PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/um
    ${CMAKE_CURRENT_SOURCE_DIR}/test_um
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common 
)

# Compiler optimization flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_lru_hash PRIVATE
        -O3 -march=native -flto=auto -funroll-loops
        -fomit-frame-pointer -fno-rtti -fexceptions -pthread
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_lru_hash PRIVATE
        -O3 -march=native -flto -funroll-loops
        -fomit-frame-pointer -fno-rtti -fexceptions -pthread 
    )
endif()

target_link_options(test_lru_hash PRIVATE -flto)
target_link_libraries(test_lru_hash PRIVATE Threads::Threads)

# NUMA Support (Linux only)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(NUMA_LIBRARY numa REQUIRED)
    target_link_libraries(test_lru_hash PRIVATE ${NUMA_LIBRARY})
    message(STATUS "NUMA library found and linked: ${NUMA_LIBRARY}")
endif()

# jemalloc Support (Linux only - New)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(JEMALLOC_LIBRARY jemalloc)
    if(JEMALLOC_LIBRARY)
        target_link_libraries(test_lru_hash PRIVATE ${JEMALLOC_LIBRARY})
        message(STATUS "jemalloc found and linked: ${JEMALLOC_LIBRARY}")
    else()
        message(WARNING "jemalloc not found. Falling back to system heap.")
    endif()
endif()

set_target_properties(test_lru_hash PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
