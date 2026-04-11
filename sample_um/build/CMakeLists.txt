cmake_minimum_required(VERSION 3.25)

project(sample_lru_hash_project LANGUAGES CXX)

# ------------------------------------------------------------
# Compiler Verification (Unix Only)
# ------------------------------------------------------------
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    message(FATAL_ERROR "This CMake project supports only GCC or Clang (Linux/macOS).")
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
find_package(Threads REQUIRED)

# ============================================================
# TARGET: sample_lru_hash
# ============================================================
# Because CMakeLists.txt is in sample_um/build/, our source is one level up
set(SAMPLE_SOURCES
    ../sample_lru_hash.cpp
)

add_executable(sample_lru_hash ${SAMPLE_SOURCES})

# Include directories relative to sample_um/build/
target_include_directories(sample_lru_hash PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..        # For sample_string_key.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../um  # For lru_hash_table.h, lru_string_key.h
)

# Compiler optimization flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(sample_lru_hash PRIVATE
        -O3 -march=native -flto=auto -funroll-loops
        -fomit-frame-pointer -fno-rtti -fexceptions -pthread
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(sample_lru_hash PRIVATE
        -O3 -march=native -flto -funroll-loops
        -fomit-frame-pointer -fno-rtti -fexceptions -pthread 
    )
endif()

target_link_options(sample_lru_hash PRIVATE -flto)
target_link_libraries(sample_lru_hash PRIVATE Threads::Threads)

# NUMA Support (Linux only - RESTORED)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(NUMA_LIBRARY numa REQUIRED)
    target_link_libraries(sample_lru_hash PRIVATE ${NUMA_LIBRARY})
    message(STATUS "NUMA library found and linked: ${NUMA_LIBRARY}")
endif()

# jemalloc Support (Linux only)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(JEMALLOC_LIBRARY jemalloc)
    if(JEMALLOC_LIBRARY)
        target_link_libraries(sample_lru_hash PRIVATE ${JEMALLOC_LIBRARY})
        message(STATUS "jemalloc found and linked: ${JEMALLOC_LIBRARY}")
    else()
        message(WARNING "jemalloc not found. Falling back to system heap.")
    endif()
endif()

# Output the binary to sample_um/build/bin/
set_target_properties(sample_lru_hash PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
