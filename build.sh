#!/bin/bash

set -e

# Defaults
BUILD_TYPE="Release"
COMPILER="g++"
USE_HYBRID="OFF"
CLEAN_BUILD=false

# ==========================================
# 0. Detect and remove in-source cache
# ==========================================
SCRIPT_DIR=$(dirname "$(realpath "$0")")  # folder where build.sh is
if [ -f "$SCRIPT_DIR/CMakeCache.txt" ] || [ -d "$SCRIPT_DIR/CMakeFiles" ]; then
    echo "Warning: Found CMake cache in source directory!"
    echo "Removing in-source CMake cache to prevent build issues..."
    rm -f "$SCRIPT_DIR/CMakeCache.txt"
    rm -rf "$SCRIPT_DIR/CMakeFiles"
fi

# ==========================================
# 1. Parse Command Line Arguments
# ==========================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --hybrid)
            USE_HYBRID="ON"
            shift
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --compiler)
            COMPILER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo "Options:"
            echo "  -c, --clean         Wipe old build directory before building"
            echo "  --hybrid            Enable hybrid spin policy"
            echo "  -t, --type TYPE     Set build type (e.g., Release, Debug) [Default: Release]"
            echo "  --compiler COMP     Set compiler (e.g., g++, clang++) [Default: g++]"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Run './build.sh --help' for usage."
            exit 1
            ;;
    esac
done

# ==========================================
# 2. Execute Pre-Build Actions
# ==========================================
if [ "$CLEAN_BUILD" = true ]; then
    echo "Clean build requested: Wiping old build directory..."
    rm -rf build
fi

echo "========================================"
echo " Building test_lru_hash                 "
echo " Build Type:  $BUILD_TYPE               "
echo " Compiler:    $COMPILER                 "
echo " Hybrid Opt:  $USE_HYBRID               "
echo "========================================"

mkdir -p build
cd build

# ==========================================
# 3. Configure and Build
# ==========================================
echo "-> Configuring CMake..."
CXX=$COMPILER cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DUSE_EXPONENTIAL_BACKOFF=$USE_HYBRID

echo "-> Compiling..."
cmake --build . -j $(nproc)

echo "========================================"
echo " Build successful!                      "
echo " You can run your test via:             "
echo " ./build/bin/test_lru_hash              "
echo "========================================"
