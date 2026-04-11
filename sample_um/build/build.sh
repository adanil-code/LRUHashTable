#!/bin/bash

set -e

# Defaults
BUILD_TYPE="Release"
COMPILER="g++"
CLEAN_BUILD=false

# ==========================================
# 1. Parse Command Line Arguments
# ==========================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
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

# Navigate to the directory where the script is located (sample_um/build/)
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# ==========================================
# 2. Execute Pre-Build Actions
# ==========================================
if [ "$CLEAN_BUILD" = true ]; then
    echo "Clean build requested: Wiping old CMake cache and bin..."
    rm -rf CMakeCache.txt CMakeFiles Makefile cmake_install.cmake bin/
fi

echo "========================================"
echo " Building sample_lru_hash               "
echo " Build Type:  $BUILD_TYPE               "
echo " Compiler:    $COMPILER                 "
echo "========================================"

# ==========================================
# 3. Configure and Build
# ==========================================
echo "-> Configuring CMake..."
# Build in-place since we are already inside the dedicated build folder
CXX=$COMPILER cmake -S . -B . \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE

echo "-> Compiling..."
cmake --build . -j $(nproc)

echo "========================================"
echo " Build successful!                      "
echo " You can run your sample via:           "
echo " ./bin/sample_lru_hash                  "
echo "========================================"
