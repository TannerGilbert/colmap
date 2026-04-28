#!/usr/bin/env bash
# Incremental colmap C++ build with ccache.
# First run configures CMake; subsequent runs only rebuild changed TUs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
PREFIX="$ROOT/local"

if [ ! -f "$BUILD/build.ninja" ]; then
    cmake -S "$ROOT" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
fi

cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"
