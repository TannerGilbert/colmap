#!/usr/bin/env bash
# Editable, incremental pycolmap rebuild.
# - Persistent CMake build dir at colmap/pybuild/
# - ccache on the binding compile
# - -ve: editable install (Python wrapper edits don't need reinstall)
# - Post-install: symlink _core.so back into source tree so editable install can find it
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CMAKE_C_COMPILER_LAUNCHER=ccache \
CMAKE_CXX_COMPILER_LAUNCHER=ccache \
CMAKE_PREFIX_PATH="$ROOT/local" \
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc" \
    pip install \
        -Cbuild-dir="$ROOT/pybuild" \
        --no-build-isolation \
        -ve \
        "$ROOT"

# Symlink compiled _core.so into source tree (editable install needs it there).
site_pkg=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
core_so=$(ls -t "$site_pkg/pycolmap"/_core*.so 2>/dev/null | head -1)
if [ -n "$core_so" ]; then
    ln -sf "$core_so" "$ROOT/python/pycolmap/"
fi
