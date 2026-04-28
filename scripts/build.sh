#!/usr/bin/env bash
# One-shot: rebuild colmap C++ (incremental + ccache) then pycolmap binding (editable + ccache).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/build_cpp.sh"
"$DIR/build_py.sh"
