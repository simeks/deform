#!/bin/sh
# Usage: sh build_python.sh [--use-cuda] [cmake flags...]
set -e

if [ "$1" = "--use-cuda" ]; then
    shift
    set -- "$@" -DDF_USE_CUDA=ON -DSTK_USE_CUDA=ON
fi

export CMAKE_ARGS="$*"

# Install STK first
(cd third_party/stk && pip install .)
# Install deform
pip install .
