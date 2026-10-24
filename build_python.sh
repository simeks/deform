#!/bin/sh

export CMAKE_ARGS=$@

# Install STK first
(cd third_party/stk && pip install .)
# Install deform
pip install .