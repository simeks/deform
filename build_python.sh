#!/bin/sh

export CMAKE_ARGS=$@

# Install STK first
(cd third_party/stk && pip install -e .)
# Install deform
python -m build