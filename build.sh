#!/bin/sh

# Install STK first
(cd third_party/stk && python setup.py install $@)
# Install deform
python setup.py install $@
