#!/usr/bin/env bash

source .venv/bin/activate

# Robotoc local install
export PYTHONPATH=$(pwd)/external/install/lib/python./site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/external/install/lib:$LD_LIBRARY_PATH

# Robotpkg Pinocchio
export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
