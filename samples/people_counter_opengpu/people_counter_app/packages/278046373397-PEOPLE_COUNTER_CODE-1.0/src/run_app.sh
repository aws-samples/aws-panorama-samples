#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export OPENBLAS_CORETYPE=ARMV8
python3 /panorama/app.py
