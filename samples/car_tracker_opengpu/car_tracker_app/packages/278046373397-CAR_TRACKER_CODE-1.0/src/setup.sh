#!/bin/bash
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so

python3 /panorama/app.py
