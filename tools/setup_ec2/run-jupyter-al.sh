#!/bin/bash

export LD_LIBRARY_PATH=$HOME/glibc-2.27-subset:$LD_LIBRARY_PATH

jupyter-lab --no-browser --allow-root --port 8888 --notebook-dir ~
