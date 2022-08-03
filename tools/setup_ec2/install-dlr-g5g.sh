#!/bin/bash -xe

# See : https://neo-ai-dlr.readthedocs.io/en/latest/install.html#building-for-nvidia-gpu-cloud-or-desktop

git clone --recursive https://github.com/neo-ai/neo-ai-dlr
cd neo-ai-dlr
#git checkout tags/v1.10.0 # to install to python3.7

mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON
make -j4
cd ../python
python3.8 setup.py install --user

cd ../..
rm -rf neo-ai-dlr
