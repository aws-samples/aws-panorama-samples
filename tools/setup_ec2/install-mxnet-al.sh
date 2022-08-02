#!/bin/bash -xe

# This doesn't work on ARM
# On ARM, need to install mxnet from source code
#sudo python3.7 -m pip install mxnet

# install dependencies for mxnet
#sudo apt install libopenblas-dev -y
#sudo apt install libopencv-dev -y

# download source code package
wget https://archive.apache.org/dist/incubator/mxnet/1.8.0/apache-mxnet-src-1.8.0-incubating.tar.gz
tar xvzf apache-mxnet-src-1.8.0-incubating.tar.gz
cd apache-mxnet-src-1.8.0-incubating

# configure
echo "USE_OPENCV = 1" >> ./make/config.mk
echo "USE_BLAS = openblas" >> ./make/config.mk
echo "USE_SSE = 0" >> ./make/config.mk
echo "USE_CUDA = 0" >> ./make/config.mk
echo "MSHADOW_STAND_ALONE = 1" >> ./make/config.mk

# build & install
make -j4
cd python
sudo python3.7 setup.py install

sudo python3.7 -m pip install gluoncv

# clean up
cd ../..
sudo rm -rf apache-mxnet-src-1.8.0-incubating
rm apache-mxnet-src-1.8.0-incubating.tar.gz
