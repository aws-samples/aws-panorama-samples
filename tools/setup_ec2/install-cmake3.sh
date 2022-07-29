#!/bin/bash -xe

# sudo apt install cmake -y # -> this installs old version

# See : https://www.matbra.com/2017/12/07/install-cmake-on-aws-linux.html

wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2.tar.gz
tar xvzf cmake-3.22.2.tar.gz
cd cmake-3.22.2
./bootstrap
make
sudo make install

rm -rf cmake-3.22.2
rm cmake-3.22.2.tar.gz
