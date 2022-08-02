#!/bin/bash -xe

sudo yum install bison -y

wget http://ftp.gnu.org/gnu/libc/glibc-2.27.tar.gz
tar xvzf glibc-2.27.tar.gz
mkdir glibc-2.27-build
mkdir glibc-2.27-install
cd glibc-2.27-build
../glibc-2.27/configure --prefix=$HOME/glibc-2.27-full
make
make install

# Cherry pick libm (only glibc-2.27 version of libm is needed to use DLR)
mkdir -p $HOME/glibc-2.27-subset
cp $HOME/glibc-2.27-full/lib/libm.so.6 $HOME/glibc-2.27-subset/

cd ..
rm -rf glibc-2.27
rm -rf glibc-2.27-build
rm -rf glibc-2.27-install
rm glibc-2.27.tar.gz
