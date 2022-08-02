#!/bin/bash -xe

# install dev foundation

sudo yum install git -y
sudo yum install gcc-c++ -y         # for cmake
sudo yum install openssl-devel -y   # for cmake
sudo yum install python3-devel -y
sudo yum install opencv -y          # is this needed before installing opencv-python?

# install Python libraries and CLI tools

sudo pip3 install boto3
sudo pip3 install awscli
sudo pip3 install sagemaker
sudo pip3 install panoramacli
sudo pip3 install numpy
sudo pip3 install matplotlib
sudo pip3 install opencv-python
sudo pip3 install jupyterlab

./install-docker-al.sh

./install-cmake3.sh

./install-dlr.sh

./install-glibc-al.sh

./install-videos.sh

./create-storage-dirs-al.sh

echo "INSTALLATION COMPLETE" > ~/INSTALLATION_COMPLETE.txt
