#!/bin/bash

# install dev foundation

sudo apt -y update
sudo apt -y upgrade

sudo apt install unzip
sudo apt install libssl-dev # for cmake

# install Python3.7 and pip
sudo apt install python3.7-dev -y
sudo apt install python3-pip
sudo python3.7 -m pip install pip --upgrade

sudo python3.7 -m pip install pexpect # for jupyterlab
sudo python3.7 -m pip install boto3
sudo python3.7 -m pip install awscli
sudo python3.7 -m pip install cython # to build numpy
sudo python3.7 -m pip install numpy
sudo python3.7 -m pip install sagemaker
sudo python3.7 -m pip install panoramacli
sudo python3.7 -m pip install cppy # to build matplotlib
sudo python3.7 -m pip install Pillow
sudo python3.7 -m pip install matplotlib
sudo python3.7 -m pip install jupyterlab # failes with pexpect uninstall

./install-docker.sh

./install-cmake3.sh

./install-dlr.sh

./install-videos.sh

./create-opt-aws-panorama.sh
