#!/bin/bash -xe

# install dev foundation

sudo apt -y update
sudo apt -y upgrade

#sudo apt install unzip -y
#sudo apt install libssl-dev -y # for cmake

# install Python3.7 and pip
#sudo apt install python3.7-dev -y
#sudo apt install python3-pip -y
#sudo python3.7 -m pip install pip --upgrade

sudo pip3 install pip --upgrade

# uninstall old version of expect which is not compatible with python3.7
#sudo apt-get remove python-pexpect python3-pexpect -y

sudo pip3 install boto3
sudo pip3 install awscli
sudo pip3 install numpy
sudo pip3 install sagemaker
sudo pip3 install panoramacli
sudo pip3 install matplotlib
sudo pip3 install jupyterlab

#./install-docker.sh

#./install-cmake3.sh

./install-dlr-g5g.sh

./install-videos.sh

./create-storage-dirs.sh

echo "INSTALLATION COMPLETE" > ~/INSTALLATION_COMPLETE.txt
