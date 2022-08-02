#!/bin/bash -xe

sudo yum install docker -y

# QEMU is needed for non-ARM architecture
# binfmt-support is not found, but that error is ignorable
#sudo yum install qemu binfmt-support qemu-user-static -y
#sudo docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

sudo gpasswd -a ec2-user docker
#sudo newgrp docker

sudo systemctl enable docker
