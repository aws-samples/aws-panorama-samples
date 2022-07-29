#!/bin/bash -xe


#sudo yum install docker -y
sudo apt install docker.io -y

#sudo yum install qemu binfmt-support qemu-user-static -y
# binfmt-support is not found, but that error is ignorable

#sudo docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

sudo gpasswd -a ubuntu docker
# FIXME : reboot needed?

#sudo newgrp docker

