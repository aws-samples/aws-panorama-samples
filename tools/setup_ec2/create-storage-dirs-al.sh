#!/bin/bash -xe

sudo mkdir -p /opt/aws
sudo mkdir -p /opt/aws/panorama
sudo mkdir -p /opt/aws/panorama/storage
sudo mkdir -p /opt/aws/panorama/logs
sudo chown ec2-user.ec2-user /opt/aws/panorama/*
