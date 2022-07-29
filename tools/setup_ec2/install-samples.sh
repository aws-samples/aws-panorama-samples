#!/bin/bash

#cd $HOME
#git clone https://github.com/aws-samples/aws-panorama-samples.git

wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/sample-videos.zip
unzip sample-videos.zip
mv videos/* ./aws-panorama-samples/samples/common/test_utility/videos/
rmdir videos
rm sample-videos.zip

