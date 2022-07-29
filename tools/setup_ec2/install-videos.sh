#!/bin/bash

wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/sample-videos.zip
unzip sample-videos.zip
mv videos/* ../../samples/common/test_utility/videos/
rmdir videos
rm sample-videos.zip
