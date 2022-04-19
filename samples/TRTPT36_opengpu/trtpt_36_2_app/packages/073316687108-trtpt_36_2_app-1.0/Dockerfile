# Use the pre-built docker image attached in this example by doing ```docker load --input panoramasdk_gpu_access_base_image.tar.gz``` or build the base image yourself using the dockerfile provided under /docker/Dockerfile
FROM trtpt36:latest
RUN apt-get update && apt-get install -y libglib2.0-0
RUN python3.6 -m pip install boto3
COPY src /panorama
