# Use the pre-built docker image attached in this example by doing ```docker load --input panoramasdk_gpu_access_base_image.tar.gz``` or build the base image yourself using the dockerfile provided under /docker/Dockerfile
FROM 201125699002.dkr.ecr.ap-southeast-1.amazonaws.com/trtpt36_opengpu
RUN apt-get update && apt-get install -y libglib2.0-0
RUN python3.6 -m pip install boto3
COPY src /panorama

# To build the docker file on x86 instance
# docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
# panorama-cli build \
#  --container-asset-name onnx2trt_app \
#  --package-path packages/accountid-onnx2trt_app-1.0/