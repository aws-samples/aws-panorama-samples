FROM public.ecr.aws/panorama/panorama-application/sdkv1/python3.8/aarch64:latest
COPY src /panorama
RUN python3.8 -m pip install opencv-python boto3
