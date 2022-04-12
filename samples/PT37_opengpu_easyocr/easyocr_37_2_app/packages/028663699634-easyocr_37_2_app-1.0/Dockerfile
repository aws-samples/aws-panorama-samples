# Use the pre-built docker image attached in this example by doing ```docker load --input panoramasdk_gpu_access_base_image.tar.gz``` or build the base image yourself using the dockerfile provided under /docker/Dockerfile
FROM pt:37
RUN apt-get update && apt-get install -y libglib2.0-0
RUN python3.7 -m pip install boto3
COPY src /panorama
# COPY saved_model_trt_fp16 /panorama/saved_model_trt_fp16

# For easyOCR
RUN pip install opencv-python-headless==4.5.4.60
RUN pip install easyocr
RUN pip install -U scikit-image==0.17.2
RUN python3 -c 'import easyocr; easyocr.Reader(["en"], gpu=False);'
RUN chmod -R 777 /root/
RUN ls /root/.EasyOCR/model/ -lrt
