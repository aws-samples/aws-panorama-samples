FROM public.ecr.aws/panorama/panorama-application:latest
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 r32.4 main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc

# Install Cuda, cuDNN, TensorRT
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    cuda-libraries-10-2 \
    cuda-nvtx-10-2 \
    cuda-libraries-dev-10-2 \
    cuda-minimal-build-10-2 \
    libcudnn8 \
    tensorrt \
    python3-dev \
    python3.7-dev \
    libpython3.7-dev \
    python3-pip \
    gfortran \
    liblapack-dev \
    libblas-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /usr/local/cuda-10.2/targets/aarch64-linux/lib/*.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/*static*

RUN echo "/usr/lib/aarch64-linux-gnu/tegra" >> /etc/ld.so.conf.d/nvidia-tegra.conf && \
    echo "/usr/lib/aarch64-linux-gnu/tegra-egl" >> /etc/ld.so.conf.d/nvidia-tegra.conf
RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ && echo '\
{\
    "file_format_version" : "1.0.0",\
    "ICD" : {\
        "library_path" : "libEGL_nvidia.so.0"\
    }\
}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN mkdir -p /usr/share/egl/egl_external_platform.d/ && echo '\
{\
    "file_format_version" : "1.0.0",\
    "ICD" : {\
        "library_path" : "libnvidia-egl-wayland.so.1"\
    }\
}' > /usr/share/egl/egl_external_platform.d/nvidia_wayland.json
RUN echo "/usr/local/cuda-10.2/targets/aarch64-linux/lib" >> /etc/ld.so.conf.d/nvidia.conf
RUN ln -s /usr/local/cuda-10.2 /usr/local/cuda && \
    ln -s /usr/local/cuda-10.2/targets/aarch64-linux/include /usr/local/cuda/include && \
    ln -s /usr/local/cuda-10.2/targets/aarch64-linux/lib /usr/local/cuda/lib64

ARG HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/"

#
# Install tensorflow dependencies
#
RUN python3 -m pip install --no-cache-dir setuptools Cython==0.29.36 wheel
RUN python3 -m pip install --no-cache-dir --verbose numpy
RUN python3 -m pip install --no-cache-dir --verbose future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

#
# TensorFlow 2.4. See build instructions for building tensorflow wheel here: https://apivovarov.medium.com/run-tensorflow-2-object-detection-models-with-tensorrt-on-jetson-xavier-using-tf-c-api-e34548818ac6
#
ARG TENSORFLOW_WHL=tensorflow-2.4.4-cp37-cp37m-linux_aarch64.whl
RUN wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/TF_Sample/Docker/tensorflow-2.4.4-cp37-cp37m-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir --pre --verbose ${TENSORFLOW_WHL} \
    && rm ${TENSORFLOW_WHL}

RUN python3 -m pip install --upgrade --no-cache-dir numpy
RUN python3 -m pip install -U scipy --no-cache-dir
RUN pip3 install opencv-python boto3 --no-cache-dir

COPY src /panorama
