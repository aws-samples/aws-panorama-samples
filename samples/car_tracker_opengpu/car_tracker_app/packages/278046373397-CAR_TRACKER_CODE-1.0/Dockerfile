FROM public.ecr.aws/panorama/panorama-application:1.2.1-arm64v8
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ARG CUDA=11.4

RUN echo "deb https://repo.download.nvidia.com/jetson/common r35.3 main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 r35.3 main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc

# Install Cuda, cuDNN, TensorRT
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
RUN CUDAPKG=$(echo $CUDA | sed 's/\./-/'); \
    apt-get update -y && apt-get install -y --no-install-recommends \
    cuda-libraries-$CUDAPKG \
    cuda-nvtx-$CUDAPKG \
    cuda-libraries-dev-$CUDAPKG \
    cuda-minimal-build-$CUDAPKG \
    libcudnn8 \
    tensorrt \
    python3-dev \
    python3.8-dev \
    libpython3.8-dev \
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
    && rm -rf /usr/local/cuda-$CUDA/targets/aarch64-linux/lib/*.a \
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
RUN echo "/usr/local/cuda-$CUDA/targets/aarch64-linux/lib" >> /etc/ld.so.conf.d/nvidia.conf
RUN ln -s /usr/local/cuda-$CUDA /usr/local/cuda && \
    ln -s /usr/local/cuda-$CUDA/targets/aarch64-linux/include /usr/local/cuda/include && \
    ln -s /usr/local/cuda-$CUDA/targets/aarch64-linux/lib /usr/local/cuda/lib64

ARG HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/"

#
# Install tensorflow dependencies
#
RUN python3.8 -m pip install --no-cache-dir setuptools Cython==0.29.36 wheel
RUN python3.8 -m pip install --no-cache-dir --verbose numpy
RUN python3.8 -m pip install --no-cache-dir --verbose future==0.18.2 mock==3.0.5 h5py==3.11.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11


ARG TENSORFLOW_WHL=tensorflow-2.12.0+nv23.05-cp38-cp38-linux_aarch64.whl
RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v511/tensorflow/${TENSORFLOW_WHL} -O ${TENSORFLOW_WHL} && \
    pip3 install --no-cache-dir --verbose ${TENSORFLOW_WHL} && \
    rm ${TENSORFLOW_WHL}

RUN python3.8 -m pip install --upgrade --no-cache-dir numpy
RUN python3.8 -m pip install -U scipy --no-cache-dir
RUN python3.8 -m pip install opencv-python boto3 --no-cache-dir

COPY src /panorama
