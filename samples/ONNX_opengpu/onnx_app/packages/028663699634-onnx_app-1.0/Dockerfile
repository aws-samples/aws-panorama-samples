FROM public.ecr.aws/panorama/panorama-application:1.2.1-arm64v8

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates

ARG CUDA=11.4
ARG RELEASE=r35.3
RUN echo "deb https://repo.download.nvidia.com/jetson/common $RELEASE main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 $RELEASE main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc

RUN CUDAPKG=$(echo $CUDA | sed 's/\./-/'); \
    apt-get update && apt-get install -y --no-install-recommends \
    awscli \
    cuda-libraries-$CUDAPKG \
    cuda-nvtx-$CUDAPKG \
    cuda-libraries-dev-$CUDAPKG \
    cuda-minimal-build-$CUDAPKG \
    gfortran \
    git \
    libcudnn8 \
    libglib2.0-0 \
    libjpeg-dev \
    libopenblas-dev \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    openmpi-bin \
    openmpi-common \
    python3-pip \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /usr/local/cuda-$CUDA/targets/aarch64-linux/lib/*.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/*static*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
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
ENV PATH /usr/local/cuda-$CUDA/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-$CUDA/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}
        
######################
#    OnnxRuntime     #
######################        
# the whl file can be found from https://elinux.org/Jetson_Zoo#ONNX_Runtime
ARG ONNX_WHL=onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
RUN wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O ${ONNX_WHL} \
    && python3.8 -m pip install --verbose ${ONNX_WHL} && rm ${ONNX_WHL}

######################
#      OpenCV        #
######################
RUN python3.8 -m pip install --no-cache-dir Cython opencv-python six

#########################
# Pytorch 2.0.0+nv23.05 #
#########################
RUN python3.8 -m pip install --no-cache-dir setuptools

ARG TORCH_WHL=torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
RUN wget https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/${TORCH_WHL} -O ${TORCH_WHL} \
   && python3.8 -m pip install --no-cache --verbose ${TORCH_WHL} \
   && rm ${TORCH_WHL}
RUN python3.8 -c 'import torch'

#######################
# torchvision 0.15.1  #
#######################
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda-$CUDA
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7"
ARG TORCHVISION_VERSION="v0.15.1"
    
# It takes about 15 mins to build from source
RUN git clone --branch ${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    git checkout ${TORCHVISION_VERSION} && \
    python3.8 setup.py bdist_wheel && \
    cp dist/torchvision*.whl /opt && \
    python3.8 -m pip install --no-cache-dir --verbose /opt/torchvision*.whl && \
    cd ../ && \
    rm -rf torchvision

# patch for https://github.com/pytorch/pytorch/issues/45323
ARG TORCH_CMAKE_CONFIG=/usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake
RUN echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    echo "contents of ${TORCH_CMAKE_CONFIG} before patching..." && \
    cat ${TORCH_CMAKE_CONFIG} && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG} && \
    echo "contents of ${TORCH_CMAKE_CONFIG} after patching..." && \
    cat ${TORCH_CMAKE_CONFIG}

# Required to run to create the Python_egg cache, since panorama is a read-only container. May get a numpy initialization warning.
ARG PYTHON_EGG_CACHE="/panorama/.cache"
RUN python3.8 -c "import torch; import torchvision"
RUN python3.8 -m pip install --no-cache-dir boto3
COPY src /panorama
