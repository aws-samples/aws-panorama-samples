FROM public.ecr.aws/panorama/panorama-application:latest

# Lets use Python3.6
RUN python3.7 -m pip uninstall -y numpy \
    && rm /usr/bin/python3 \
    && ln -s /usr/bin/python3.6 /usr/bin/python3 \
    && apt-get remove -y python3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates

ARG CUDA=10.2
ARG RELEASE=r32.4

RUN echo "deb https://repo.download.nvidia.com/jetson/common $RELEASE main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 $RELEASE main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc


#
# install prerequisites - https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
#
RUN apt-get update && \
 apt-get install -y --no-install-recommends \
       python3-pip \
       python3-dev \
       gfortran \
       build-essential \
       liblapack-dev \
       libblas-dev \
       libhdf5-serial-dev \
       hdf5-tools \
       libhdf5-dev \
       zlib1g-dev \
       zip \
       libjpeg8-dev \
       wget

#RUN mkdir -p /usr/lib/python3.6/lib-dynload
#RUN cp /usr/lib/python3.7/lib-dynload/panoramasdk.so /usr/lib/python3.6/lib-dynload/panoramasdk.so

#
# Install Cuda, cuDNN, TensorRT
#
RUN CUDAPKG=$(echo $CUDA | sed 's/\./-/'); \
    apt-get update && apt-get install -y --no-install-recommends \
	cuda-libraries-$CUDAPKG \
	cuda-nvtx-$CUDAPKG \
	cuda-libraries-dev-$CUDAPKG \
	cuda-minimal-build-$CUDAPKG \
	cuda-license-$CUDAPKG \
	cuda-command-line-tools-$CUDAPKG \
    libcudnn8 \
    tensorrt \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /usr/local/cuda-$CUDA/targets/aarch64-linux/lib/*.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a
    #Remove static libraries to save space


ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

RUN echo "/usr/lib/aarch64-linux-gnu/tegra" >> /etc/ld.so.conf.d/nvidia-tegra.conf && \
    echo "/usr/lib/aarch64-linux-gnu/tegra-egl" >> /etc/ld.so.conf.d/nvidia-tegra.conf

#RUN rm /usr/share/glvnd/egl_vendor.d/50_mesa.json
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

RUN ln -s /usr/local/cuda-$CUDA /usr/local/cuda && \
    ln -s /usr/local/cuda-$CUDA/targets/aarch64-linux/include /usr/local/cuda/include && \
    ln -s /usr/local/cuda-$CUDA/targets/aarch64-linux/lib /usr/local/cuda/lib64

ENV PATH /usr/local/cuda-$CUDA/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-$CUDA/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

ENV DEBIAN_FRONTEND=noninteractive
ARG HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/"
ARG MAKEFLAGS=-j$(nproc)

RUN printenv

#
# Install tensorflow dependencies
#
RUN python3.6 -m pip install --no-cache-dir setuptools Cython wheel
RUN python3.6 -m pip install --no-cache-dir numpy
#
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN python3.6 -m pip  install --upgrade pip
RUN python3.6 -m pip install --no-cache-dir --verbose pycuda six

#
# Run ld config
#
RUN ldconfig

#
# Set environment variables
# Note: These environment variables don't get persisted for runtime.
#
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

CMD [ "/bin/bash" ]


ARG TORCH_WHL=torch-1.10.0-cp36-cp36m-linux_aarch64.whl
RUN wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/TRT_Sample/Docker/torch-1.10.0-cp36-cp36m-linux_aarch64.whl \
    && python3.6 -m pip install --pre --verbose ${TORCH_WHL} \
    && rm ${TORCH_WHL}
    

RUN apt-get update -y && apt-get install -y libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN python3.6 -m pip install --no-cache-dir opencv-python

# TesnorRT
RUN apt-get update -y \
    && apt-get install libopenblas-base libopenmpi-dev python3-libnvinfer-dev -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

######################
# torchvision 0.10.0 #
######################
# There are two ways to install it. Either install if from source or install the 
# prebuilt whl we provide.

ARG PILLOW_VERSION=pillow<9.0.1
RUN python3.6 -m pip install --no-cache-dir "${PILLOW_VERSION}"

# # Install from source:
# ARG FORCE_CUDA=1
# ARG TORCHVISION_VERSION=v0.10.0
# ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
# RUN printenv && echo "torchvision version = $TORCHVISION_VERSION" && echo "pillow version = $PILLOW_VERSION" && echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#           git \
#           build-essential \
#             libjpeg-dev \
#           zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean
# RUN git clone -b ${TORCHVISION_VERSION} https://github.com/pytorch/vision torchvision
# RUN cd torchvision && \
#     python3.6 setup.py install && \
#     cd ../  && \
#     rm -rf torchvision

# # Install from whl
ARG TORCHVISION_WHL=torchvision-0.10.0a0+300a8a4-cp36-cp36m-linux_aarch64.whl
COPY ${TORCHVISION_WHL} .
RUN python3.6 -m pip install --pre --verbose ${TORCHVISION_WHL} && rm ${TORCHVISION_WHL}

# # note:  this was used on older torchvision versions (~0.4) to workaround a bug,
# #       but has since started causing another bug as of torchvision 0.11.1
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#           software-properties-common \
#           apt-transport-https \
#           ca-certificates \
#           gnupg \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# patch for https://github.com/pytorch/pytorch/issues/45323
ARG TORCH_CMAKE_CONFIG=/usr/local/lib/python3.6/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake
RUN echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    echo "contents of ${TORCH_CMAKE_CONFIG} before patching..." && \
    cat ${TORCH_CMAKE_CONFIG} && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG} && \
    echo "contents of ${TORCH_CMAKE_CONFIG} after patching..." && \
    cat ${TORCH_CMAKE_CONFIG}


ARG PYTHON_EGG_CACHE="/panorama/.cache"
RUN python3.6 -c "import torch; import torchvision"

RUN mkdir -p /tmp/test && \
    mkdir -p /panorama/test && \
    ln -s /tmp/test /panorama/test && \ 
    mkdir -p /panorama/storage-test && \
    ln -s /panorama/storage-test /opt/aws/panorama/storage/test