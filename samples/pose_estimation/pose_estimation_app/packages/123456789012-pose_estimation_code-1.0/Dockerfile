FROM public.ecr.aws/panorama/panorama-application:latest

# Use Python3.6. Install issues with 3.8 / 3.7
RUN python3.7 -m pip uninstall -y numpy \
    && rm /usr/bin/python3 \
    && ln -s /usr/bin/python3.6 /usr/bin/python3 \
    && apt-get remove -y python3.7

RUN apt-get update -y && apt-get install -y gnupg2
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 r32.4 main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.7 main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 r32.7 main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc
RUN mkdir -p /opt/nvidia/l4t-packages/
RUN touch /opt/nvidia/l4t-packages/.nv-l4t-disable-boot-fw-update-in-preinstall
RUN apt-get update -y && apt-get install build-essential \
    autoconf automake libtool curl make g++ unzip \
    pkg-config \
    swig \
    git \
    ccache \
    ninja-build \
    wget \
    libopenblas-dev \
    libopencv-dev \
    cuda-libraries-10-2=10.2.89-1 \
    cuda-libraries-dev-10-2=10.2.89-1 \
    cuda-minimal-build-10-2=10.2.89-1 \
    libcudnn8=8.2.1.32-1+cuda10.2 \
    libcudnn8-dev=8.2.1.32-1+cuda10.2 \
    python3-pip \
    libnvinfer-bin=7.1.3-1+cuda10.2 \
    libnvinfer-dev=7.1.3-1+cuda10.2 \
    libnvinfer-plugin-dev=7.1.3-1+cuda10.2 \
    libnvparsers-dev=7.1.3-1+cuda10.2 \
    libnvonnxparsers-dev=7.1.3-1+cuda10.2 \
    libnvinfer-samples=7.1.3-1+cuda10.2 \
    libnvinfer-doc=7.1.3-1+cuda10.2\
    tensorrt=7.1.3.0-1+cuda10.2 \
    nvidia-l4t-core=32.7.2-20220417024839 \
    nvidia-l4t-multimedia=32.7.2-20220417024839 \
    nvidia-l4t-cuda=32.7.2-20220417024839 \
    nvidia-l4t-multimedia-utils=32.7.2-20220417024839 \
    nvidia-l4t-3d-core=32.7.2-20220417024839 \
    nvidia-l4t-firmware=32.7.2-20220417024839 \
    nvidia-l4t-init=32.7.2-20220417024839 \
    nvidia-l4t-libvulkan=32.7.2-20220417024839 \
    nvidia-l4t-wayland=32.7.2-20220417024839 \
    nvidia-l4t-x11=32.7.2-20220417024839 \
    cuda-nvprof-10-2=10.2.300-1 -y \
    # Uninstall protobuf to install from source later. Source needed for dev libraries of same version.
    && apt-get remove libprotobuf10 -y \
    # Clean up to save space
    && rm -rf /var/lib/apt/lists/* && apt-get clean \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/*static*

RUN ln -sfn /usr/local/cuda-10.2 /usr/local/cuda
RUN ln -sfn /usr/bin/python3 /usr/bin/python

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-aarch64.tar.gz -O /root/cmake-3.27.7-linux-aarch64.tar.gz
WORKDIR /root/
RUN tar -xvzf cmake-3.27.7-linux-aarch64.tar.gz && rm cmake-3.27.7-linux-aarch64.tar.gz
RUN ln -s /root/cmake-3.27.7-linux-aarch64/bin/cmake /usr/bin/cmake

RUN python3 -mpip install --no-cache-dir --upgrade pip
RUN python3 -mpip install --no-cache-dir --upgrade Cython numpy setuptools boto3 opencv-python

# Get the same protobuf library that the panorama sdk looks for. Install from source
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.12.4/protobuf-all-3.12.4.tar.gz -P /root/
WORKDIR /root/
RUN tar -xvzf protobuf-all-3.12.4.tar.gz && rm protobuf-all-3.12.4.tar.gz
WORKDIR /root/protobuf-3.12.4/
RUN ./autogen.sh
RUN ./configure
RUN make -j$(nproc)
RUN make install
RUN ldconfig
RUN ln -sfn /usr/local/lib/libprotobuf.so.23.0.4 /usr/lib/aarch64-linux-gnu/libprotobuf.so.23

ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64/stubs:{$LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV OPENBLAS_CORETYPE=ARMV8

RUN git clone --recursive -b v1.6.x https://github.com/apache/mxnet.git /root/mxnet
COPY v1.6.x.patch /root/mxnet/v1.6.x.patch
RUN git -C /root/mxnet apply v1.6.x.patch
COPY v1.6.x_onnx_tensorrt.patch /root/mxnet/3rdparty/onnx-tensorrt/v1.6.x_onnx_tensorrt.patch
RUN git -C /root/mxnet/3rdparty/onnx-tensorrt/ apply v1.6.x_onnx_tensorrt.patch

WORKDIR /root/mxnet
RUN ./ci/docker/runtime_functions.sh build_ubuntu_gpu_tensorrt
RUN python3 -mpip install --no-cache-dir --upgrade /root/mxnet/python
RUN ln -sfn /usr/local/mxnet/libmxnet.so /usr/local/lib/python3.6/dist-packages/mxnet/

# libcuda.so.1 missing from ldd libmxnet.so
RUN ln -sfn /usr/lib/aarch64-linux-gnu/tegra/libcuda.so /usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1

# libonnx_proto.so missing from ldd libmxnet.so
# copy to somewhere on the LD_LIBRARY_PATH, instead of adding the mxnet lib folder, since we want to delete it for space.
RUN cp /root/mxnet/lib/libonnx_proto.so /usr/lib/aarch64-linux-gnu/tegra/libonnx_proto.so

# https://stackoverflow.com/questions/47258882/theano-gpu-support-ptx-jit-compiler-library-not-found
RUN ln -sfn /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.440.18 /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.1

# Cleanup to save space
WORKDIR /
RUN rm -rf /root/mxnet
RUN rm -rf /root/protobuf-3.12.4
RUN rm -rf /root/cmake-3.27.7-linux-aarch64
RUN rm -rf /usr/local/cuda-10.2/targets/aarch64-linux/lib/*.a
COPY src /panorama
