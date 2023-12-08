FROM public.ecr.aws/panorama/panorama-application:latest

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.7 main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 r32.7 main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc

#
# Install Cuda, cuDNN, TensorRT
#
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    cuda-libraries-10-2=10.2.460-1 \
    cuda-libraries-dev-10-2=10.2.460-1 \
    cuda-minimal-build-10-2=10.2.460-1 \
    cuda-nvtx-10-2=10.2.300-1 \
    libcudnn8=8.2.1.32-1+cuda10.2 \
    libcudnn8-dev=8.2.1.32-1+cuda10.2 \
    tensorrt=8.2.1.9-1+cuda10.2 \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    python3-pip \
    wget build-essential cmake git pkg-config libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libpython3.7-dev gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad-videoparsers gstreamer1.0-plugins-ugly m4 \
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

RUN ln -s /usr/local/cuda-10.2 /usr/local/cuda

# Install Pytorch cuda
ARG PYTORCH_WHL=torch-1.10.0-cp37-cp37m-linux_aarch64.whl
RUN wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/PT_Sample/Docker/torch-1.10.0-cp37-cp37m-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir --verbose ${PYTORCH_WHL} && rm ${PYTORCH_WHL}

ARG TORCHVISION_WHL=torchvision-0.10.0a0+300a8a4-cp37-cp37m-linux_aarch64.whl
RUN wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/PT_Sample/Docker/torchvision-0.10.0a0%2B300a8a4-cp37-cp37m-linux_aarch64.whl -O ${TORCHVISION_WHL} \
    && python3 -m pip install --no-cache-dir --pre --verbose ${TORCHVISION_WHL} && rm ${TORCHVISION_WHL}

#compile opencv for gstreamer activation
RUN pip3 install numpy --upgrade --ignore-installed --no-cache-dir
RUN git clone https://github.com/opencv/opencv.git -b 4.5.5 --depth 1
RUN mkdir opencv/build
RUN cd opencv/build && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D WITH_FFMPEG=OFF -D WITH_TBB=OFF -D WITH_IPP=OFF BUILD_IPP_IW=OFF -D BUILD_ITT=OFF -D WITH_1394=OFF -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_QT=OFF -D WITH_GTK=OFF -D WITH_OPENGL=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_GSTREAMER=ON -D BUILD_opencv_python2=OFF -D WITH_OPENCL=OFF -D BUILD_WITH_STATIC_CRT=ON -D BUILD_SHARED_LIBS=OFF -D WITH_V4L=OFF -D PYTHON3_INCLUDE_DIR=/usr/include/python3.7m -D PYTHON3_PACKAGES_PATH=/usr/lib/python3.7/dist-packages .. && make -j8 && make install && ldconfig
RUN rm -rf /opencv

#install kvssink
RUN git clone https://github.com/awslabs/amazon-kinesis-video-streams-producer-sdk-cpp.git
RUN mkdir -p amazon-kinesis-video-streams-producer-sdk-cpp/build && cd amazon-kinesis-video-streams-producer-sdk-cpp/build && cmake -DBUILD_GSTREAMER_PLUGIN=TRUE -DBUILD_TEST=FALSE .. && make -j${nproc} && make install && ldconfig

RUN pip3 install boto3 lap cython --no-cache-dir
RUN pip3 install cython-bbox scipy --no-cache-dir
COPY kvs_log_configuration /amazon-kinesis-video-streams-producer-sdk-cpp

# Install bazel, dependency for PyTorch TensorRT
RUN wget https://github.com/bazelbuild/bazel/releases/download/6.3.2/bazel-6.3.2-linux-arm64 --no-check-certificate -O /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel

# Build and install PyTorch TensorRT from source
RUN git clone --branch v1.0.0 https://github.com/pytorch/TensorRT /TensorRT
# Use locally installed dependencies instead of pulling from remote
COPY tensorrt_patch.txt /TensorRT/tensorrt_patch.txt
RUN git -C /TensorRT apply tensorrt_patch.txt
WORKDIR /TensorRT/py
RUN python3 setup.py bdist_wheel --use-cxx11-abi
RUN python3 -mpip install dist/torch_tensorrt-1.0.0+55c3bab41-cp37-cp37m-linux_aarch64.whl --no-cache-dir
# Save space after install
WORKDIR /
RUN rm -rf /TensorRT

# There is a /tmp/hsperfdata_root entry causing problems at container launch time, let's remove it as it is not needed
RUN rm -rf /tmp/*
COPY src /panorama
