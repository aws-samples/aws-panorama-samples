FROM public.ecr.aws/panorama/panorama-application:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean 

ARG CUDA=10.2
# r32.7 for JP4.6.2, with TRT8.2
ARG RELEASE=r32.7

RUN echo "deb https://repo.download.nvidia.com/jetson/common $RELEASE main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 $RELEASE main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc

ENV DEBIAN_FRONTEND noninteractive



#
# Install Cuda, cuDNN
#
RUN CUDAPKG=$(echo $CUDA | sed 's/\./-/'); \
    apt-get update && apt-get install -y --no-install-recommends \
	cuda-libraries-$CUDAPKG \
	cuda-nvtx-$CUDAPKG \
	cuda-libraries-dev-$CUDAPKG \
	cuda-minimal-build-$CUDAPKG \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /usr/local/cuda-$CUDA/targets/aarch64-linux/lib/*.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/*static*

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


# install prerequisites - https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29
# python3 dev is for numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-base \
        libopenmpi-dev \
        libomp-dev \
        python3-pip \
        wget \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean


######################
#      OpenCV        #
######################

# # (Optino1) Build OpenCV with Cudnn
# # OpenCV - https://github.com/mdegans/nano_build_opencv/blob/master/build_opencv.sh
# #
# ARG OPENCV_VERSION="4.4.0"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
# 	    gfortran \
#         cmake \
#         git \
# 	    file \
# 	    tar \
# 	    python3-pip \
# 	    python3-dev \
# 	    python3-distutils \
# 	    python3-setuptools \
#         libatlas-base-dev \
#         libavcodec-dev \
#         libavformat-dev \
#         libavresample-dev \
#         libcanberra-gtk3-module \
#         libdc1394-22-dev \
#         libeigen3-dev \
#         libglew-dev \
#         libgstreamer-plugins-base1.0-dev \
#         libgstreamer-plugins-good1.0-dev \
#         libgstreamer1.0-dev \
#         libgtk-3-dev \
#         libjpeg-dev \
#         libjpeg8-dev \
#         libjpeg-turbo8-dev \
#         liblapack-dev \
#         liblapacke-dev \
#         libopenblas-dev \
#         libpng-dev \
#         libpostproc-dev \
#         libswscale-dev \
#         libtbb-dev \
#         libtbb2 \
#         libtesseract-dev \
#         libtiff-dev \
#         libv4l-dev \
#         libxine2-dev \
#         libxvidcore-dev \
#         libx264-dev \
# 	    libgtkglext1 \
# 	    libgtkglext1-dev \
#         pkg-config \
#         qv4l2 \
#         v4l-utils \
#         v4l2ucp \
#         zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# # OpenCV looks for the cuDNN version in cudnn_version.h, but it's been renamed to cudnn_version_v8.h
# RUN ln -s /usr/include/aarch64-linux-gnu/cudnn_version_v8.h /usr/include/aarch64-linux-gnu/cudnn_version.h

# RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
#     git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git && \
#     cd opencv && \
#     mkdir build && \
#     cd build && \
#     cmake \
#         -D CPACK_BINARY_DEB=ON \
# 	    -D BUILD_EXAMPLES=OFF \
#         -D BUILD_opencv_python2=OFF \
#         -D BUILD_opencv_python3=ON \
# 	    -D BUILD_opencv_java=OFF \
#         -D CMAKE_BUILD_TYPE=RELEASE \
#         -D CMAKE_INSTALL_PREFIX=/usr/local \
#         -D CUDA_ARCH_BIN=5.3,6.2,7.2 \
#         -D CUDA_ARCH_PTX= \
#         -D CUDA_FAST_MATH=ON \
#         -D CUDNN_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
#         -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
# 	    -D WITH_EIGEN=ON \
#         -D ENABLE_NEON=ON \
#         -D OPENCV_DNN_CUDA=ON \
#         -D OPENCV_ENABLE_NONFREE=ON \
#         -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
#         -D OPENCV_GENERATE_PKGCONFIG=ON \
#         -D WITH_CUBLAS=ON \
#         -D WITH_CUDA=ON \
#         -D WITH_CUDNN=ON \
#         -D WITH_GSTREAMER=ON \
#         -D WITH_LIBV4L=ON \
#         -D WITH_OPENGL=ON \
# 	    -D WITH_OPENCL=OFF \
# 	    -D WITH_IPP=OFF \
#         -D WITH_TBB=ON \
# 	    -D BUILD_TIFF=ON \
# 	    -D BUILD_PERF_TESTS=OFF \
# 	    -D BUILD_TESTS=OFF \
# 	    ../

# RUN cd opencv/build && make -j$(nproc)
# RUN cd opencv/build && make install
# RUN cd opencv/build && make package

# RUN cd opencv/build && tar -czvf OpenCV-${OPENCV_VERSION}-aarch64.tar.gz *.deb


# (Optino2) pip install
RUN python3.7 -m pip install --no-cache-dir opencv-python


######################
#      Pytorch       #
######################
# Install Pytorch cuda
RUN echo "install Torch and Torchvision Cuda"
ENV DEBIAN_FRONTEND=noninteractive


ARG PYTORCH_WHL=torch-1.10.0-cp37-cp37m-linux_aarch64.whl
RUN wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/PT_Sample/Docker/torch-1.10.0-cp37-cp37m-linux_aarch64.whl \
    && python3.7 -m pip install --no-cache-dir --verbose ${PYTORCH_WHL} && rm ${PYTORCH_WHL}

#######################
# torchvision 0.10.0  #
#######################
# RUN python3.7 -m pip install --no-cache-dir setuptools Cython wheel
# ARG FORCE_CUDA=1
# ARG TORCHVISION_VERSION=v0.10.0
# ARG PILLOW_VERSION=pillow<7
# ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
# RUN printenv && echo "torchvision version = $TORCHVISION_VERSION" && echo "pillow version = $PILLOW_VERSION" && echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         git \
#         build-essential \
#         libjpeg-dev \
#         zlib1g-dev \
#         python3.7-dev \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean
# RUN git clone -b ${TORCHVISION_VERSION} https://github.com/pytorch/vision torchvision

## There are three options:
# (Option1) Install Torchvision
# RUN cd torchvision && \
#     python3.7 setup.py install && \
#     cd ../  && \
#     rm -rf torchvision

# (Option2) Build Torchvision wheel & Install
# You can get the wheel from /torchvision/dist/torchvision-0.10.0a0+300a8a4-cp37-cp37m-linux_aarch64.whl
# RUN cd torchvision && \
#     python3.7 setup.py bdist_wheel

# (Option3) Install from the wheel directly (We prebuilt it already)
ARG TORCHVISION_WHL=torchvision-0.10.0a0+300a8a4-cp37-cp37m-linux_aarch64.whl
RUN wget https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/PT_Sample/Docker/torchvision-0.10.0a0%2B300a8a4-cp37-cp37m-linux_aarch64.whl -O ${TORCHVISION_WHL} \
    && python3.7 -m pip install --pre --verbose ${TORCHVISION_WHL} && rm ${TORCHVISION_WHL}

# The built torch vision 
# note:  this was used on older torchvision versions (~0.4) to workaround a bug,
#        but has since started causing another bug as of torchvision 0.11.1
# pip3 install --no-cache-dir "${PILLOW_VERSION}"
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         software-properties-common \
#         apt-transport-https \
#         ca-certificates \
#         gnupg \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# patch for https://github.com/pytorch/pytorch/issues/45323
ARG TORCH_CMAKE_CONFIG=/usr/local/lib/python3.7/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake
RUN echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    echo "contents of ${TORCH_CMAKE_CONFIG} before patching..." && \
    cat ${TORCH_CMAKE_CONFIG} && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG} && \
    echo "contents of ${TORCH_CMAKE_CONFIG} after patching..." && \
    cat ${TORCH_CMAKE_CONFIG}

#Required to run to create the Python_egg cache, since panorama is a read-only container. May get a numpy initialization warning.

ARG PYTHON_EGG_CACHE="/panorama/.cache"
RUN python3.7 -c "import torch; \
import site; \
site.addsitedir('/usr/lib/python3.7/site-packages/'); \
import torchvision"