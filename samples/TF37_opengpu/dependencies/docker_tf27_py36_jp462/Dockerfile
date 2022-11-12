FROM public.ecr.aws/panorama/panorama-application:latest

# Lets use Python3.6
RUN python3.7 -m pip uninstall -y numpy \
    && rm /usr/bin/python3 \
    && ln -s /usr/bin/python3.6 /usr/bin/python3 \
    && apt-get remove -y python3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean 

# COPY jetson-ota-public.key /etc/jetson-ota-public.key
# RUN apt-key add /etc/jetson-ota-public.key

ARG CUDA=10.2
# r32.7 for JP4.6.2, with TRT8.2
ARG RELEASE=r32.7

RUN echo "deb https://repo.download.nvidia.com/jetson/common $RELEASE main" >> /etc/apt/sources.list
RUN echo "deb https://repo.download.nvidia.com/jetson/t194 $RELEASE main" >> /etc/apt/sources.list
RUN apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc


#
# install prerequisites - https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
#
RUN apt-get update && \
 apt-get install -y --no-install-recommends \
       python3-dev \
       python3-dev \
       libpython3-dev \
       python3-pip \
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
       wget \
       unzip \
       && rm -rf /var/lib/apt/lists/* \
       && apt-get clean

#RUN mkdir -p /usr/lib/python3.7/lib-dynload
#RUN cp /usr/lib/python3.7/lib-dynload/panoramasdk.so /usr/lib/python3.7/lib-dynload/panoramasdk.so

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
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/*static*


# patching for JP4.6.2 for Jetson NX 16GB
RUN mkdir /jp46_patch && cd /jp46_patch \
    && wget https://developer.nvidia.com/tensorrt-8219-patch-jetpack46-jetson-nx-16gbtargz -O  TensorRT8.2FixforJetsonNX16GB.tar.gz  \
    && tar zxvf TensorRT8.2FixforJetsonNX16GB.tar.gz \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvinfer8_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvinfer-plugin8_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvinfer-plugin-dev_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvonnxparsers8_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvonnxparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvparsers8_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvinfer-bin_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvinfer-doc_8.2.1-1+cuda10.2_all.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/libnvinfer-samples_8.2.1-1+cuda10.2_all.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/tensorrt_8.2.1.9-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/python3-libnvinfer_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/python3-libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/graphsurgeon-tf_8.2.1-1+cuda10.2_arm64.deb \
    && dpkg -i /jp46_patch/TensorRT_8.2.1.9_Patch_for_Jetpack4.6_Jetson_NX_16GB/uff-converter-tf_8.2.1-1+cuda10.2_arm64.deb \
    && rm -rf /usr/local/cuda-$CUDA/targets/aarch64-linux/lib/*.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a \
    && rm -rf /usr/lib/aarch64-linux-gnu/*static* \
    && rm -r /jp46_patch

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
RUN printenv

#
# Install tensorflow dependencies
# Following the tutorial here https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
RUN python3 --version
RUN python3.6 -m pip install -U --no-cache-dir pip testresources setuptools==49.6.0 
RUN python3.6 -m pip install --no-cache-dir Cython wheel
RUN python3.6 -m pip install --no-cache-dir -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging
RUN env H5PY_SETUP_REQUIRES=0 python3.6 -m pip install -U --no-cache-dir --verbose h5py==3.1.0
#
# TensorFlow 2.4. See build instructions for building tensorflow wheel here: https://apivovarov.medium.com/run-tensorflow-2-object-detection-models-with-tensorrt-on-jetson-xavier-using-tf-c-api-e34548818ac6
#
ARG TENSORFLOW_WHL=tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
RUN wget "https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/${TENSORFLOW_WHL}" -O ${TENSORFLOW_WHL} \
    && python3.6 -m pip install --pre --verbose ${TENSORFLOW_WHL} \
    && rm ${TENSORFLOW_WHL}


RUN python3 -c "import tensorflow as tf; print(tf.__version__)"
RUN nm -D /usr/lib/aarch64-linux-gnu/libnvinfer.so | grep tensorrt_version


#
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

# RUN python3.7 -m pip install --no-cache-dir --verbose pycuda six
# RUN pip3 install --upgrade numpy
#
# Run ld config
#
# RUN ldconfig

#
# Set environment variables
# Note: These environment variables don't get persisted for runtime.
#
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

CMD [ "/bin/bash" ]
