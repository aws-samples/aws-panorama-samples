FROM public.ecr.aws/panorama/panorama-application:1.1.0-arm64v8

#compile opencv for gstreamer activation
RUN apt-get update -y
RUN pip3 install numpy --upgrade --ignore-installed
RUN apt-get install -y wget build-essential cmake git pkg-config libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt-get install -y libpython3.7-dev

#install opencv
ENV VERSION=4.5.5
RUN git clone https://github.com/opencv/opencv.git -b $VERSION --depth 1
RUN mkdir opencv/build
RUN cd opencv/build && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D WITH_FFMPEG=OFF -D WITH_TBB=OFF -D WITH_IPP=OFF BUILD_IPP_IW=OFF -D BUILD_ITT=OFF -D WITH_1394=OFF -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_QT=OFF -D WITH_GTK=OFF -D WITH_OPENGL=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_GSTREAMER=ON -D BUILD_opencv_python2=OFF -D WITH_OPENCL=OFF -D BUILD_WITH_STATIC_CRT=ON -D BUILD_SHARED_LIBS=OFF -D WITH_V4L=OFF -D PYTHON3_INCLUDE_DIR=/usr/include/python3.7m -D PYTHON3_PACKAGES_PATH=/usr/lib/python3.7/dist-packages .. && make -j8 && make install && ldconfig
RUN rm -rf /opencv

#install gstreamer plugins
RUN apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad-videoparsers gstreamer1.0-plugins-ugly

#install kvssink
RUN git clone https://github.com/awslabs/amazon-kinesis-video-streams-producer-sdk-cpp.git
RUN mkdir -p amazon-kinesis-video-streams-producer-sdk-cpp/build && cd amazon-kinesis-video-streams-producer-sdk-cpp/build && cmake -DBUILD_GSTREAMER_PLUGIN=TRUE -DBUILD_TEST=FALSE .. && make && make install && ldconfig

#add env
RUN rm -rf 
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/amazon-kinesis-video-streams-producer-sdk-cpp/open-source/local/lib"
ENV GST_PLUGIN_PATH "$GST_PLUGIN_PATH:/usr/local/lib/gstreamer-1.0/:/amazon-kinesis-video-streams-producer-sdk-cpp/build"

#Copy app finally
RUN pip3 install boto3
RUN pip3 install torch==1.8.1 torchvision==0.9.1
RUN pip3 install lap
RUN pip3 install cython
RUN pip3 install cython-bbox
RUN pip3 install scipy
COPY kvs_log_configuration /amazon-kinesis-video-streams-producer-sdk-cpp
COPY src /panorama