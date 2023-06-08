FROM public.ecr.aws/panorama/panorama-application/sdkv1/python3.8/aarch64:latest
COPY src /panorama

# set Python3.8 as default
RUN apt-get update -y && apt-get install -y python3.8-dev libpython3.8 python3.8 python3.8-venv python3-pip 
RUN python3.8 -m pip install --user --upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --set python3 /usr/bin/python3.8

RUN apt-get update -y && apt-get install -y libcairo2 libcairo2-dev 
RUN python3.8 -m pip install --ignore-installed pycairo 

RUN python3.8 -m pip install opencv-python boto3

RUN apt-get update -y && apt-get -y install libgirepository1.0-dev
RUN python3.8 -m pip install --ignore-installed PyGObject

RUN apt-get update && \
	apt-get install -y \
	libgirepository1.0-dev gcc libcairo2-dev pkg-config gir1.2-gtk-3.0 \
	cmake \
	curl \
	g++ \
	gcc \
	git \
	gstreamer1.0-plugins-base-apps \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-tools \
	gstreamer1.0-omx \
	libglib2.0-0 \
	libssl-dev \
	libcurl4-openssl-dev \
	liblog4cplus-dev \
	libgstreamer1.0-dev \
    gstreamer1.0-plugins-bad-videoparsers \
	libgstreamer-plugins-base1.0-dev \
	m4 \
	make \
	openssh-server \
	pkg-config \
	vim \
	ninja-build

WORKDIR /opt/
RUN	git clone https://github.com/awslabs/amazon-kinesis-video-streams-producer-sdk-cpp.git
WORKDIR /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/
RUN cmake -G "Ninja" .. -DBUILD_GSTREAMER_PLUGIN=ON -DBUILD_DEPENDENCIES=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	ninja
