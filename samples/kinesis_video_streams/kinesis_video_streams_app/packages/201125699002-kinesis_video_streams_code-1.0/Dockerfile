FROM public.ecr.aws/panorama/panorama-application
COPY src /panorama

RUN apt-get update && \
	apt-get install -y \
	python3.7-dev \
	libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 \
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

RUN python3 -m pip install --upgrade pip
RUN pip3 install opencv-python boto3
RUN python3 -m pip install --ignore-installed pycairo
RUN python3 -m pip install --ignore-installed PyGObject

WORKDIR /opt/
RUN	git clone https://github.com/awslabs/amazon-kinesis-video-streams-producer-sdk-cpp.git
WORKDIR /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/
RUN cmake -G "Ninja" .. -DBUILD_GSTREAMER_PLUGIN=ON -DBUILD_DEPENDENCIES=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	ninja