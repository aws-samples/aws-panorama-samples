import json
import logging
import time
import os
import re
from logging.handlers import RotatingFileHandler

import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import panoramasdk

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

#GST Environment Variables
os.environ['GST_PLUGIN_PATH'] = '/opt/amazon-kinesis-video-streams-producer-sdk-cpp/build'
os.environ['LD_LIBRARY_PATH'] = '/opt/amazon-kinesis-video-streams-producer-sdk-cpp/open-source/local/lib'

Gst.init(None)

class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.kvs_stream_name = self.inputs.kvs_stream_name.get()
        self.role_alias = self.inputs.iot_role_alias.get()
        self.kvs_region = self.inputs.kvs_region.get()
        self.root_ca = '/panorama/certs/cacert.pem'
        self.kvs_pipeline = {}
        self.app_src = {}

        logger.info(' ---- Parameters---- ')
        logger.info(self.kvs_stream_name)
        logger.info(self.role_alias)
        logger.info(self.kvs_region)
        logger.info(' ++++ Parameters++++ ')

        try:
            iot_client = boto3.client('iot',region_name=self.kvs_region)
            self.endpoint = iot_client.describe_endpoint(endpointType='iot:CredentialProvider')['endpointAddress']
        except:
            logger.exception('Error on getting IoT credential endpoint.')

        self.cameras = list(self.kvs_stream_name.split(","))

        for camera in self.cameras:
            # Parse Gstreamer appsrc pipeline and hook with camera name as appsrc name
            iot_cert = f'/panorama/certs/{camera}-cert.pem'
            iot_key = f'/panorama/certs/{camera}-private.key'

            pipe = f"appsrc name={camera} is-live=true block=true format=GST_FORMAT_TIME do-timestamp=TRUE " \
                f" caps=video/x-raw,format=BGR,width=1280,height=720,framerate=15/1 " \
                f"! videoconvert ! x264enc ! video/x-h264,stream-format=(string)byte-stream ! h264parse " \
                f"! kvssink stream-name={camera} storage-size=512 " \
                f" iot-certificate=\"iot-certificate,endpoint={self.endpoint}," \
                f"cert-path={iot_cert}," \
                f"key-path={iot_key}," \
                f"ca-path={self.root_ca}," \
                f"role-aliases={self.role_alias}\" " \
                f"aws-region={self.kvs_region}"

            self.kvs_pipeline[camera] = Gst.parse_launch(pipe)
            self.app_src[camera] = self.kvs_pipeline[camera].get_by_name(camera)

            self.kvs_pipeline[camera].set_state(Gst.State.READY)
            self.kvs_pipeline[camera].set_state(Gst.State.PLAYING)

    def push_to_pipeline(self, src, frame):
        data = frame.tobytes()
        buf = Gst.Buffer.new_wrapped(data)
        buf.duration = Gst.CLOCK_TIME_NONE
        timestamp = Gst.CLOCK_TIME_NONE
        buf.pts = buf.dts = Gst.CLOCK_TIME_NONE
        buf.offset = Gst.BUFFER_OFFSET_NONE
        src.emit('push-buffer', buf)

    def process_streams(self):
        """Processes one frame of video from one or more video streams."""
        while True:
        # Loop through attached video streams
            streams = self.inputs.video_in.get()
            for stream in streams:
                
                # media.stream_id can contain UUID at the end. Remove it.
                stream_name = stream.stream_id
                re_result = re.fullmatch( "(.*)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", stream_name )
                if re_result is not None:
                    stream_name = re_result.group(1)
                
                self.push_to_pipeline(self.app_src[stream_name], stream.image)

            self.outputs.video_out.put(streams)

def get_logger(name=__name__,level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=100000000, backupCount=2)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    try:
        logger.info("INITIALIZING APPLICATION")
        app = Application()
        logger.info("PROCESSING STREAMS")
        app.process_streams()
    except:
        logger.exception('Exception during processing loop.')

logger = get_logger(level=logging.INFO)
main()