import os
os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"

import boto3
s3 = boto3.resource('s3')

import panoramasdk as p
import os
import random
import sys
import threading
import cv2
import numpy as np
import onnx_tensorrt
from yolov5trt import YoLov5TRT

import logging
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=10000000, backupCount=2)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)

class ObjectDetectionApp(p.node):

    def __init__(self):
        self.model_batch_size = self.inputs.batch_size.get()
        self.pre_processing_output_size = 640
        self.onnx_file_path = "/panorama/yolov5s.onnx"
        self.engine_file_path = "/opt/aws/panorama/storage/yolov5s_dynamic_148.engine"
        if not os.path.exists(self.engine_file_path):
            onnx_tensorrt.onnx2tensorrt(self.onnx_file_path, self.engine_file_path, dynamic_batch=[1, 4, 8])
        
        self.yolov5_wrapper = YoLov5TRT(self.engine_file_path, self.model_batch_size, True)
    
    def get_frames(self):
        input_frames = self.inputs.video_in.get()
        return input_frames


    def run(self):
        input_images = list()
        # TODO: supprot batch_size > 1, with multiple
        image_list = [] # An image queue
        while True:
            try:
                input_frames = self.get_frames()
                input_images = [frame.image for frame in input_frames]
                image_list+=input_images
                if len(image_list) >= self.model_batch_size:
                    self.yolov5_wrapper.infer(image_list[:self.model_batch_size])
                    image_list = image_list[self.model_batch_size:]
                self.outputs.video_out.put(input_frames)
        
            except Exception as e:
                self.yolov5_wrapper.destroy()
                log.exception('Exception is {}'.format(e))
                pass


if __name__ == '__main__':
    try:
        app = ObjectDetectionApp()
        app.run()
    except Exception as err:
        log.exception('App Did not Start {}'.format(err))
        sys.exit(1)
