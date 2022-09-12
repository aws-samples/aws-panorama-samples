import os
import sys

import cv2
import numpy as np
import logging
import panoramasdk as p
import time
import sys
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
import copy
import time

from metrics import MetricsFactory
from cw_post_metric import MetricsHandler
from log_utils import get_logger

log = get_logger()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
input_saved_model_dir="/panorama/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
output_saved_model_dir = "/panorama/saved_model_trt_fp16"
mocked_frame_resolution = '1080'

class MockedFrame:
    __slots__ = ["image", "resolution"]  # <-- allowed attributes

    def __init__(self, resolution):
        self.resolution = resolution
        self.image = self.random_arr()

    def random_arr(self):
        if '720' in self.resolution:
            return np.array(np.random.rand(720, 1280, 3) * 255, dtype='uint8')
        elif '1080' in self.resolution:
            return np.array(np.random.rand(1080, 1920, 3) * 255, dtype='uint8')
        else:
            raise RuntimeError('Unsupported resolution for mocked frames')

class ObjectDetectionApp(p.node):

    def __init__(self):
        self.model_batch_size = self.inputs.batch_size.get()
        self.pre_processing_output_size = 300

        root = tf.saved_model.load(output_saved_model_dir)
        self.model_infer = root.signatures['serving_default']

        # Note: These are CW dimensions. Change as necessary
        dimensions = list()
        stage_dimension = {'Name': 'Stage', 'Value': 'Gamma'}
        region_dimension = {'Name': 'Region', 'Value': 'us-west-2'}
        model_name_dimension = {'Name': 'ModelName', 'Value': 'SSD'}
        batch_size_dimention = {'Name': 'BatchSize', 'Value': str(self.model_batch_size)}
        app_function_dimension = {'Name': 'AppName', 'Value': 'OpenGPUAccessDemoApp'}
        dimensions.append(stage_dimension)
        dimensions.append(region_dimension)
        dimensions.append(app_function_dimension)
        dimensions.append(model_name_dimension)
        dimensions.append(batch_size_dimention)
        metrics_factory = MetricsFactory(dimensions)
        self.metrics_handler = MetricsHandler("AppMetrics", metrics_factory)


    def metric_latency_decorator(**decorator_params):
        def wrapper(method_ref):
            def inner(self, *method_param):
                start = time.time()
                m = self.metrics_handler.get_metric(decorator_params['metric_name'])
                out = method_ref(self, *method_param)
                m.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(m)
                end = time.time()
                return out

            return inner

        return wrapper

    def get_mocked_frames(self):
        """ Get a batch of random frames """
        input_frames = list()
        for i in range(self.model_batch_size):
            input_frames.append(MockedFrame(mocked_frame_resolution))
        return input_frames

    @metric_latency_decorator(metric_name='PreProcessBatchTime')
    def pre_process(self, image):
        """Resizes and normalizes a frame of video."""
        size = self.pre_processing_output_size
        resized = cv2.resize(image, (size, size))
        x1 = np.asarray(resized)
        x1 = np.expand_dims(x1, 0)
        return x1

    @metric_latency_decorator(metric_name='InferenceBatchTime')
    def infer(self, pre_processed_images_arr):
        input_tensor = tf.convert_to_tensor(pre_processed_images_arr)
        input_tensor_float = tf.cast(input_tensor, dtype=tf.float32)
        probs = self.model_infer(input_tensor_float)
        return probs

    @metric_latency_decorator(metric_name='InputFrameGetTime')
    def get_frames(self):
        input_frames = self.inputs.video_in.get()
        return input_frames
    
    def run(self):
        input_images = list()
        # Media event loop
        while True:
            input_frames = self.get_frames()
            self.metrics_handler.put_metric_count('InputFrameCount', len(input_frames))
            # Uncomment below line to get random mock frames instead od actual frames.
            #input_frames = self.get_mocked_frames()
            input_images.extend([frame.image for frame in input_frames])
            if len(input_images) >= self.model_batch_size:
                total_process_metric = self.metrics_handler.get_metric('TotalProcessBatchTime')
                input_images_batch = input_images[:self.model_batch_size]
                pre_processed_images = [self.pre_process(image) for image in input_images_batch]
                pre_processed_images_arr = np.vstack(pre_processed_images)
                prob = self.infer(pre_processed_images_arr)
                total_process_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_process_metric)
                ## TODO: Add code for any post-processing required
                input_images = input_images[self.model_batch_size:]
        
        

if __name__ == '__main__':
    try:
        app = ObjectDetectionApp()
        app.run()
    except Exception as err:
        log.exception("App did not start {}".format(err))
        sys.exit(1)
