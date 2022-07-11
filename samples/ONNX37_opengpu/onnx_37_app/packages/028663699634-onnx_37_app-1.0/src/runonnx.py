import os
os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"
import site

site.addsitedir('/usr/local/lib/python3.7/site-packages')
site.addsitedir('/usr/lib/python3.7/site-packages/')

import cv2
import onnxruntime as ort
import numpy as np
import torch
from PIL import Image
import panoramasdk as p
import sys
import traceback
import time

from metrics import MetricsFactory
from cw_post_metric import MetricsHandler

import logging
from logging.handlers import RotatingFileHandler
import utils
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=10000000, backupCount=2)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)

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
        elif 'actual' in self.resolution:
            return cv2.imread('/panorama/zidane.jpg')
        else:
            raise RuntimeError('Unsupported resolution for mocked frames')



class ObjectDetectionApp(p.node):

    def __init__(self):
        self.model_batch_size = self.inputs.batch_size.get()
        self.pre_processing_output_size = 640
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.onnx = ort.InferenceSession('/panorama/onnx_model/yolov5s.onnx') # or use batch_dynamic_fp16_yolov5s.onnx
        self.input_name = self.onnx.get_inputs()[0].name
        log.info('Model Loaded')
        self.stride = 32
        
        # Note: These are CW dimensions. Change as necessary
        dimensions = list()
        stage_dimension = {'Name': 'Stage', 'Value': 'Gamma'}
        region_dimension = {'Name': 'Region', 'Value': 'us-east-1'}
        model_name_dimension = {'Name': 'ModelName', 'Value': 'YoloV5s'}
        batch_size_dimention = {'Name': 'BatchSize', 'Value': str(self.model_batch_size)}
        app_function_dimension = {'Name': 'AppName', 'Value': 'ONNXDemo'}
        dimensions.append(stage_dimension)
        dimensions.append(region_dimension)
        dimensions.append(app_function_dimension)
        dimensions.append(model_name_dimension)
        dimensions.append(batch_size_dimention)
        metrics_factory = MetricsFactory(dimensions)
        self.metrics_handler = MetricsHandler("ONNXAppMetrics2", metrics_factory)
        
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
        input_frames = list()
        for i in range(self.model_batch_size):
            input_frames.append(MockedFrame('actual'))
        return input_frames

    @metric_latency_decorator(metric_name='PreProcessBatchTime')
    def preprocess_onnx_batch(self, input_images_batch):
        return np.vstack([utils.preprocess(image) for image in input_images_batch])

    @metric_latency_decorator(metric_name='TotalInferenceTime')
    def infer(self, pre_processed_images):
        return self.onnx.run(None, {self.input_name: pre_processed_images})

    @metric_latency_decorator(metric_name='InputFrameGetTime')
    def get_frames(self):
        input_frames = self.inputs.video_in.get()
        return input_frames

    @metric_latency_decorator(metric_name='PostProcessBatchTime')
    def postprocess(self, pred, preprocessed_image, orig_image):
        pred = utils.non_max_suppression(pred)
        output = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = utils.scale_coords(preprocessed_image.shape[1:], det[:, :4], orig_image.shape).round()
                output.append(det)
        return output

    def run(self):
        input_images = list()
        while True:
            input_frames = self.get_frames()
            self.metrics_handler.put_metric_count('InputFrameCount', len(input_frames))
            input_images.extend([frame.image for frame in input_frames])
            if len(input_images) >= self.model_batch_size:
                total_process_metric = self.metrics_handler.get_metric('TotalProcessBatchTime')
                input_images_batch = input_images[:self.model_batch_size]
                
                # Image preprocessing, sequentially
                pre_processed_images = self.preprocess_onnx_batch(input_images_batch)
                
                # Inference
                pred = self.infer(pre_processed_images)
                
                total_process_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_process_metric)
                
                # Post Process
                try:
                    prediction = torch.from_numpy(pred[3])
                    prediction = self.postprocess(prediction, pre_processed_images[0], input_images_batch[0])

                    # uncomment the below section to draw the bounding box, drawing takes time and slow.
                    
                    # for image_idx, det_results in enumerate(prediction):
                    #     for box_idx, bbox in enumerate(det_results):
                    #         bbox = bbox.tolist()
                    #         coord = bbox[:4]
                    #         score = bbox[4]
                    #         class_id = bbox[5]
                    #         utils.plot_one_box(coord, input_images_batch[image_idx],
                    #             label="{}:{:.2f}".format(class_id, score))

                except Exception as e:
                    log.exception('Exception from Try is {}'.format(e))
                    pass
                
                input_images = list()
            
            app_inference_state = self.metrics_handler.get_metric('ApplicationStatus')
            app_inference_state.add_value(float("1"), "None", 1)
            self.metrics_handler.put_metric(app_inference_state)

if __name__ == '__main__':
    try:
        app = ObjectDetectionApp()
        app.run()
    except Exception as err:
        log.exception('App Did not Start {}'.format(err))
        log.exception(traceback.format_exc())
        sys.exit(1)

