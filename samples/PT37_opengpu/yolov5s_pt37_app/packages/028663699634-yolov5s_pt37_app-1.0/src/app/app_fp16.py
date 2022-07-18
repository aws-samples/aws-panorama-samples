import os
os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"
import site
site.addsitedir('/usr/lib/python3.7/site-packages/')
site.addsitedir('/usr/local/lib/python3.7/site-packages/')

import panoramasdk as p

import torch
import sys
import img_utils

from metrics import MetricsFactory
from cw_post_metric import MetricsHandler

import logging
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=10000000, backupCount=2)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)

log.info('Logging and Imports Done')

class ObjectDetectionApp(p.node):

    def __init__(self):
        self.model_batch_size = int(self.inputs.batch_size.get()) or 1
        self.pre_processing_output_size = 640
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolov5s = torch.jit.load('/panorama/yolov5s_model/yolov5s_half.pt', map_location=torch.device(self.device))
        self.line_thickness = 3
        
        # Note: These are CW dimensions. Change as necessary
        dimensions = list()
        stage_dimension = {'Name': 'Stage', 'Value': 'Gamma'}
        region_dimension = {'Name': 'Region', 'Value': 'us-west-2'}
        model_name_dimension = {'Name': 'ModelName', 'Value': 'YoloV5s'}
        batch_size_dimention = {'Name': 'BatchSize', 'Value': str(self.model_batch_size)}
        app_function_dimension = {'Name': 'AppName', 'Value': 'YoloPTDemo'}
        dimensions.append(stage_dimension)
        dimensions.append(region_dimension)
        dimensions.append(app_function_dimension)
        dimensions.append(model_name_dimension)
        dimensions.append(batch_size_dimention)
        metrics_factory = MetricsFactory(dimensions)
        self.metrics_handler = MetricsHandler("YoloAppMetrics", metrics_factory)

    def metric_latency_decorator(**decorator_params):
        def wrapper(method_ref):
            def inner(self, *method_param):
                m = self.metrics_handler.get_metric(decorator_params['metric_name'])
                out = method_ref(self, *method_param)
                m.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(m)
                return out

            return inner

        return wrapper

    @metric_latency_decorator(metric_name='InputFrameGetTime')
    def get_frames(self):
        input_frames = self.inputs.video_in.get()
        return input_frames

    def run(self):
        log.info("Pytorch Yolov5s FP16 App starts")
        input_images = list()

        while True:
            input_frames = self.get_frames()           
            self.metrics_handler.put_metric_count('InputFrameCount', len(input_frames))
            input_images.extend([frame.image for frame in input_frames])
            if len(input_images) >= self.model_batch_size:
                total_process_metric = self.metrics_handler.get_metric('TotalProcessBatchTime')
                input_images_batch = input_images[:self.model_batch_size]
                
                # Create Torch Arrays from Preprocessed Images
                preprocessing_metric = self.metrics_handler.get_metric('PreProcessBatchTime')
                pre_processed_images = [torch.from_numpy(img_utils.preprocess_v1(image)).to(self.device).half() for image in input_images_batch]
                preprocessing_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(preprocessing_metric)

                # Create Torch Stack
                pre_processed_images_arr = torch.stack(pre_processed_images)
                
                # the latest yolov5s preprocessing (preprocess_v2) adding one more dimension
                # e.g. with batch size 4, have [4, 1, 3, 640, 640] squeezed into [4, 3, 640, 640]
                # pre_processed_images_arr = torch.squeeze(pre_processed_images_arr, dim=1)
                
                # Inference
                total_inference_metric = self.metrics_handler.get_metric('TotalInferenceTime')
                pred = self.yolov5s(pre_processed_images_arr)[0]
                total_inference_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_inference_metric)  
                
                # Reset Input Images
                input_images = input_images[self.model_batch_size:]
                
                total_process_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_process_metric)
            
            self.outputs.video_out.put(input_frames)
            
            app_inference_state = self.metrics_handler.get_metric('ApplicationStatus')
            app_inference_state.add_value(float("1"), "None", 1)
            self.metrics_handler.put_metric(app_inference_state)
            

if __name__ == '__main__':
    try:
        app = ObjectDetectionApp()
        app.run()
    except Exception as err:
        log.exception('App Did not Start {}'.format(err))
        sys.exit(1)
