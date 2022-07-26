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

categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

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
        self.onnx = ort.InferenceSession('/panorama/onnx_model/yolov5s_fp16.onnx')
        self.io_binding = self.onnx.io_binding()
        self.input_name = self.onnx.get_inputs()[0].name
        self.output_name = self.onnx.get_outputs()[0].name
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
    def preprocess_onnx_batch(self):
        self.preprocessed_images = np.vstack([utils.preprocess(image) for image in self.org_image_list])
        X_ortvalue = ort.OrtValue.ortvalue_from_numpy(self.preprocessed_images, 'cuda', 0)
        self.io_binding.bind_input(self.input_name, device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
        self.io_binding.bind_output(self.output_name)

    @metric_latency_decorator(metric_name='TotalInferenceTime')
    def infer(self):
        # pred = self.onnx.run(None, {self.input_name: pre_processed_images})
        self.onnx.run_with_iobinding(self.io_binding)

    @metric_latency_decorator(metric_name='InputFrameGetTime')
    def get_frames(self):
        input_frames = self.inputs.video_in.get()
        return input_frames

    @metric_latency_decorator(metric_name='PostProcessBatchTime')
    def postprocess(self, filtered_classes = None, conf_thres=0.5, iou_thres=0.45):
        pred = self.io_binding.copy_outputs_to_cpu()[0]
        pred = torch.from_numpy(pred)
        pred = utils.non_max_suppression(pred, conf_thres = conf_thres, 
            iou_thres=iou_thres, classes=filtered_classes)
        
        output = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = utils.scale_coords(self.preprocessed_images[0].shape[1:], 
                    det[:, :4], self.org_image_list[0].shape).round()
                output.append(det.cpu().detach().numpy())
            else:
                output.append(np.array([]))
        return output

    def run(self):
        input_images = list()
        while True:
            input_frames = self.get_frames()
            self.metrics_handler.put_metric_count('InputFrameCount', len(input_frames))
            input_images.extend([frame.image for frame in input_frames])
            if len(input_images) >= self.model_batch_size:
                total_process_metric = self.metrics_handler.get_metric('TotalProcessBatchTime')
                self.org_image_list = input_images[:self.model_batch_size]
                
                # Image preprocessing, sequentially
                self.preprocess_onnx_batch()
                
                # Inference
                self.infer()
                
                total_process_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_process_metric)
                
                # Post Process
                # you can filter the prediction by a list of class_idx before nms. 
                # we recommend to do this in the begining of this file. Don't put categories.index inside while loop.
                # ex: filtered_classes = [ categories.index("person") ]
                # ex: prediction = self.postprocess(filtered_classes) 
                prediction = self.postprocess()

                # uncomment the below section to draw the bounding box, drawing takes time and slow.
                visualize_metric = self.metrics_handler.get_metric('VisualizeBatchTime')
                for image_idx, det_results in enumerate(prediction):
                    for box_idx, bbox in enumerate(det_results):
                        bbox = bbox.tolist()
                        coord = bbox[:4]
                        score = bbox[4]
                        class_id = bbox[5]
                        utils.plot_one_box(coord, self.org_image_list[image_idx],
                            label="{}:{:.2f}".format(categories[int(class_id)], score))
                visualize_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(visualize_metric)
                
                input_images = list()
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
        log.exception(traceback.format_exc())
        sys.exit(1)

