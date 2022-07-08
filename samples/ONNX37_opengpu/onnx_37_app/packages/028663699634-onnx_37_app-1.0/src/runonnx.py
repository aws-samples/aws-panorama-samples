import os
os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"
import site

site.addsitedir('/usr/local/lib/python3.7/site-packages')
site.addsitedir('/usr/lib/python3.7/site-packages/')

import cv2
import onnxruntime as ort
import numpy as np
import torch
import torchvision
from PIL import Image
import panoramasdk as p
import sys
from models.utils import *
import traceback
import time

from metrics import MetricsFactory
from cw_post_metric import MetricsHandler
from log_utils import get_logger

import logging
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)
handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=10000000, backupCount=2)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)

anchor_list = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
stride = [8, 16, 32]

IMAGE_SIZE = (640, 640)
CONF_TH = 0.3
NMS_TH = 0.45
CLASSES = 80

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
        self.onnx = ort.InferenceSession('/opt/aws/panorama/storage/src_onnx/onnx_model/batch_dynamic_fp16_yolov5s.onnx')
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


    def draw(self, img, boxinfo, dst, id):
        for *xyxy, conf, cls in boxinfo:
            label = '{}|{}'.format(int(cls), '%.2f' % conf)
            plot_one_box(xyxy, img, label=label, color=[0, 0, 255])
        cv2.imencode('.jpg', img)[1].tofile(dst)

    @metric_latency_decorator(metric_name='PreProcessBatchTime')
    def preprocess_onnx_batch(self, input_images_batch):
        return np.vstack([self.preprocess_onnx(image) for image in input_images_batch])

    @metric_latency_decorator(metric_name='TotalInferenceTime')
    def infer(self, pre_processed_images):
        return self.onnx.run(None, {self.input_name: pre_processed_images})

    def preprocess_onnx(self, img):
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32')
        img_size = [img.shape[1], img.shape[2]]
        img /= 255.0
        img = img.reshape(1, 3, img_size[0], img_size[1])
        return img

    @metric_latency_decorator(metric_name='InputFrameGetTime')
    def get_frames(self):
        input_frames = self.inputs.video_in.get()
        return input_frames

    @metric_latency_decorator(metric_name='PostProcessBatchTime')
    def postprocess_onnx(self, pred, preprocessed_image, orig_image):
        anchor = torch.tensor(anchor_list).float().view(3, -1, 2)
        area = IMAGE_SIZE[0]*IMAGE_SIZE[1]
        size = [int(area/stride[0]**2), int(area/stride[1]**2), int(area/stride[2]**2)]
        feature = [[int(j/stride[i]) for j in IMAGE_SIZE] for i in range(3)]

        y = []
        y.append(pred[:, :size[0]*3, :])
        y.append(pred[:, size[0]*3:size[0]*3+size[1]*3, :])
        y.append(pred[:, size[0]*3+size[1]*3:, :])
        grid = []
        for k, f in enumerate(feature):
            grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

        z = []
        for i in range(3):
            src = y[i]
            xy = src[..., 0:2] * 2. - 0.5
            wh = (src[..., 2:4] * 2) ** 2
            dst_xy = []
            dst_wh = []
            for j in range(3):
                dst_xy.append((xy[:, j*size[i]:(j+1)*size[i], :] + torch.tensor(grid[i])) * stride[i])
                dst_wh.append(wh[:, j*size[i]:(j+1)*size[i], :] * anchor[i][j])
            src[..., 0:2] = torch.from_numpy(np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1))
            src[..., 2:4] = torch.from_numpy(np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1))
            z.append(src.view(self.model_batch_size, -1, CLASSES+5)) #85
        pred = torch.cat(z, 1)
        pred = nms(pred, CONF_TH, NMS_TH)
        

        output = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(preprocessed_image.shape[1:], det[:, :4], orig_image.shape).round()
                output.append(det)
        if det == None:
            return np.array([])
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
                
                # Create Torch Arrays from Preprocessed Images
                pre_processed_images = self.preprocess_onnx_batch(input_images_batch)
                
                # Inference
                pred = self.infer(pre_processed_images)
                
                total_process_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_process_metric)
                
                # Post Process
                try:
                    pred1 = torch.tensor(pred[3])
                    output_pred = self.postprocess_onnx(pred1, pre_processed_images[0], input_images_batch[0])
                except Exception as e:
                    log.exception('Exception from Try is {}'.format(e))
                    pass
                
                input_images = list()
                
                # if you need to draw annotations, use this space to do so
            
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

