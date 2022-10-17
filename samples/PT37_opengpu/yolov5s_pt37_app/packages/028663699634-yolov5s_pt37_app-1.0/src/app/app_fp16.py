import os
os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"
import site
site.addsitedir('/usr/lib/python3.7/site-packages/')
site.addsitedir('/usr/local/lib/python3.7/site-packages/')

import panoramasdk as p
import numpy as np
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

categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

class ObjectDetectionApp(p.node):

    def __init__(self):
        self.model_batch_size = int(self.inputs.model_batch_size.get()) or 1
        self.pre_processing_output_size = 640
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolov5s = torch.jit.load('/panorama/yolov5s_model/yolov5s_half.pt', map_location=torch.device(self.device))
        self.num_classes = 80

        # NMS: set the threshold and filtered class
        self.conf_thres = 0.5
        self.iou_thres = 0.45

        # classes you want to detect. None as disable filter.
        # this will filter out other classes before nms.
        # ex: filtered_classes = categories.index("person")
        self.filtered_classes = None

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
        image_list = [] # An image queue

        while True:
            input_frames = self.get_frames()
            self.metrics_handler.put_metric_count('InputFrameCount', len(input_frames))
            image_list += [frame.image for frame in input_frames]

            if len(image_list) >= self.model_batch_size:
                total_process_metric = self.metrics_handler.get_metric('TotalProcessBatchTime')
                input_images_batch = image_list[:self.model_batch_size]

                preprocessing_metric = self.metrics_handler.get_metric('PreProcessBatchTime')
                pre_processed_images = [torch.from_numpy(img_utils.preprocess_v2(image)).to(self.device).half() for image in input_images_batch]

                # Create Torch Stack
                pre_processed_images = torch.stack(pre_processed_images) # 4 (batch size) * [1 * 100 * 100] -> [4, 1, 100, 100]

                # the latest yolov5s preprocessing (preprocess_v2) adding one more dimension
                # e.g. with batch size 4, have [4, 1, 3, 640, 640] squeezed into [4, 3, 640, 640]
                pre_processed_images = torch.squeeze(pre_processed_images, dim=1)

                # PyTorch CUDA is asynchronous: call synchronize() to wait for the prior GPU operation's completion.
                # This can be disabled if no need to do time measurement.
                torch.cuda.synchronize()
                preprocessing_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(preprocessing_metric)

                # Inference
                total_inference_metric = self.metrics_handler.get_metric('TotalInferenceTime')
                pred = self.yolov5s(pre_processed_images)[3] # 1, 25200, 85

                # PyTorch CUDA is asynchronous: call synchronize() to wait for the prior GPU operation's completion.
                # This can be disabled if no need to do time measurement.
                torch.cuda.synchronize()
                total_inference_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_inference_metric)

                # total process time includes only preprocess and inference
                total_process_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(total_process_metric)

                # Post Process
                postprocess_metric = self.metrics_handler.get_metric('PostProcessBatchTime')

                pred = img_utils.non_max_suppression(pred, conf_thres = self.conf_thres,
                       iou_thres=self.iou_thres, classes=self.filtered_classes)

                scaled_pred = []
                for det in pred:
                    if det is not None and len(det):
                        det[:, :4] = img_utils.scale_coords(pre_processed_images[0].shape[1:],
                                     det[:, :4], input_images_batch[0].shape).round()
                        scaled_pred.append(det.cpu().detach().numpy())
                    else:
                        scaled_pred.append(np.array([]))

                # PyTorch CUDA is asynchronous: call synchronize() to wait for the prior GPU operation's completion.
                # This can be disabled if no need to do time measurement.
                torch.cuda.synchronize()
                postprocess_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(postprocess_metric)

                visualize_metric = self.metrics_handler.get_metric('VisualizeBatchTime')
                # Draw rectangles and labels on the original image
                for image_idx, det_results in enumerate(scaled_pred):
                    for box_idx, bbox in enumerate(det_results):
                        bbox = bbox.tolist()
                        coord = bbox[:4]
                        score = bbox[4]
                        class_id = bbox[5]
                        img_utils.plot_one_box(coord, input_images_batch[image_idx],
                            label="{}:{:.2f}".format(categories[int(class_id)], score))
                visualize_metric.add_time_as_milliseconds(1)
                self.metrics_handler.put_metric(visualize_metric)

                # Reset Input Images
                image_list = image_list[self.model_batch_size:]

            self.outputs.video_out.put(input_frames)

            app_inference_state = self.metrics_handler.get_metric('ApplicationStatus')
            app_inference_state.add_value(float("1"), "None", 1)
            self.metrics_handler.put_metric(app_inference_state)


if __name__ == '__main__':
    try:
        app = ObjectDetectionApp()
        app.run()
    except Exception as err:
        log.exception('App did not Start {}'.format(err))
        app.metrics_handler.kill()
        sys.exit(1)
