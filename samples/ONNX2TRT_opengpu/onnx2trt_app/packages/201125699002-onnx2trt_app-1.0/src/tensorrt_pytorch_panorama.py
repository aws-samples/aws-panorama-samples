import os
os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"

import boto3
s3 = boto3.resource('s3')

import panoramasdk as p
import os
import sys
from yolov5trt import YoLov5TRT

import logging
from logging.handlers import RotatingFileHandler
import utils
import tensorrt as trt

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


class ObjectDetectionApp(p.node):

    def __init__(self):
        self.model_batch_size = self.inputs.batch_size.get()
        self.pre_processing_output_size = 640
        self.onnx_file_path = None
        if trt.__version__[0] == '7':
            self.onnx_file_path = "/panorama/yolov5s.onnx"
        elif trt.__version__[0] == '8':
            self.onnx_file_path = "/panorama/yolov5s_opset13.onnx"
        else:
            raise ValueError("Currently only support TRT 7 and 8 but trt version {} found.".format(trt.__version__[0]))

        self.engine_file_path = "/panorama/yolov5s_dynamic_148.engine" # this is a TRT7 prebuilt engine.
        # If you want to use your own onnx model and convert it to engine file, please 
        # uncommnet the line below. This will build your engine and save it to the following path.
        # self.engine_file_path = "/opt/aws/panorama/storage/yolov5s_dynamic_148.engine"
        
        self.fp = 16
        self.engine_batch_size = "1 4 8"
        self.is_dynamic = True
        # The reason we use os system here instead of using function call is to save memory.
        # Building engines will runtime load more tensorrt library, and cuase more memory usage.
        # And the loaded library will be released only after the app ends.
        # Thus here using a system call to trigger a standalone process can solve the problem. 
        # (The process will die after building engine file, and thus release loaded library)
        # Another possible way is using Python inbuilt Process. However, Process under Panorama cannot access GPU.
        if not os.path.exists(self.engine_file_path):
            os.system("python3 /panorama/onnx_tensorrt.py -i {} -o {} -p {} -b {}".format(
                self.onnx_file_path, self.engine_file_path, self.fp, self.engine_batch_size
            ))
        
        self.yolov5_wrapper = YoLov5TRT(self.engine_file_path, self.model_batch_size, self.is_dynamic)

        ########### CLOUD WATCH METRICS #################
        # Note: These are CW dimensions. Change as necessary
        # sending performance related metrics to cloudwatch metrics
        # Please deploy this app with IAM role that has CW permission if you want to send the 
        # metrics to CW metrics. Otherwise, please replace all cloudwatch metrics to simply time.time()
        from cw_metrics.metrics import MetricsFactory
        from cw_metrics.cw_post_metric import MetricsHandler
        dimensions = list()
        stage_dimension = {'Name': 'Stage', 'Value': 'Gamma'}
        region_dimension = {'Name': 'Region', 'Value': 'us-east-1'}
        model_name_dimension = {'Name': 'ModelName', 'Value': 'YoloV5s'}
        batch_size_dimention = {'Name': 'BatchSize', 'Value': str(self.model_batch_size)}
        app_function_dimension = {'Name': 'AppName', 'Value': 'TensorRTDemo'}
        dimensions.append(stage_dimension)
        dimensions.append(region_dimension)
        dimensions.append(app_function_dimension)
        dimensions.append(model_name_dimension)
        dimensions.append(batch_size_dimention)
        metrics_factory = MetricsFactory(dimensions)
        self.metrics_handler = MetricsHandler("TensorRTAppMetrics", metrics_factory)
        ########### CLOUD WATCH METRICS #################
    
    def get_frames(self):
        input_frames = self.inputs.video_in.get()
        return input_frames


    def run(self):
        input_images = list()
        image_list = [] # An image queue
        while True:
            try:
                input_frames = self.get_frames()
                input_images = [frame.image for frame in input_frames]
                image_list+=input_images
                if len(image_list) >= self.model_batch_size:
                    total_process_metric = self.metrics_handler.get_metric('TotalProcessBatchTime')
                    
                    org_image_list = image_list[:self.model_batch_size]
                    # preprocessing and memcp to device mem
                    preprocess_metric = self.metrics_handler.get_metric('PreProcessBatchTime')
                    self.yolov5_wrapper.preprocess_image_batch(org_image_list)
                    preprocess_metric.add_time_as_milliseconds(1)
                    
                    # inference and left the inferenced results in device memory
                    infernce_metric = self.metrics_handler.get_metric('TotalInferenceTime')
                    self.yolov5_wrapper.infer()
                    infernce_metric.add_time_as_milliseconds(1)

                    # exclude postprocessing time, since this is business logic heavy.
                    total_process_metric.add_time_as_milliseconds(1)
                    

                    # memcp from device to host memory, and nms + postprocessing.
                    postprocess_metric = self.metrics_handler.get_metric('PostProcessBatchTime')
                    # you can filter the prediction by class before nms. 
                    # ex: prediction = self.yolov5_wrapper.post_process_batch(filtered_classes=['person'])
                    prediction = self.yolov5_wrapper.post_process_batch() 
                    postprocess_metric.add_time_as_milliseconds(1)

                    visualize_metric = self.metrics_handler.get_metric('VisualizeBatchTime')
                    # Draw rectangles and labels on the original image
                    for image_idx, det_results in enumerate(prediction):
                        for box_idx, bbox in enumerate(det_results):
                            bbox = bbox.tolist()
                            coord = bbox[:4]
                            score = bbox[4]
                            class_id = bbox[5]
                            utils.plot_one_box(coord, org_image_list[image_idx],
                                label="{}:{:.2f}".format(categories[int(class_id)], score))
                    visualize_metric.add_time_as_milliseconds(1)

                    image_list = image_list[self.model_batch_size:]

                    self.metrics_handler.put_metric(total_process_metric)
                    self.metrics_handler.put_metric(preprocess_metric)
                    self.metrics_handler.put_metric(infernce_metric)
                    self.metrics_handler.put_metric(postprocess_metric)
                    self.metrics_handler.put_metric(visualize_metric)
                    
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
