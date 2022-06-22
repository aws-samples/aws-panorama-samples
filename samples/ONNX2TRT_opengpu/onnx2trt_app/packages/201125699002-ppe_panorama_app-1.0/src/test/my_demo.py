import os
os.environ["PYTHON_EGG_CACHE"] = "/panorama/.cache"

import os
import sys

import cv2

import logging
from logging.handlers import RotatingFileHandler
log = logging.getLogger('my_logger')
log.setLevel(logging.DEBUG)

import sys
sys.path.insert(0, "..") 
import onnx_tensorrt
from yolov5trt import YoLov5TRT

if __name__ == '__main__':
    onnx_file_path = "../yolov5s.onnx"
    # engine_file_path = "../yolov5s_dynamicbatch_148.engine"
    batch_size = 8
    engine_file_path = "../yolov5s_staticbatch_8.engine"
    dynamic = False
    print(engine_file_path)
    yolov5_wrapper = YoLov5TRT(engine_file_path, batch_size, dynamic)
    input_images_list = [os.path.join("samples", fn )for fn in os.listdir("samples")]*1000
    input_images_list = [cv2.imread(pth) for pth in input_images_list]
    input_images_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in input_images_list]
    i = 0
    while(len(input_images_list)>batch_size):
        # create a new thread to do inference
        yolov5_wrapper.infer(input_images_list[:batch_size])
        # for input_image in input_images_list[:batch_size]:
            # cv2.imwrite('res-{}.jpg'.format(i), cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
            # i+=1
            # print(i)
        input_images_list = input_images_list[batch_size:]
    yolov5_wrapper.destroy()