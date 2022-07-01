# yolov5

The Pytorch implementation is based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

## Brief Code Structure

- tensorrt_pytorch_panorama.py
    - This is the main panorama app. 
    - It will call onnx_tensorrt.py to convert onnx model to engine file if engine file does not exist.
    - After building the engine, it will start the inferencing.
    - It usually takes 12~20 mins to build the engine. Just one time build.
- onnx_tensorrt.py
    - The code that will convert onnx to engine file.
- yolov5trt.py
    - It is a yolov5 inference code wrapper. It includes preprocessing, inference and postprocessing. And in this app we only plot bboxes for human. You can easily change the logic in the `post_process` function