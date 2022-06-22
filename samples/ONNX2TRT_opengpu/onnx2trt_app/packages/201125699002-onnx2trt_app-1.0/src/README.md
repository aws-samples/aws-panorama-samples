# yolov5

The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

Based on this commit: 47233e1698b89fc437a4fb9463c815e9171be955

1. Convert yolov5 model to onnx file, then we will deploy the onnx file along with the app.
2. The first time the app is deployed, we will compile a tensorrt file from onnx, and start the app.


To debug your code locally, we recommend (Currently Panorama Environment)

1. CUDA 10.02, Tensorrt 7.1.3.4

To use  tensorrt

```
pip install nvidia-tensorrt==7.2.3.4 --index-url https://pypi.ngc.nvidia.com
```

# TODO

Tuesday:
- Add an export file for Matthew
- Merge with Tec and upload to git

Wed:
- Make it an App
- Support Batching

Thursday:
- Support Batching