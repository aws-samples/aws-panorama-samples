# Export Yolov5 to ONNX for TRT7

> **IMPORTANT**: The ONNX model is already included in the src folder. We did not export the yolov5 model to ONNX for you in this sample; thus you need to export it yourself if you trained a custom model.

- To training on custom dataset, please refer to the [ultralytics yolov5](https://github.com/ultralytics/yolov5/blob/master/export.py)
- After training, please run the following code to export your yolov5 model to onnx before we start building and deploying our app to Panorama. 
    - The export_onnx.py is part of the export.py script in the ultralytics yolov5 repo. Thus the python environment setup is exactly the same as the yolov5 repo. Please place this export_onnx.py under the ultralytics yolov5 repo. And execute:

```
python export_onnx.py --weights yolov5s.pt --target_trt_version <your target tensorrt version> --dynamic
```

Currently it will export yolov5.pt r6.1 to ONNX.

*Note:*
- The export_onnx.py is tested in G4dn with Cuda 10.02 installed.
- Exporting onnx alone does not depends on Cuda and TensorRT. As long as one can install the requirements.txt provided in ultralytics/yolov5 repo, one should be able to finish the export_onnx.py with no issue.