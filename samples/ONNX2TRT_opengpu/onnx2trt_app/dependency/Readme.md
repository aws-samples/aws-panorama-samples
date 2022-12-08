# Export Yolov5 to ONNX for TRT7

> **IMPORTANT**: The ONNX model is already included in the src folder. We did not export the yolov5 model to ONNX for you in this sample; thus you need to export it yourself if you trained a custom model.

> **SETTINGS**: We are using yolov5r6.1, Tested under this [commit](https://github.com/ultralytics/yolov5/tree/fd004f56485d44c9c65b37c47d0e5f6165e1d944). Strongly recommend to use the same settings.

- To training on custom dataset, please refer to the [ultralytics yolov5](https://github.com/ultralytics/yolov5/blob/master/export.py)
- After training, please run the following code to set up the environment to export the model. 

```
# Set up a python3.7 virtualenv
# ex: conda create --name yolov5_export  python=3.7 
# conda activate yolov5_export
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/cu102/torch_stable.html
git clone https://github.com/ultralytics/yolov5.git
cd yolov5 && git checkout fd004f56485d44c9c65b37c47d0e5f6165e1d944
cp ../export_onnx.py .
```
export your yolov5 model to onnx before we start building and deploying our app to Panorama. 
- If one want to export the model with image size 320, please set `--imgsz 320`
- ex:

```
python export_onnx.py --weights yolov5s.pt --target_trt_version <your target tensorrt version> --dynamic --imgsz 640
```

Currently it will export yolov5.pt r6.1 to ONNX. 

*Note:*
- The export_onnx.py is tested in G4dn with Cuda 10.02 installed.
- Tested under this [commit](https://github.com/ultralytics/yolov5/tree/fd004f56485d44c9c65b37c47d0e5f6165e1d944)
- Exporting onnx alone does not depends on Cuda and TensorRT. As long as one can install the requirements.txt, one should be able to finish the export_onnx.py with no issue.


 