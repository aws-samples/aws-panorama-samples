# Run Yolov5 TensorRT in Panorama: YoloV5s.pt ➡️ ONNX  ➡️ TensorRT

## Brief

In this guide, we show how to runtime build a tensorrt engine file in panorama and import the engine file for faster inference. This will includes:
- Export yolov5.pt model to ONNX format **before deploying to Panroama** (Optional. Use it when you trained a custom yolov5 model).
- Deploy the app along with the ONNX model.
- Convert ONNX model to TensorRT engine **in runtime**.
- Finally, inference using the engine file.

## Why Convert to ONNX First?
ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

And thus TensorRT also provides an ONNX Parser to parse the model and build engine.

![TensorRT+ONNX](https://developer-blogs.nvidia.com/wp-content/uploads/2021/07/onnx-workflow.png)

To learn more, please go to [this Nvidia Blog](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)

The Pros:
- Independent to the ML framework one is using. As long as you can convert your Tensorflow/Pytorch/MxNet/Caffe (and so on) models to ONNX, you can leverage TensorRT ONNX Parser to build an engine file.
- The ONNX to TensorRT script (onnx_tensorrt.py under packages/<app_name>-1.0/src/ folder) is generic. Thus it can fit different models. One only need to focus on exporting the model to ONNX format.

The Cons:
- ONNX TensorRT Parser sometimes does not support the latest ops. But at least currently Yolov5 is fully supported.

## Why Runtime Build a TensorRT Engine File?
Currently Nvidia GPUs do not support cross GPU architecture building engine file.
- ex: Build an engine file on Nvidia T4 GPU and execute it on Jetson Xavier AGX.

And thus the most common way to build engine file is build it runtime on target device (i.e. Panorama)
- ex: In [tensorrt-tensorflow](https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html ), it also mentioned:
    - > As a rule of thumb, we recommend building at runtime
- ex: In onnxruntime-gpu, it also provides TensorRT as exectution provider. And the execution provider will build the engine first before inference.

## Model

* We have already included the yolov5s.onnx model in the packages/<app_name>-1.0/src/ folder.
* If you would like to convert any other model from any other framework to ONNX, please see the [ONNX.ai](https://onnx.ai/) website.

```
packages/<app_name>/src/onnx_model/yolov5s.onnx
```

## Train Custom Yolov5 Model

* Training Custom Yolov5 Model : [Refer to this link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* Convert the custom model to .onnx file before deployment: [Refer to this readme](./onnx2trt_app/dependency/Readme.md)


## Setup the application

This application requires a Docker base image. 
- Please open a terminal, go to ./dependencies/docker
- run: `sudo docker build -t  trtpt36:latest .`
- This step takes a long time (approximately 5hrs.)


**VERY IMPORTANT** : This example will build the engine file with dynamic batch size ranges from 1 to 8. If you decide to use batch size larger than 4, please use ATLEAST 2 CAMERAS for this example.

**VERY IMPORTANT** : Batch size 8 is suitable for Jetson Xavier AGX. And for devices using Jetson Xavier NX module, please select at most batch size 4 instead of 8. please refer to this [link](https://aws.amazon.com/tw/panorama/appliance/) for more information about your device.

## Steps for setting this up

* Step 1: Navigate to ./dependencies/docker and build the base docker image.
* Step 2 : Open ONNX2TRT_opengpu.ipynb and follow along.

## Special flags in package.json

* Step 1 : Before you deploy the application, open ONNX2TRT_opengpu/onnx2trt_app/packages/(account-id)-onnx2trt_app-1.0/package.json
* Step 2 : Add the following flags to the package.json

```
"requirements": 
            [{
                    "type" : "hardware_access",
                    "inferenceAccelerators": [ 
                        {
                            "deviceType": "nvhost_gpu",
                            "sharedResourcePolicy": {
                                "policy" : "allow_all"
                            }
                        }
                    ]
            }]
```

The assets should look something like this

```
"assets": [
    {
        "name": "onnx2trt_app",
        "implementations": [
            {
                "type": "container",
                "assetUri": "9a49a98784f4571adacc417f00942dac7ef2e34686eef21dca9fcb7f4b7ffd70.tar.gz",
                "descriptorUri": "4bab130ec48eea84e072d9fe813b947e9d9610b2924099036b0165026a91d306.json",
                "requirements": 
                [{
                    "type" : "hardware_access",
                    "inferenceAccelerators": [ 
                        {
                            "deviceType": "nvhost_gpu",
                            "sharedResourcePolicy": {
                                "policy" : "allow_all"
                            }
                        }
                    ]
                }]
            }
        ]
    }
],
```
    

## Debugging

If you encounter issues with deploying from this, once the application is uploaded to the cloud, you can use the graph.json and deploy using the Panorama console as well