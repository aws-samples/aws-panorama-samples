# Run YoloV5s that is converted to ONNX using the ONNX Runtime on AWS Panorama

## Brief

In this guide, we show how to get inference from a Yolov5s pytorch model that is converted to an onnx file using ONNX Runtime

## Why ONNX and ONNX Runtime?

ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

![Supported Frameworks and Converters](ONNX_Supported.png)

To learn more, go to [ONNX.ai](https://onnx.ai/)

## Model

* We have already included the yolov5s.onnx model in the depdendencies folder and also packages/<app_name>-1.0/src/onnx_model folder.
* If you would like to convert any other model from any other framework to ONNX, please see the ONNX.ai website 

```
packages/<app_name>/src/onnx_model/yolov5s.onnx
```

## Train Custom Model

* Training Custom Model : [Refer to this link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* Convert the custom model to .onnx file : [Refer to this link](https://docs.ultralytics.com/tutorials/torchscript-onnx-coreml-export)


## Setup the application

The dependencies folder included with this application has 

* The Dockerfile
* Cuda Enabled PyTorch for Jetson Xavier
* ONNX Runtime GPU 1.6
* yolov5s.onnx

**VERY IMPORTANT** : This example uses a batch of 8, this means we have to use ATLEAST 2 CAMERAS for this example

## Steps for setting this up

* Step 1: Navigate to ./dependencies
* Step 2 : ``` sudo docker build -t onnx37:latest . ```
* Step 3 : Open onnx_example.ipynb and make sure you configure the following
    * The Device ID
    * The Camera node information
* Step 4 : Follow the steps outlined in the notebook

## Special flags in package.json

* Step 1 : Before you deploy the application, open ONNX37_opengpu/onnx_37_app/packages/(account-id)-onnx_37_app-1.0/package.json
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
        "name": "onnx_37_app",
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
    

# Debugging

If you encounter issues with deploying from this, once the application is uploaded to the cloud, you can use the graph.json and deploy using the Panorama console as well