# Run YoloV5s implemented with tensorrt network definition APIs

## Brief

In this guide, we show how to use a Yolov5s model using TensorRT network definition APIs to build the whole network. 

This is an implementation of the YoloV5s model in this repository : [Link](https://github.com/wang-xinyu/tensorrtx). 

## Explain 3.6 usage here


## Model

* We have already included the yolov5s.engine model in the packages/src/tensorrtx/yolov5/build folder.
* But if you would like to convert your own .pt file to a .engine file, please follow the instructions in this [link](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5) 
* We used yolov5 v3.0 from this repository
* The .engine file is in the following folder

```
packages/<app_name>/src/tensorrtx/yolov5/build
```

## Train Custom Model

* Training Custom Model : [Refer to this link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* Convert the custom model to .engine file : [Refer to this link](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)


## Setup the application

The dependencies folder included with this application has 

* The Dockerfile
* TensorRT 7.1
* Cuda Enabled PyTorch for Jetson Xavier
* The yolov5s.pt model

## Steps for setting this up

* Step 1: Navigate to ./dependencies
* Step 2 : ``` sudo docker build -t trtpt36:latest . ```
* Step 3 : Open trtpt_example.ipynb and make sure you configure the following
    * The Device ID
    * The Camera node information
* Step 4 : Follow the steps outlined in the notebook

## Special flags in package.json

* Step 1 : Before you deploy the application, open TRTPT36_opengpu/trtpt_36_2_app/packages/(account-id)-trtpt_36_2_app-1.0/package.json
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
        "name": "trtpt_36_2_app",
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