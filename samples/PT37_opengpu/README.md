# Run YoloV5s Model on AWS Panorama using Torch + TorchVision

## Brief
In this guide, we show how to use a Yolov5s model with PyTorch GPU on the Panorama device. 

## Model

We have yolov5_half.pt (FP16) model in

```
aws-panorama-samples/samples/PT37_opengpu/yolov5s_37_app/packages/028663699634-yolov5s_37_app-1.0/src/yolov5s_model/
```

## Download Pre-Built model (Optional)

You can download a pre-built model from the yolov5s ultralytics github repository here : [Link](https://github.com/ultralytics/yolov5).
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
python3 export.py --weights yolov5s.pt
```
If you have a custom model, please use that instead of the prebuilt yolov5s.pt

## Train Custom Model

Training Custom Model : [Refer to this link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)


## Setup the application

The dependencies folder included with this application has 

* The Dockerfile
* Cuda Enabled PyTorch for Jetson Xavier

## Steps for setting this up

* Step 1: Navigate to ./dependencies/docker or ./dependencies/docker_jp462
    * ./dependencies/docker is for Panorama system software v4.3.x and this corresponds to Jetpack 4.4
    * ./dependencies/docker_jp462 is for Panorama system software v5.0+ and this corresponds to Jetpack 4.6.
* Step 2 : ``` sudo docker build -t pt:37 . ```
* Step 3 : Open pytorch_example.ipynb and make sure you configure the following
    * The Device ID
    * The Camera node information
* Step 4 : Follow the steps outlined in the notebook

## Special flags in package.json

* Step 1 : Before you deploy the application, open PT37_opengpu/yolov5s_37_app/packages/(account-id)-yolov5s_37_app-1.0/package.json
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
        "name": "yolov5s_pt37_app_asset",
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