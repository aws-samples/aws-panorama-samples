# Run EasyOCR library on AWS Panorama

## Brief
In this guide, we show how to use easyocr library on the Panorama device. 

## Setup the application

The dependencies folder included with this application has 

* The Dockerfile
* TensorRT 7.1
* Cuda Enabled PyTorch for Jetson Xavier

## Steps for setting this up

* Step 1: Navigate to ./dependencies
* Step 2 : ``` sudo docker build -t pt:37 . ```
* Step 3 : Open pytorch_example.ipynb and make sure you configure the following
    * The Device ID
    * The Camera node information
* Step 4 : Follow the steps outlined in the notebook

## Special flags in package.json

* Step 1 : Before you deploy the application, open PT37_opengpu_easyocr/easyocr_37_app/packages/(account-id)-easyocr_37_app-1.0/package.json
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
        "name": "easyocr_37_2_app",
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