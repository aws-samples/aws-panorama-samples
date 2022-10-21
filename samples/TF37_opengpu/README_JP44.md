# Runing TF-TRT on Panorama - JetPack 4.4

If your panorama device is still using Panorama system software v4.3.x (JetPack4.4), this is the right guide you are looking for. 

If your panorama device is using Panorama system software v5.0+ (Jetpack4.6.2), please refer to README.md

## Brief
In this guide, we show how to convert a Tensorflow SSD model to a TF-TRT model. This guide is derived from and the full version is available at : https://apivovarov.medium.com/run-tensorflow-2-object-detection-models-with-tensorrt-on-jetson-xavier-using-tf-c-api-e34548818ac6
To improve Object Detection model performance the model will be exported to inference model with combined NMS in the post-processing option. 

## Setup

TF-TRT converter works correctly on a computer with NVIDIA GPU installed. We can use AWS g4dn.xlarge instance ($0.526/hr) to prepare TF-TRT model
Launch AWS EC2 GPU instance with the following parameters:
```
    AMI: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type, x86
    Type: g4dn.xlarge
    Root File system: 50 GiB
```
ssh to the instance

Update existing packages

```
sudo apt update
sudo apt upgrade
sudo reboot
```

We are going to install TensorRT-7.1.3 which is the same version as on Jetson Xavier JetPack 4.4.1. To export the model to TF-TRT we will use tensorflow-2.4.4 from pypi which needs cuda-11.0

### Install Cuda 11.0 from deb(local)

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
### Install pip
```
sudo apt install python3-pip
sudo pip3 install -U pip setuptools
```
### Optionally install awscli in case you need to copy to/from S3
```
sudo pip3 install -U awscli
aws configure
```

## Install TensorRT 7.1.3 for Cuda 11.0

Download TensorRT 7.1.3 for Cuda 11.0 deb(local) repo from NVIDIA developer website — file nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.1.3.4-ga-20200617_1-1_amd64.deb

```
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.1.3.4-ga-20200617_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-cuda11.0-trt7.1.3.4-ga-20200617/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrtsudo pip3 install protobuf
sudo apt-get install python3-libnvinfer-dev uff-converter-tf
```

## Install Tensorflow 2.4.4
```
Install Tensorflow 2.4.4
```

Verify that tensorflow can load Cuda libraries
```
#!python3
import tensorflow as tf
tf.test.is_gpu_available()
```

## Install Tensorflow Models object_detection project

```
sudo apt install protobuf-compiler
git clone https://github.com/tensorflow/models.git tensorflow_models
cd tensorflow_models/research# Compile protos.
protoc object_detection/protos/*.proto --python_out=.# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .# Edit setup.py and modufy REQUIRED_PACKAGES list
# - add tfensorflow==2.4.4
# - change tf-models-official>=2.4.0python3 -m pip install .
```

## Export SSD Mobilenet model to inference saved model
Download SSD Mobilenet model from TF2 model zoo

```
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
tar zxf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
```
To improve the model performance we are going to enable combined NMS in the post-processing

Edit ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config and add change_coordinate_frame: false and use_combined_nms: true to post_processing -> batch_non_max_suppression block.

It should look the following:
```
post_processing {
      batch_non_max_suppression {
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: false
        change_coordinate_frame: false
        use_combined_nms: true
      }
      score_converter: SIGMOID
    }
```
In order to get exported model with dynamic batch size we will use input_type float_image_tensor

Export inference graph
```
python3 object_detection/exporter_main_v2.py \
    --input_type=float_image_tensor \
    --pipeline_config_path=ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config \
    --trained_checkpoint_dir=ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint \
    --output_directory=output/ssd_mobilenet_v2_320x320_coco17_tpu-8_float_batchN_nms
```

Validate the exported model
```
cd output/ssd_mobilenet_v2_320x320_coco17_tpu-8_float_batchN_nms#!python3
import tensorflow as tfm = tf.saved_model.load("saved_model")
ff = m.signatures['serving_default']
x = tf.ones(shape=(8,300,300,3))
y = ff(x)import time
N = 1000
t1 = time.time()
for i in range(N):
  out = ff(x)
tt = time.time() - t1
print("exec time:", tt)
print(8*N/tt, "fps")
```

Convert the model to TF-TensorRT model
```
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
# FP16
conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP16)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="saved_model",
    conversion_params=conversion_params)
converter.convert()
converter.save("saved_model_trt_fp16")# FP32
conversion_params = trt.TrtConversionParams()
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="saved_model",
    conversion_params=conversion_params)
converter.convert()
converter.save("saved_model_trt_fp32")
```

Validate the exported TF-TRT models
```
#!python3
import tensorflow as tfm = tf.saved_model.load("saved_model_trt_fp32")
ff = m.signatures['serving_default']
x = tf.ones(shape=(8,300,300,3))
y = ff(x)# It should print the following indicating that TensorRT infer libraries are loaded
# Linked TensorRT version: 7.1.3
# Successfully opened dynamic library libnvinfer.so.7
# Loaded TensorRT version: 7.1.3
# Successfully opened dynamic library libnvinfer_plugin.so.7import time
N = 1000
t1 = time.time()
for i in range(N):
  out = ff(x)
tt = time.time() - t1
print("exec time:", tt)
print(8*N/tt, "fps")
```

## Special flags in package.json

* Step 1 : Before you deploy the application, open /packages/(account-id)-<APP NAME>_app-1.0/package.json
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
        "name": "<APP_NAME>",
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