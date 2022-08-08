# Runing TF-TRT on Panorama - JetPack 4.6.1


If your panorama device is still using device image v4.3.x (JetPack4.4), please refer to README.md

If your panorama device is using device image v4.4.x (Jetpack4.6.1), this is the right guide you are looking for.

In this readme, we will walk you through 
- Export your tensorflow model to TF-TRT inside NGC. 
- Prepare the docker base image.

## Exporting Model Using NGC

The following works are done on g4dn.2xlarge + ami-0184e674549ab8432
- g4dn.xlarge should be also workable.
- ami-0184e674549ab8432 （Deep Learning AMI (Ubuntu 18.04) Version 60.4) which has cuda driver installed by default.

We will use TF2.5 and TF2.7 as an example to guide the user through exporting the ssd_mobilenet inside the NGC.

### Why Using NGC ?
- By using NGC, we can spare the effort of installing the cuda cudnn and tensorrt on the host machine ourselves.
- Tensorflow version 2.5 can be compiled with either TRT7 or TRT8. And NGC provides different containers that has prebuilt tensorflow 2.x + trt 7/8 inside the container.
- Some of the version is even compaitable with the settings with Jetson released tensorflow wheel file.
- For more information, please refer to
  - [Installing Tensorflow on Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
  - [TensorFlow compatibility with NVIDIA containers and Jetpack](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel)
  - [NGC Tensorflow List](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_22-06.html#rel_22-06)

With the help of NGC, we just need to choose the right Nvidia Jetson provided TF version (and wheel file), and export model with the corresponding NGC. For example:
- Nvidia provides a [Jetson TF wheel file here](https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl). 
- After we look up inside the table in [TensorFlow compatibility with NVIDIA containers and Jetpack](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel), one can realize that this is built with Jetpack4.6.1 (i.e. TF2.7 + TRT8.2). And NGC Tensorflow 22.01 also provides the similar environment (i.e. tensorflow 2.7 + TRT8.2)
- Then we can export our model under nvcr.io/nvidia/tensorflow:22.01-tf2-py3

### Exporting Model with TF2.7 + TRT8.2

We will export the the model inside the NGC 22.01. Which has TF2.7 + TRT8.2 installed inside the contaienr.

```
docker run --gpus all -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    nvcr.io/nvidia/tensorflow:22.01-tf2-py3 bash
```

Inside docker container
```
apt-get update
apt-get install protobuf-compiler
git clone https://github.com/tensorflow/models.git tensorflow_models
cd tensorflow_models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .

# manual edit setup.py
# - add tensorflow==2.7.0
# - change tf-models-official>=2.7.0
# - change tensorflow_io==0.23.0
python3 -m pip install .

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


### Test the Exported Model and Convert it to TF-TRT

Test the exported model and calculate the FPS
Let's cd to the model directory first.

`cd output/ssd_mobilenet_v2_320x320_coco17_tpu-8_float_batchN_nms`

And execute the following python3 script.

```
import tensorflow as tf
import time
m = tf.saved_model.load("saved_model")
ff = m.signatures['serving_default']
x = tf.ones(shape=(8,300,300,3))
y = ff(x)
N = 1000
t1 = time.time()
for i in range(N):
    out = ff(x)
tt = time.time() - t1
print("exec time:", tt)
print(8*N/tt, "fps")
```

Lets convert the model using the following python3 script. This will export two model:
- TF-TRT with FP16
- TF-TRT with FP32

```
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
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

After the conversion, lets try the TF-TRT, to see if the FPS is better

```
import tensorflow as tf
import time
m = tf.saved_model.load("saved_model_trt_fp16")
ff = m.signatures['serving_default']
x = tf.ones(shape=(8,300,300,3))
y = ff(x)
N = 1000
t1 = time.time()
for i in range(N):
  out = ff(x)
tt = time.time() - t1
print("exec time:", tt)
print(8*N/tt, "fps")
```

Now we can copy the folder saved_model_trt_fp16 and put it inside our Panorama app.
- We have provided the model for you already. We will downalod the model inside the ipynb


## Prepare The Docker Image

- Please run the docker build insdie dependencies/docker_tf27_py36
- This docker image will install the cuda, cudnn, trt that is compaitable with Jetpack 4.6.1 and also install tensorflow 2.7 

```
docker build -t tf27:latest .
```

## Prepare the App with Jupyter Notebook

Let's prepare our tf27 app using the jupyter notebook.

Please open the TF27_opengpu.ipynb and follow along.

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