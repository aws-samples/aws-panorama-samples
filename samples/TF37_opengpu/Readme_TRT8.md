# Export Model Using NGC (Fast)

## Using TF2.5

```
docker run --gpus all -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    nvcr.io/nvidia/tensorflow:21.08-tf2-py3 bash
```

Inside docker
```
apt-get update
apt-get install protobuf-compiler
git clone https://github.com/tensorflow/models.git tensorflow_models
cd tensorflow_models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
# - add tensorflow==2.5.0
# - change tf-models-official>=2.4.0
# - change tensorflow_io==0.19.0
# - change keras==2.3.1
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

## Using TF2.7

```
docker run --gpus all -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    nvcr.io/nvidia/tensorflow:22.01-tf2-py3 bash
```

Inside docker
```
apt-get update
apt-get install protobuf-compiler
git clone https://github.com/tensorflow/models.git tensorflow_models
cd tensorflow_models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
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


## Test The Export Model and Convert to TF-TRT

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

> Note: in tf25, you should see logs like Successfully opened dynamic library libnvinfer.so.8
> But for tf27, there will be no log anymore.

> Note: in tf27, the converted model has lower FPS in T4 GPU. While TF25 has better performance. Not sure about the reason. Need more investigations.