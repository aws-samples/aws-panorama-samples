# How to use Test Utility

Test Utility has a commandline interface. You can use the commandline interface both in terminal environment and Jupyter environment.


### Steps

1. Make sure graph.json, package.json, descriptor.json, are prepared and edited correctly.
    * Test Utility uses these files to run the application.

2. Change your working directory to the parent directory of your Panorama application directory. For example, if your application structured as beloe, you need to change your working directory to people_counter.
    * people_counter/
        * people_counter_app/
            * assets/
            * graphs/
            * packages/

3. Run the Test Utility command (samples/common/test_utility/panorama_test_utility_main.py).
    * When you run this command for the first time, it compiles models for the platform you are using to run the Test Utility, and it takes time.
    * There are many parameters you need to specify. For details, please refer to the section below. (We are planning to reduce the number of parameters by automatically finding in graph/package JSON files.)

4. Check if your application runs as expected.
    * Applications stdout/stderr are directly printed in your terminal or Jupyter output cell.
    * HDMI output is simulated by sequentially numbered PNG files.
    
5. As needed, edit your Python script and repeat Step 3 ~ Step 4.

---

### Commandline reference

```
$ python3 panorama_test_utility_main.py --help

usage: panorama_test_utility_main.py [-h] [--region REGION] --app-name
                                     APP_NAME --code-package-name
                                     CODE_PACKAGE_NAME --model-package-name
                                     MODEL_PACKAGE_NAME --camera-node-name
                                     CAMERA_NODE_NAME --s3-model-location
                                     S3_MODEL_LOCATION --model-node-name
                                     MODEL_NODE_NAMES --model-file-basename
                                     MODEL_FILE_BASENAMES --model-data-shape
                                     MODEL_DATA_SHAPES --model-framework
                                     MODEL_FRAMEWORKS --video-file VIDEO_FILE
                                     [--video-start VIDEO_START]
                                     [--video-stop VIDEO_STOP]
                                     [--video-step VIDEO_STEP]
                                     [--screenshot-dir SCREENSHOT_DIR]
                                     --py-file PY_FILE

Panorama Test-Utility

optional arguments:
  -h, --help            show this help message and exit
  --region REGION       Region name such as us-east-1
  --app-name APP_NAME   Application name
  --code-package-name CODE_PACKAGE_NAME
                        Code package name
  --model-package-name MODEL_PACKAGE_NAME
                        Model package name
  --camera-node-name CAMERA_NODE_NAME
                        Camera node name
  --s3-model-location S3_MODEL_LOCATION
                        S3 location for model compilation. e.g.
                        s3://mybucket/myapp/
  --model-node-name MODEL_NODE_NAMES
                        Model node name
  --model-file-basename MODEL_FILE_BASENAMES
                        Model filename excluding .tar.gz part
  --model-data-shape MODEL_DATA_SHAPES
                        Model input data shape. e.g. {"data":[1,3,512,512]}
  --model-framework MODEL_FRAMEWORKS
                        Model framework name. e.g. MXNET
  --video-file VIDEO_FILE
                        Video filename to simulate camera stream
  --video-start VIDEO_START
                        Video start frame (default: 0)
  --video-stop VIDEO_STOP
                        Video stop frame (default: 30)
  --video-step VIDEO_STEP
                        Video frame step (default: 1)
  --screenshot-dir SCREENSHOT_DIR
                        Directory name to save screenshot files. You can use
                        Python's datetime format.
  --py-file PY_FILE     Python source path to execute
```

### Commandline samples


**Single model**

```sh
python3 ../common/test_utility/panorama_test_utility_main.py \
    --app-name people_counter_app \
    --code-package-name PEOPLE_COUNTER_CODE \
    --model-package-name SSD_MODEL \
    --camera-node-name abstract_rtsp_media_source \
    --s3-model-location s3://shimomut-panorama-test-us-east-1/people_counter_app \
    \
    --model-node-name model_node \
    --model-file-basename ./models/ssd_512_resnet50_v1_voc \
    --model-data-shape '{"data":[1,3,512,512]}' \
    --model-framework MXNET \
    \
    --video-file ../common/test_utility/videos/TownCentreXVID.avi \
    --screenshot-dir ./screenshot/%Y%m%d-%H%M%S \
    --py-file ./people_counter_app/packages/357984623133-PEOPLE_COUNTER_CODE-1.0/src/app.py
```

**Multiple models**

```sh
python3 ../common/test_utility/panorama_test_utility_main.py \
    --app-name pose_estimation_app \
    --code-package-name pose_estimation_code \
    --model-package-name pose_estimation_models \
    --camera-node-name abstract_rtsp_media_source \
    --s3-model-location s3://shimomut-panorama-test-us-east-1/pose_estimation_app \
    \
    --model-node-name people_detection_model \
    --model-file-basename ./models/yolo3_mobilenet1.0_coco_person \
    --model-data-shape '{"data":[1,3,480,600]}' \
    --model-framework MXNET \
    \
    --model-node-name pose_estimation_model_1 \
    --model-file-basename ./models/simple_pose_resnet152_v1d \
    --model-data-shape '{"data":[1,3,256,192]}' \
    --model-framework MXNET \
    \
    --model-node-name pose_estimation_model_2 \
    --model-file-basename ./models/simple_pose_resnet152_v1d \
    --model-data-shape '{"data":[2,3,256,192]}' \
    --model-framework MXNET \
    \
    --model-node-name pose_estimation_model_3 \
    --model-file-basename ./models/simple_pose_resnet152_v1d \
    --model-data-shape '{"data":[3,3,256,192]}' \
    --model-framework MXNET \
    \
    --model-node-name pose_estimation_model_4 \
    --model-file-basename ./models/simple_pose_resnet152_v1d \
    --model-data-shape '{"data":[4,3,256,192]}' \
    --model-framework MXNET \
    \
    --video-file ./internal/test_videos/dance_480_bf0.mp4 \
    --video-start 1000 \
    --video-stop 1000 \
    --video-step 10 \
    \
    --screenshot-dir ./screenshot/%Y%m%d-%H%M%S \
    --py-file ./pose_estimation_app/packages/357984623133-pose_estimation_code-1.0/src/app.py
```
