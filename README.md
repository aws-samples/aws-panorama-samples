# AWS Panorama Samples and Test Utility
  

## Introduction

AWS Panorama is a machine learning appliance and SDK, which enables you to add computer vision (CV) to your on-premises cameras or on new AWS Panorama enabled cameras. AWS Panorama gives you the ability to make real-time decisions to improve your operations, by giving you compute power at the edge.

This repository contains **sample applications** for AWS Panorama, and **Test Utility** which allows running Panorama applications in simulation environment without real Panorama appliance device.


### About Test Utility

Test Utility is a set of python libraries and commandline commands, which allows you to test-run Panorama applications without Panorama appliance device. With Test Utility, you can start running sample applications and developing your own Panorama applications before preparing real Panorama appliance. Sample applications in this repository also use Test Utility. 

For **more about the Test Utility and its current capabilities**, please refer to [Introducing AWS Panorama Test Utility](docs/AboutTestUtility.md) document.

To **set up your environment** for Test Utility, please refer to [Test Utility environment setup](docs/EnvironmentSetup.md).

To know **how to use Test Utility**, please refer to [How to use Test Utility](docs/HowToUseTestUtility.md).


## List of Samples

| Application | Description | Framework | Usecase | Complexity | Model 
| ------ | ------ |------ |------ |------ |------ |
| **People Counter**| This is a sample computer vision application that can count the number of people in each frame of a streaming video (**Start with this**) | MXNet | Object Detection | Easy | [Download](https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ssd_512_resnet50_v1_voc.tar.gz)
| **Car Detector and Tracker**| This is a sample computer vision application that can detect and track cars | Tensorflow | Object Detection | Medium | [Download](https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ssd_mobilenet_v2_coco.tar.gz)
| **Pose estimation**| This is a sample computer vision application that can detect people and estimate pose of them | MXNet | Pose estimation | Advanced | yolo3_mobilenet1.0_coco, simple_pose_resnet152_v1d
| **Object Detection Tensorflow SSD**| This example shows how to run a TF SSD Mobilenet Model using Tensorflow | Tensorflow (Open GPU) | Object Detection (BYO Container) | Advanced | 
| **Object Detection PyTorch Yolov5s**| This example shows how to run your own YoloV5s model using PyTorch | PyTorch (Open GPU) | Object Detection (BYO Container) | Advanced | 
| **Object Detection ONNX Runtime Yolov5s**| This example shows how to run your own YoloV5s model using ONNX Runtime | ONNX Runtime (Open GPU) | Object Detection (BYO Container) | Advanced |


## Running the Samples

**Step 1** : Go to aws-panorama-samples/samples and open your choice of project  
**Step 2** : Open the .ipynb notebook and follow the instructions in the notebook  
**Step 3** : To make any changes, change the corresponding node package.json or the graph.json in the application folder  

For more information, check out the documentation for the AWS Panorama DX CLI [here](https://github.com/aws/aws-panorama-cli)

## Documentations

* [AWS Panorama
Developer Guide](https://docs.aws.amazon.com/panorama/latest/dev/)
* [AWS Panorama Service API Reference](https://docs.aws.amazon.com/panorama/latest/api/)
* [AWS Panorama boto3 reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/panorama.html)
* [AWS Panorama Application SDK reference](https://github.com/awsdocs/aws-panorama-developer-guide/blob/main/resources/applicationsdk-reference.md)
* [panorama-cli document](https://github.com/aws/aws-panorama-cli/blob/main/README.md)


## Getting Help

We use [AWS Panorama Samples GitHub issues](https://github.com/aws-samples/aws-panorama-samples/issues) for tracking questions, bugs, and feature requests.

## License

This library is licensed under the MIT-0 License. 
