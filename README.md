# Introduction to AWS Panorama

AWS Panorama is a service that enables you to run computer vision applications at the edge. By using the AWS Panorama Appliance with your existing network cameras, you can run applications that use machine learning to collect data from video streams, output video with text and graphical overlays, and interact with other AWS services.

## AWS Panorama Computer Vision Examples

This repository provides some examples to get you kick-started on building applications for AWS Panorama.

### Resources

* [AWS Panorama Documentation](https://docs.aws.amazon.com/panorama/)

Please refer to README file in each folder for more specific instructions.

### How to Run a Sample
* We recommend using a Sagemaker Jupyter Notebook Instance
  + Go to your [AWS Sagemaker console](https://aws.amazon.com/sagemaker/)
  + Click  `Notebook Instances -> Create`
  + Choose the instance type (These were built on P2 instances)
  + Choose the Volume size in GB (5 GB should be enough)
  + Create Notebook Instance 
* Once your notebook instance is created, click the name of your Notebook instance
  + Go to Permissions and encryption
  + Click on the IAM role arn
  + In permissions, attach the following policies
    + `AWSLambdaFullAccess`
    + `IAMFullAccess`
    + `AmazonS3FullAccess`
    + `AWSIoTFullAccess`
    + `AmazonSageMakerFullAccess`
* Restart your Notebook Instance and launch Jupyter Lab
* Launch a terminal session and do the following
    + `cd SageMaker`
* Clone the repository 
  `git clone https://github.com/aws-samples/aws-panorama-samples.git`
* `cd aws-panorama-samples`
* `wget https://panorama-starter-kit.s3.amazonaws.com/public/v1/Models/Models.zip`
* `unzip -q Models.zip`
* `sudo rm Models.zip`
* `wget https://panorama-starter-kit.s3.amazonaws.com/public/v1/Models/panorama_sdk.zip`
* `unzip -q panorama_sdk.zip`
* `sudo rm panorama_sdk.zip`
* `cd ..`
* `sudo sh aws-panorama-samples/panorama_sdk/run_after_clone.sh`
* We suggest using `conda_mxnet_latest_p37` kernel for all use cases (Except specified in the individual README)
* At this point, the set up is done. You can explore the different applications in each of the folders
* Follow the README in the individual examples for information about the use case

### List of Samples

For each of the samples below, we include instructions on how to deploy them to the edge Panorama device inside the notebook. 


| Application | README | Description | Framework | Usecase | Complexity
| ------ | ------ |------ |------ |------ |------ |
| **People Counter** | [README.md](PeopleCounter/README.md) | This is a sample computer vision application that can count the number of people in each frame of a streaming video (**Start with this**) | MXNet | Object Detection | Easy
| **Custom Object Detector** | [README.md](PikachuDetection/README.md) | This is a sample computer vision application that showcases how to build your own models using Gluoncv, and then deploy them on the AWS Panorama device | MXNet | Object Detection | Medium
| **Fall Detection** | [README.md](FallDetection/README.md) |This is a sample computer vision application that showcases how to build a fall detection computer vision application,and then deploy them on the AWS Panorama device. This also showcases how to use multiple models for a single usecase on the Panorama device | MXNet | Pose Detection & Object Detection | Advanced
| **Social Distance Calculation** | [README.md](SocialDistance/README.md) | This is an advanced use case where we build a sample computer vision application that uses object detection models and some simple math to detect social distancing infractions | MXNet | Object Detection | Advanced
| **Handwash Detection** | [README.md](HandWashingDetection/README.md) |This is a sample computer vision application that showcases how to detect Hand washing using an action detection model | MXNet |  Action Detection | Easy
| **Smoking Detection** | [README.md](SmokingDetection/README.md) | This is a sample computer vision application that showcases how to detect somone Smoking using an action detection model | MXNet |  Action Detection | Easy
| **Image Classification** | [README.md](ImageClassification/README.md) | This is a sample computer vision application that showcases build a simple image classification model using AWS Panorama | MXNet |  Image Classification | Easy
| **Semantic Segmentation** | [README.md](SemanticSegmentation/README.md) | This is a sample computer vision application that showcases how to use a Gluoncv Segmentation model and build a segmentation use case | MXNet |  Semantic Segmentation | Medium

## Getting Help
We use [AWS Panorama Samples GitHub issues](https://github.com/aws-samples/aws-panorama-samples/issues) for tracking questions, bugs, and feature requests.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.