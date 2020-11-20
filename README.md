# AWS Panorama Computer Vision Examples

### Resources

* [AWS Panorama Documentation](https://docs.aws.amazon.com/panorama/)

Please refer to README file in each folder for more specific instructions.

### How to Run a Sample
* We recommend using a Sagemaker Jupyter Notebook Instance
  + Go to your AWS Sagemaker console
  + Click  `Notebook Instances -> Create`
  + Choose the instance type
  + Choose the Volume size in GB (1 to 2 GB should be enough)
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
* Without going into the aws-panorama-samples do the following
  `sudo sh aws-panorama-samples/panorama_sdk/run_after_clone.sh`
* We suggest using `conda_mxnet_latest_p37` kernel for all use cases (Except specified in the individual README)
* At this point, the set up is done. You can explore the different applications in each of the folders
* Follow the README in the individual examples for information about the use case

### List of Samples

* [People Counter Application](PeopleCounter/) This is a sample computer vision application that can count the number of people in each frame of a streaming video (**Start with this**)
 * [Pikachu Detection](PikachuDetection/) This is a sample computer vision application that showcases how to build your own models using Gluoncv, and then deploy them on the AWS Panorama device
 * [Fall Detection](FallDetection/) This is a sample computer vision application that showcases how to build a fall detection computer vision application,and then deploy them on the AWS Panorama device
 * [Hand Washing Detection](HandWashingDetection/) This is a sample computer vision application that showcases how to detect Hand washing using an action detection model. We then show how this can be deployed to the edge, using the AWS Panorama device
 * [Image Classification](ImageClassification/) This is a sample computer vision application that showcases build a simple image classification model using AWS Panorama. We then show how this can be deployed to the edge, using the AWS Panorama device
 * [Semantic Segmentation](SemanticSegmentation/) This is a sample computer vision application that showcases how to use a Gluoncv Segmentation model and build a segmentation use case. We then show how this can be deployed to the edge, using the AWS Panorama device
 * [Smoking Detection](SmokingDetection/) This is a sample computer vision application that showcases how to detect somone Smoking using an action detection model. We then show how this can be deployed to the edge, using the AWS Panorama device
 * [Social Distance Calculation](SocialDistance/) This is an advanced use case where we build a sample computer vision application that uses object detection models and some simple math to detect social distancing infractions. We then show how this can be deployed to the edge, using the AWS Panorama device

## Getting Help
We use [AWS Panorama Samples GitHub issues](https://github.com/aws-samples/aws-panorama-samples/issues) for tracking questions, bugs, and feature requests.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
