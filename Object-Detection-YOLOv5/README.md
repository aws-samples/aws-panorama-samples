# AWS Panorama Object Detection with YOLOv5 Example

This is an end to end example that shows how to deploy a YOLOv5 machine learning model, pre-trained with MS COCO dataset using PyTorch, to a Panorama appliance.

## Files Included

- utils.py
- yolov5s_lambda.py
- yolov5s.ipynb
- test.png
- test-result.png

### Use Case

- Build a custom object detector using a pre-trained model using PyTorch
- First step towards training your own PyTorch trained object detection model with YOLOv5

### Setup

This sample includes a submodule pointing to [YOLOv5 repository](https://github.com/ultralytics/yolov5). Make sure to add `--recursive` when pulling the code from this repository.

### How to use the Notebook

The included Jupyter Notebook gives a guided tour of deploying a pre-trained YOLOv5 ML model. Follow the step by step instructions in the Notebook to deploy the provided model and Application code to the Panorama appliance. We recommend using SageMaker Notebook Instance to run the Notebook as it will have most of the dependencies pre-installed (including the AWS CLI tool). Select `conda_python3` kernel is using Amazon SageMaker Notebook Instance.

If the Notebook is not executed on Amazon SageMaker Notebook Instance then you'll have to install and configure AWS CLI tools (see the [Resources](#Resources) section for details) 

### Example Output

This is an example output from running the inference on a Panorama appliance

![alt Test image inference results](test-result.png "Test image inference results")


### How to use the Lambda Function

Notebook has all the code necessary to pack the included Lambda function into a zip file and deploy it. Alternatively, that zip file can be directly uploaded to the Lambda console to create a Lambda. 

## Warning

This package depends on and may incorporate or retrieve a number of third-party
software packages (such as open source packages) at install-time or build-time
or run-time ("External Dependencies"). The External Dependencies are subject to
license terms that you must accept in order to use this package. If you do not
accept all of the applicable license terms, you should not use this package. We
recommend that you consult your company’s open source approval policy before
proceeding.

Provided below is a list of External Dependencies and the applicable license
identification as indicated by the documentation associated with the External
Dependencies as of Amazon's most recent review.

THIS INFORMATION IS PROVIDED FOR CONVENIENCE ONLY. AMAZON DOES NOT PROMISE THAT
THE LIST OR THE APPLICABLE TERMS AND CONDITIONS ARE COMPLETE, ACCURATE, OR
UP-TO-DATE, AND AMAZON WILL HAVE NO LIABILITY FOR ANY INACCURACIES. YOU SHOULD
CONSULT THE DOWNLOAD SITES FOR THE EXTERNAL DEPENDENCIES FOR THE MOST COMPLETE
AND UP-TO-DATE LICENSING INFORMATION.

YOUR USE OF THE EXTERNAL DEPENDENCIES IS AT YOUR SOLE RISK. IN NO EVENT WILL
AMAZON BE LIABLE FOR ANY DAMAGES, INCLUDING WITHOUT LIMITATION ANY DIRECT,
INDIRECT, CONSEQUENTIAL, SPECIAL, INCIDENTAL, OR PUNITIVE DAMAGES (INCLUDING
FOR ANY LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, OR
COMPUTER FAILURE OR MALFUNCTION) ARISING FROM OR RELATING TO THE EXTERNAL
DEPENDENCIES, HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, EVEN
IF AMAZON HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. THESE LIMITATIONS
AND DISCLAIMERS APPLY EXCEPT TO THE EXTENT PROHIBITED BY APPLICABLE LAW.

***YOLOv5***: [GPL-3.0 License](https://github.com/ultralytics/yolov5/blob/master/LICENSE)

### Resources

- [AWS Panorama Documentation](https://docs.aws.amazon.com/panorama/)
- [Transfer learning](https://github.com/ultralytics/yolov5/issues/1314)
- [How to install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)