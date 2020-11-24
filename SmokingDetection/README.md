# Panorama Smoking Detection Example

This is an end to end example that shows how to use an action detection model to detect smoking and time it.

## Files Included
- Lambda (Folder)
	- config.py 
	- action_detection.py
	- SmokingDetectionLambda.zip
- Notebook(Folder)
	- Smoking-Panorama-Example.ipynb
- smoking.mp4 (Video to test)
- resnet101_v1b_kinetics400.tar.gz (Model to Use)

### Use Case
- Detect Smoking using the resnet101_v1b_kinetics400 action detection model. 
- Once the action is detected, 


### How to use the Notebook
The included Jupyter Notebook gives a helpful introduction of 
- Task at hand 
- Step by step walk thru of the MXNet / Panorama SDK code
- Understanding the Lambda structure by creating code in the same format
- Creating a Lambda function by uploading the included Lambda zip file
- Publishing the Lambda and displaying the version number and the Lambda console link

### Example Output From Notebook

The output displays the top 3 actions detected on the screen. 

![Example Notebook](Example_Image_Notebook.png)


### How to use the Lambda Function

The included Lambda function is a zip file that can be directly uploaded to the Lambda console to create a usable Lambda arn. 

### Other resources to use

- [AWS Panorama Documentation](https://docs.aws.amazon.com/panorama/)
