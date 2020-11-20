# Panorama People Counter Example

This is an end to end example that shows how to use an Object Detection Model to count people

## Files Included
- Lambda (Folder)
	- people_counter.py 
	- people-counter.zip
- Notebook(Folder)
	- People_Counter_Panorama_Example.ipynb
    - street_empty.jpg
	- requirements.txt

- ssd_512_resnet50_v1_voc.tar.gz (Model to Use)

### Use Case
- Detect people in each frame using ssd_512_resnet50_v1_voc MXNet model
- Once people are detected, we count the number of people detected and display on the frame

### How to use the Notebook
The included Jupyter Notebook gives a helpful introduction of 
- Task at hand 
- Step by step walk thru of the MXNet code
- Understanding the Lambda structure by creating code in the same format
- Creating a Lambda function by uploading the included Lambda zip file
- Publishing the Lambda and displaying the version number and the Lambda console link

### How to use the Lambda Function

The included Lambda function is a zip file that can be directly uploaded to the Lambda console to create a usable Lambda arn. 

### Other resources to use

- [AWS Panorama Documentation](https://docs.aws.amazon.com/panorama/)
