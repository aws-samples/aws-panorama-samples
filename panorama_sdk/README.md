# Introducing AWS Panorama Test Utility

Application deployment for edge CV applications involves many moving pieces including compiling the model, setting up the application, and connecting the cameras. AWS Panorama simplifies all these pieces by making it easier for customers to write simple Python code and deploy their models with managing their own hardware or application runtime. 

To accelerate application development, AWS Panorama has bundled the Test Utility for AWS Panorama within the AWS Panorama samples code repo. This Python-based utility is a functionally complete mock of the actual panoramasdk that runs inside the base container on the device. Developers can use the Python library to test their code and iterate prior to deploying to a device.

We recommend developers start using the Test Utility by launching an EC2 test instance using our CloudFormation template. Using the CloudFormation template, the EC2 instance has the same ARM architecture and base operating system (Ubuntu 18.04) as the default base container image which developers’ applications will run on. As a result of using the same ARM architecture on the development environment as the device, developers will have a higher confidence (but not guaranteed) that any additional libraries they bring into their application will be compiled for the correct architecture. Developer interact with the Test Utility through Jupyter notebooks, an interactive Python editing environment. With Jupyter notebooks, developers can see a preview of the HDMI output as it would appear on a monitor connected to the device. 

## FAQ

**What are the limitations of the Test Utility?**



| Task | Device | Test Utility | Notes |
| ------ | ------ |------ | ------ |
|Measure performance |	Yes	| No |
|Validate application graph | 	Yes |	No |	The application graph is metadata that defines how cameras should connect to models and business logic code. The Test Utility can catch some errors in the application graph, but does not guarantee that the app graph will work when deployed to the device. |
|Check Neo compatibility of the model |	Yes |	Somewhat |	While we expect little to no issues in compiling model to run with the Test Utility vs AWS Panorama appliance, we cannot guarantee the model will run the same due to differences in CPU/GPU available in the EC2 environment vs AWS Panorama appliance.|
|Process RTSP streams |	Yes	| Coming soon |	The Test Utility cannot consume RTSP camera streams. Developers use static videos to test their application. This has not been an issue as during early development, when the Test Utility is most useful, developers want repeatable tests.|
|Use multiple models |	Yes	|Coming soon |	The Test Utility does not support multiple models or multiple applications to run simultaneously,  while the AWS Panorama appliance does. The Test Utility maintainers are looking at this as a roadmap item and will prioritize based on customer feedback.|
|Use mutiple data sources|	Yes	|Coming soon|	The Test Utility does not support multiple data sources feeding into an application simultaneously.|


**What if I find an issue using the Test Utility?**

File an issue on Github.
