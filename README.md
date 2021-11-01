# AWS Panorama Samples and Test Utility
  

## Introduction

> AWS Panorama is a machine learning appliance and SDK, which enables you to add computer vision (CV) to your on-premises cameras or on new Panorama enabled cameras. AWS Panorama gives you the ability to make real-time decisions to improve your operations, by giving you compute power at the edge.


## Section 1: Setting Up Samples

### Pre-Requisites
* **Step 1**: You need to have an already created PEM key in your aws account and downloaded it to your local computer. [Instructions on how to create a PEM key](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair)

* **Step 2**: You need to subscribe to the Ubuntu 18.04 LTS Bionic ARM AMI on AWS Marketplace. Please go to [this link](https://aws.amazon.com/marketplace/pp/prodview-5cjjlmwk54f2o?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) and click "Continue to Subscribe". "Accept Terms" on the next page completes the Subscription

* **Step 3**: An S3 bucket created in the region of your choice that you can use in the Test Utility


### Launching EC2 ARM Instance  

* **Step 1**: Click the Launch Stack Button below. **NOTE** : This process will take about ```20 minutes```  
    * **US-EAST-1** :  
 [![Foo](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/template?stackName=arm-ec2-instance&templateURL=https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ec2-instance-panorama.yml)

    * **US-WEST-2** :  
 [![Foo](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/template?stackName=arm-ec2-instance&templateURL=https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ec2-instance-panorama.yml)

* **Step 2**: Log into the EC2 Instance (See next section) and wait for the file ```INSTALLATION-COMPLETE.txt``` to appear on your ```/home/ubuntu```. This marks the end of the EC2 instance set up
* **OPTIONAL STEP** : If you would like to monitor the set up progress, log into the EC2 instance (See Next Section), and type ```tail -f /var/log/cloud-init-output.log``` in a terminal session. 

### Logging into the EC2 Instance    

* **Step 1**:  From the AWS EC2 Console, get the **Public IPv4 DNS** for the instance you launched. It should look something like this
    ```sh
    ec2-1-234-567-8.compute-1.amazonaws.com
    ```
* **Step 2**: Make sure the PEM key that was created is in the same folder as you are. At this point, you can do this
    ```sh
    ssh -i "My_Awesome_Key.pem" ubuntu@ec2-1-234-567-8.compute-1.amazonaws.com
    ```
* **Step 3**: Launch Jupyter Lab Session from the Ec2 console
    ```sh
    sudo jupyter-lab --no-browser --allow-root
    ```
    You Should see something like this
    ```sh
    http://ip-123-45-678-910:8888/lab?token=e718819a3eb9b464aa81e14fe73439b49337e5d9fdef2676
    ```
    Note the Port (8888) and the token number

* **Step 4**: Creating a SSH tunnel (May not be necessary).Open an another terminal session on your Computer.  Make sure the port number here is the same as the output from ```Step 3```
    ```sh
    ssh -i My_Awesome_Key.pem -NL 8157:localhost:8888 ubuntu@ec2-1-234-567-8.compute-1.amazonaws.com
    ```
    
* **Step 5**: Launch your browser, paste the following address in your browser window
    ```sh
    http://localhost:8157/
    ```
    At this point it should ask for the token number, paste it and click Ok. You are inside your EC2 instance Jupyter Lab session  

* **Step 6**: Once logged into Jupyter Lab session, open a terminal session and run ```aws configure```. Fill in the Access Key and Secret key and Region for your account. 
    ```sh
    aws configure
    ``` 

## Section 2: List of Samples

| Application | Description | Framework | Usecase | Complexity | Model | Contributors
| ------ | ------ |------ |------ |------ |------ |------ |
| **People Counter**| This is a sample computer vision application that can count the number of people in each frame of a streaming video (**Start with this**) | MXNet | Object Detection | Medium | [Download](https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ssd_512_resnet50_v1_voc.tar.gz) | Surya Kari, Phu Nguyen
| **Car Detector and Tracker**| This is a sample computer vision application that can detect and track cars | Tensorflow | Object Detection | Advanced | [Download](https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ssd_mobilenet_v2_coco.tar.gz)| Surya Kari, Phu Nguyen
| **Shelf Monitoring**| This is a sample computer vision application that can detect and count bottles on a shelf | MXNet| Object Detection | Advanced | [Download](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/yolo3_darknet53_coco-09767802.zip) <br /> [Shelf Monitoring Application Demo](https://aws.amazon.com/blogs/machine-learning/build-a-shelf-monitoring-application-using-aws-panorama/) | Amit Mukherjee, Laith Al-Saadoon, Sourabh Agnihotri


## Section 3: Running the Samples

**Step 1** : Go to awspanoramasamples/samples and open your choice of project  
**Step 2** : Open the .ipynb notebook and follow the instructions in the notebook  
**Step 3** : To make any changes, change the corresponding node package.json or the graph.json in the application folder  

For more information, check out the documentation for the Panorama DX CLI [here](https://github.com/aws/aws-panorama-cli)

## Section 4: Documentation

* [Developer Guide](https://docs.aws.amazon.com/panorama/latest/dev/panorama-releases.html)
* [API Guide](https://docs.aws.amazon.com/panorama/latest/api/API_Operations.html)
* [Dev Guide Github](https://github.com/awsdocs/aws-panorama-developer-guide)

## Getting Help

We use [AWS Panorama Samples GitHub issues](https://github.com/aws-samples/aws-panorama-samples/issues) for tracking questions, bugs, and feature requests.

## License

This library is licensed under the MIT-0 License. 
