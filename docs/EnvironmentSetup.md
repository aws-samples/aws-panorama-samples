# Test Utility environment setup

### Supported platforms

* ARM64 based EC2
* Windows + WSL2 + Ubuntu

Note: MacOS is currently not supported due to model compilation issues.

---

### Setup **ARM64 based EC2** environment

**Pre-Requisites**
1. You need to have an already created PEM key in your aws account and downloaded it to your local computer. [Instructions on how to create a PEM key](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair)


2. An S3 bucket created in the region of your choice that you can use in the Test Utility


**Launching the Samples**

1. Click the Launch Stack Button below. **NOTE** : This process will take about ```20 minutes```  
    * **US-EAST-1** :  
 [![Foo](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/template?stackName=arm-ec2-instance&templateURL=https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ec2-instance-panorama.yml)

    * **US-WEST-2** :  
 [![Foo](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/template?stackName=arm-ec2-instance&templateURL=https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/ec2-instance-panorama.yml)

2. Log into the EC2 Instance (See next section) and wait for the file ```INSTALLATION-COMPLETE.txt``` to appear on your ```/home/ubuntu```. This marks the end of the EC2 instance set up

    **OPTIONAL STEP** : If you would like to monitor the set up progress, log into the EC2 instance (See Next Section), and type ```tail -f /var/log/cloud-init-output.log``` in a terminal session. 

**Logging into the EC2 Instance**

1. From the AWS EC2 Console, get the **Public IPv4 DNS** for the instance you launched. It should look something like this
    ```sh
    ec2-1-234-567-8.compute-1.amazonaws.com
    ```
2. Make sure the PEM key that was created is in the same folder as you are. At this point, you can do this
    ```sh
    ssh -i "My_Awesome_Key.pem" ubuntu@ec2-1-234-567-8.compute-1.amazonaws.com
    ```
3. Launch Jupyter Lab Session from the Ec2 console
    ```sh
    sudo jupyter-lab --no-browser --allow-root
    ```
    You Should see something like this
    ```sh
    http://ip-123-45-678-910:8888/lab?token=e718819a3eb9b464aa81e14fe73439b49337e5d9fdef2676
    ```
    Note the Port (8888) and the token number

4. Creating a SSH tunnel (May not be necessary).Open an another terminal session on your Computer.  Make sure the port number here is the same as the output from ```Step 3```
    ```sh
    ssh -i My_Awesome_Key.pem -NL 8157:localhost:8888 ubuntu@ec2-1-234-567-8.compute-1.amazonaws.com
    ```

5. Launch your browser, paste the following address in your browser window
    ```sh
    http://localhost:8157/
    ```
    At this point it should ask for the token number, paste it and click Ok. You are inside your EC2 instance Jupyter Lab session  

6. Once logged into Jupyter Lab session, open a terminal session and run ```aws configure```. Fill in the Access Key and Secret key and Region for your account. 
    ```sh
    aws configure
    ``` 

---

### Setup **Windows + WSL2 + Ubuntu** environment

1. Install WSL2, Ubuntu and Docker Desktop on your Windows PC, and configure Ubuntu. Please refer to [Setting up a development environment in Windows](https://docs.aws.amazon.com/panorama/latest/dev/applications-devenvwindows.html)

2. Install DLR in Ubuntu by building from source code. Please refer to [Building on Linux - Building for CPU](https://neo-ai-dlr.readthedocs.io/en/latest/install.html#building-on-linux)

3. Install additional dependencies

    ```sh
    sudo pip3 install boto3 sagemaker matplotlib opencv-python --upgrade
    ```

4. Install Jupyter or JupyterLab in Ubuntu

    ```sh
    sudo pip3 install jupyterlab
    ```

5. Clone the 'aws-panorama-samples' repository in the Ubuntu filesystem.

    ```sh
    git clone https://github.com/aws-samples/aws-panorama-samples.git
    ```

6. Run jupyter server on Ubuntu

    ```sh
    jupyter-lab --no-browser --allow-root --port 8888 --notebook-dir ~
    ```
  
    You will see console output like below:
  
    ```
    Jupyter Server 1.13.0 is running at:
    http://localhost:8888/lab?token={token}
    ```
  
    Copy the token to clipboard.

7. Open your browser on Windows, and browse http://localhost:8888/.

    You will be asked to input the token. Please use the token you copied at previous step.

