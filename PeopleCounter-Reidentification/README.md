
# AWS Panorama People Counter with Person Reidentification Example

This is an end to end example that demonstrates
- how to use an Object Detection Model to detect a person
- how to use a Person Reidentification Model to track a person across multiple frames
- how to use a Directional tracker to count the person entering or exiting the building (i.e. position in the frame)

### Prerequisites
- Follow the instructions in https://github.com/aws-samples/aws-panorama-samples/blob/main/README.md
- This application uses two models. One of the models is a ssd_512_resnet50_v1_voc MXNet model (ssd_512_resnet50_v1_voc.tar.gz) and is part of the instructions in Step 1.

### Other resources to use
- [AWS Panorama Documentation](https://docs.aws.amazon.com/panorama/)

### How to build the torch reid model
1. Create an Amazon SageMaker Notebook with GPU support (e.g.: ml.p3.2xlarge) and at least 300 GB EBS (300 GB is large enough for training images.)
2. When the Notebook instance is ready, open JupyterLab
3. Open the terminal window and change folder to ~/SageMaker
4. Clone the repository: https://github.com/KaiyangZhou/deep-person-reid.git 
5. Run the following commands in the terminal

    cd deep-person-reid/
    conda create --name torchreid python=3.7
    source activate torchreid
    pip install -r requirements.txt
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    conda install -c conda-forge tar 
    conda install ipywidgets
    conda install boto3
    conda install tensorflow
    pip install gluoncv mxnet-mkl>=1.4.0 --upgrade
    python setup.py develop

6. Create a new Notebook using conda_torchreid environment and follow the instructions in ../Notebook/Torch-reid.ipynb

### How to use the Notebook
The included Jupyter Notebook gives a helpful introduction of 
- Task at hand 
- Step by step walk through of the object detection code
- Overview of the feature extraction used by reidentification code
- Step by step walk through of the directional tracking code

Please follow the instructions in ../Notebook/Person-Reidentification.ipynb

### How to Deploy the lambda function
The lambda function is built using [AWS Serverless Application Model (SAM)](https://aws.amazon.com/serverless/sam/). The function is accompanied by the application specification file - template.yaml.  Our recommandation is for you to use the (--guided) option to deploy the application for the first time; Save the output and re-use the file for future deployments. 

Here are the commands to deploy the lambda function and publish the new version.

    sam deploy --guided 
    FUNCTIONNAME=`aws lambda list-functions | jq -r '.Functions[].FunctionName' | grep deep-sort`
    FUNCTIONARN=`aws lambda get-function --function-name $FUNCTIONNAME | jq '.Configuration.FunctionArn'`
    aws lambda publish-version --function-name $FUNCTIONNAME