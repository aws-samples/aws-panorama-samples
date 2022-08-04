import boto3
import os
import json
import datetime
from time import gmtime, strftime
from boto3.session import Session

import sagemaker as sage
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

from sagemaker.debugger import (Rule,
                                rule_configs,
                                ProfilerConfig, 
                                FrameworkProfile, 
                                DetailedProfilingConfig, 
                                DataloaderProfilingConfig, 
                                PythonProfilingConfig)

region = boto3.session.Session().region_name

sagemaker = boto3.client('sagemaker')
code_pipeline = boto3.client('codepipeline')
sqs_client = boto3.client(
    "sqs")


def lambda_handler(event, context):
    try:
        
        print(event)
             
        train_start = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        train_start_calc = datetime.datetime.now()
    
        codepipeline_job = event['CodePipeline.job']['id']
        print('[INFO]CODEPIPELINE_JOB:', codepipeline_job)
        print('[INFO]TRAIN_START:', train_start)
        
        userParamText = event['CodePipeline.job']['data']['actionConfiguration']['configuration']['UserParameters']
        user_param = json.loads(userParamText)
        job_name = 'avastus-yolov5s-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        print('[INFO]TRAINING_JOB_NAME:', job_name)
        
        # Get path to the most recent image built in the previous BUILD stage of the pipeline
        # Get Account Id from lambda function arn
        LambdaArn = context.invoked_function_arn
        print("lambda arn: ", LambdaArn)
        # Get Account ID from lambda function arn in the context
        AccountID = context.invoked_function_arn.split(":")[4]
        print("Account ID=", AccountID)
        
        region = context.invoked_function_arn.split(":")[3]
        print("region is:", region)
        
        #sqs message
        queue_url = 'https://queue.amazonaws.com/{}/avastus-queue'.format(AccountID)
        messages = sqs_client.receive_message(
                        QueueUrl=queue_url,
                        MaxNumberOfMessages=1,
                    )
        
        print('sqs avastus-q messages', messages)
                    
        sqs_message = messages["Messages"][0]["Body"]
        print('SQS Message ', sqs_message)
        bucket = sqs_message.split(',')[0].split(':')[-1]
        datafolder = sqs_message.split(',')[1].split(':')[-1]
        epochs = sqs_message.split(',')[2].split(':')[-1]
        
        code_location = f's3://{bucket}/avastus_yolov5/sm_codes'
        output_path = f's3://{bucket}/avastus_yolov5/output' 
        s3_log_path = f's3://{bucket}/avastus_yolov5/tf_logs'
    
        event['job_name'] = job_name
        event['stage'] = 'Training'
        event['status'] = 'InProgress'
        event['message'] = 'training job "{} started."'.format(job_name)

        print("Current Working Directory", os.getcwd())

        print('Downloading Yolov5 directory')
        s3_client = boto3.client('s3')
        s3_client.download_file('{}'.format(bucket), 'avastus_yolov5/source_code/yolov5.tar.gz', '/tmp/yolov5.tar.gz')
        
        os.chdir('/tmp')
        print('untar')
        os.system('tar -xvf yolov5.tar.gz .')
        print('remove')
        os.system('rm -r yolov5.tar.gz')
        
        #create_training_job(user_param, job_name, AccountID)
        print('Training Job Kick Off')
        trainingjob(AccountID, region, job_name, bucket, datafolder,  code_location, output_path, s3_log_path, epochs)
        write_job_info_s3(event)
        put_job_success(event, train_start_calc)

    except Exception as e:
        print(e)
        print('[ERROR] Unable to create training job.')
        event['message'] = str(e)
        put_job_failure(event)

    return event
    
def trainingjob(account, region, job_name, bucket, datafolder, code_location, output_path, s3_log_path, epochs):
    
    metric_definitions = [
            {'Name': 'Precision', 'Regex': r'all\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)'},
            {'Name': 'Recall', 'Regex': r'all\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)'},
            {'Name': 'mAP@.5', 'Regex': r'all\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)'},
            {'Name': 'mAP@.5:.95', 'Regex': r'all\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)'}
        ]

    print('before epoks')
    epoks = int(float(epochs.replace('"','')))
    
    print('type epochs ', type(epoks))
    
    hyperparameters = {
        'data': 'data_sm.yaml',
        'cfg': 'yolov5s.yaml',
        'weights': 'weights/yolov5s.pt',
        'batch-size': 128,
        'epochs': epoks,
    #     'epochs': 1,
        'project': '/opt/ml/model',
        'workers': 0,
        'freeze': 10
    }
    
    experiment_name = ''
    instance_type = 'ml.p3.2xlarge'  # 'ml.p3.16xlarge', 'ml.p3dn.24xlarge', 'ml.p4d.24xlarge', 'local_gpu'
    instance_count = 1
    do_spot_training = False
    max_wait = None
    max_run = 1*60*60
    image_uri = None
    distribution = None
    train_job_name = 'sm'
    distribution = {}
    
    if hyperparameters.get('sagemakerdp') and hyperparameters['sagemakerdp']:
        train_job_name = 'smdp-dist'
        distribution["smdistributed"]={ 
                            "dataparallel": {
                                "enabled": True
                            }
                    }
    else:
        distribution["mpi"]={
                            "enabled": True,
        #                     "processes_per_host": 8, # Pick your processes_per_host
        #                     "custom_mpi_options": "-verbose -x orte_base_help_aggregate=0 "
                      }
    
    if do_spot_training:
        max_wait = max_run
    
    image_uri = f'{account}.dkr.ecr.{region}.amazonaws.com/yolov5-training-sagemaker:1.0'

    source_dir = 'yolov5'
    role = get_execution_role()
    sagemaker_session = sage.Session()
    
    s3_data_path = f's3://{bucket}/dataset/{datafolder}'
    checkpoint_s3_uri = f's3://{bucket}/avastus_yolov5/checkpoints'
    
    print('role is {}'.format(role))
    print('image uri is {}'.format(image_uri))
    print('source_dir is ',source_dir)
    print('sagemaker_session is ',sagemaker_session)
    print('s3_data_path is ',s3_data_path)
    print('checkpoint_s3_uri is ',checkpoint_s3_uri)
    
    
    estimator = PyTorch(
        entry_point='train_sm.py',
        source_dir=source_dir,
        role=role,
        sagemaker_session=sagemaker_session,
        framework_version='1.10',
        py_version='py38',
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=1024,
        code_location = code_location,
        output_path=output_path,
        hyperparameters=hyperparameters,
        distribution=distribution,
        disable_profiler=True,
        debugger_hook_config=False,
        metric_definitions=metric_definitions,
        max_run=max_run,
        use_spot_instances=do_spot_training,
        max_wait=max_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
    )
    
    print('Create Training Job')
    response = estimator.fit(
        inputs={'yolov5_input': s3_data_path},
        job_name=job_name,
        wait=False,
    )
    
    # Communicate Back
    message = '{}:{}'.format(datafolder, job_name)
    print('training message : ',message)
    qurl = sqs_client.get_queue_url(
            QueueName='train-lambda-queue')
            
    print('qurl train lambda ',qurl)
    
    response = sqs_client.send_message(
            QueueUrl= qurl["QueueUrl"],
            MessageBody=json.dumps(message)
        )
    
def create_training_job(user_param, job_name, AccountID):

    try:
        print("[INFO]CODEPIPELINE_USER_PARAMETERS:", user_param)

        # Environment variable containing S3 bucket for storing the model artifact
        model_artifact_bucket = os.environ['ModelArtifactBucket']
        print("[INFO]MODEL_ARTIFACT_BUCKET:", model_artifact_bucket)

        # Environment variable containing S3 bucket containing training data
        data_bucket = os.environ['S3DataBucket']
        print("[INFO]TRAINING_DATA_BUCKET:", data_bucket)

    
        ECRRepository = os.environ['ECRRepository']
        container_path = AccountID + '.dkr.ecr.' + region + ".amazonaws.com/" + ECRRepository + ":latest"
        print('[INFO]CONTAINER_PATH:', container_path)
     
 
        # Role to pass to SageMaker training job that has access to training data in S3, etc
        SageMakerRole = os.environ['SageMakerExecutionRole']
        
        train_instance_type = user_param['traincompute']
        train_volume_size = user_param['traininstancevolumesize']
        train_instance_count = user_param['traininstancecount']
        print('[INFO]TRAIN_INSTANCE_TYPE:', train_instance_type)
        print('[INFO]TRAIN_VOLUME_SIZE:', train_volume_size)
        print('[INFO]TRAIN_INSTANCE_COUNT:', train_instance_count)

        

        create_training_params = \
        {
            "RoleArn": SageMakerRole,
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": container_path,
                "TrainingInputMode": "File"
         },
            "ResourceConfig": {
                "InstanceCount": train_instance_count,
                "InstanceType": train_instance_type,
                "VolumeSizeInGB": train_volume_size
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://{}/train".format(data_bucket),
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "csv",
                    "CompressionType": "None"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": "s3://{}/{}/output".format(model_artifact_bucket, job_name)
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 60 * 60
            }
        }    
        
    
        response = sagemaker.create_training_job(**create_training_params)

    except Exception as e:
        print(str(e))
        raise(e)
        
def write_job_info_s3(event):
    print(event)

    objectKey = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['objectKey']
    bucketname = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['bucketName']
    artifactCredentials = event['CodePipeline.job']['data']['artifactCredentials']
    artifactName = event['CodePipeline.job']['data']['outputArtifacts'][0]['name']
    
    # S3 Managed Key for Encryption
    S3SSEKey = os.environ['SSEKMSKeyIdIn']

    json_data = json.dumps(event)
    print(json_data)

    session = Session(aws_access_key_id=artifactCredentials['accessKeyId'],
                  aws_secret_access_key=artifactCredentials['secretAccessKey'],
                  aws_session_token=artifactCredentials['sessionToken'])
   

    s3 = session.resource("s3")
    object = s3.Object(bucketname, objectKey)
    print(object)
    object.put(Body=json_data, ServerSideEncryption='aws:kms', SSEKMSKeyId=S3SSEKey)
    
    print('[SUCCESS]Job Information Written to S3')

def put_job_success(event, train_start_calc):
    
    train_end = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    train_end_calc = datetime.datetime.now()
    print('[INFO]TRAIN_END_SUCCESS:', train_end)
    total_train_time = train_end_calc - train_start_calc
    print('[INFO]TOTAL_TRAIN_TIME:', total_train_time)
    print(event['message'])
    code_pipeline.put_job_success_result(jobId=event['CodePipeline.job']['id'])

def put_job_failure(event):
   
    print('[FAILURE]Putting job failure')
    print(event['message'])
    code_pipeline.put_job_failure_result(jobId=event['CodePipeline.job']['id'], failureDetails={'message': event['message'], 'type': 'JobFailed'})
    return event
