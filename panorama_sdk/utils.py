import sys
import os
import time
import json
import shutil
import tarfile
import platform

import boto3
from botocore.exceptions import ClientError
import sagemaker

import panoramasdk

s3 = boto3.resource('s3')
client = boto3.client('panorama') # FIXME : should rename client -> panorama or panorama_client

# ---

class Config:
    def __init__( self, **args ):
        self.__dict__.update(args)
        self.test_utility_dirname, _ = os.path.split(__file__)

_c = Config()

# ---


def configure( config ):
    
    global _c
    _c = config
        
    panoramasdk._configure(config)


def get_platform_config():
    
    class PlatformConfig:
        pass
    
    platform_config = PlatformConfig()
    platform_config.compiled_model_suffix = None
    platform_config.neo_target_device = None
    platform_config.neo_target_platform = None

    os_name = platform.system()
    processor = platform.processor()
    
    if os_name == "Linux":
        if processor == "aarch64":
            platform_config.compiled_model_suffix = "LINUX_ARM64"
            platform_config.neo_target_platform = { 'Os': 'LINUX', 'Arch': 'ARM64' }
        elif processor == "x86_64":
            platform_config.compiled_model_suffix = "LINUX_X86_64"
            platform_config.neo_target_platform = { 'Os': 'LINUX', 'Arch': 'X86_64' }
        else:
            assert False, f"Processor type {processor} not supported"
    elif os_name == "Darwin":
        assert False, "MacOS is not yet supported"
        platform_config.compiled_model_suffix = "COREML"
        platform_config.neo_target_device = "coreml"
    else:
        assert False, f"OS {os_name} not supported"

    return platform_config

    
def extract_targz( targz_filename, dst_dirname ):
    with tarfile.open( targz_filename, "r" ) as tar_fd:
        tar_fd.extractall( path = dst_dirname )


def resolve_sm_role():
    my_session = boto3.session.Session()
    my_region = my_session.region_name
    client = boto3.client('iam', region_name=my_region)
    response_roles = client.list_roles(
        PathPrefix='/',
        # Marker='string',
        MaxItems=999
    )

    try:
        role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": [
                            "sagemaker.amazonaws.com",
                            "s3.amazonaws.com"]},
                    "Action": "sts:AssumeRole"}]}

        rolename = 'AWSPanoramaSMRole' + \
            time.strftime('%Y-%m-%d %H:%M:%S').replace('-', '').replace(" ", "").replace(':', "")

        role = client.create_role(
            RoleName=rolename,
            AssumeRolePolicyDocument=json.dumps(role_policy_document),
        )

        client.attach_role_policy(
            RoleName=rolename,
            PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess',
        )
        client.attach_role_policy(
            RoleName=rolename,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        return role['Role']['Arn']

    except Exception as e:
        for role in response_roles['Roles']:
            if role['RoleName'].startswith('AWSPanoramaSMRole'):
                print('Resolved SageMaker IAM Role to: ' + str(role))
                return role['Arn']


def default_app_role():
    my_session = boto3.session.Session()
    my_region = my_session.region_name
    client = boto3.client('iam', region_name=my_region)
    response_roles = client.list_roles(
        PathPrefix='/',
        # Marker='string',
        MaxItems=999
    )

    try:
        role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": ["panorama.amazonaws.com"]},
                    "Action": "sts:AssumeRole"}]}

        rolename = 'AWSPanoramaSamplesDeploymentRoleTest_{}'.format(_c.app_name)

        role = client.create_role(
            RoleName=rolename,
            AssumeRolePolicyDocument=json.dumps(role_policy_document),
        )

        return role['Role']['Arn']

    except Exception as e:
        for role in response_roles['Roles']:
            if role['RoleName'].startswith('AWSPanoramaSamplesDeploymentRoleTest_{}'.format(_c.app_name)):
                print('Resolved App IAM Role to: ' + str(role))
                return role['Arn']


def download_model( model ):
    url = "https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/" + model + '.tar.gz'
    local_file_path = os.getcwd() + "/" + model + ".tar.gz"
    os.system( f"wget {url}" )
    os.system( f"mv {local_file_path} {_c.test_utility_dirname}/models/" )
    return


def compile_model(
        region,
        s3_model_location,
        data_shape,
        framework,
        target_device,
        target_platform,
        s3_output_location,
        role):
    # https://docs.aws.amazon.com/sagemaker/latest/dg/neo-job-compilation-cli.html

    AWS_REGION = region
    s3_input_location = s3_model_location
    data_shape = data_shape
    framework = framework
    s3_output_location = s3_output_location

    #role = sagemaker.get_execution_role()
    role = role

    # Create a SageMaker client so you can submit a compilation job
    sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)

    # Give your compilation job a name
    compilation_job_name = 'Comp-Job' + \
        time.strftime('%Y-%m-%d_%I-%M-%S-%p').replace('_', '').replace('-', '')
    print(f'Compilation job for {compilation_job_name} started')

    params = {
        "CompilationJobName" : compilation_job_name,
        "RoleArn" : role,
        "InputConfig" : {
            'S3Uri': s3_input_location,
            'DataInputConfig': data_shape,
            'Framework': framework.upper()
        },
        "OutputConfig" : {
            'S3OutputLocation': s3_output_location,
        },
        "StoppingCondition" : {
            'MaxRuntimeInSeconds': 7200
        }
    }
    
    if target_device:
        params["OutputConfig"]["TargetDevice"] = target_device
    if target_platform:
        params["OutputConfig"]["TargetPlatform"] = target_platform
    
    response = sagemaker_client.create_compilation_job( **params )

    # Optional - Poll every 30 sec to check completion status

    while True:
        response = sagemaker_client.describe_compilation_job(
            CompilationJobName=compilation_job_name)
        if response['CompilationJobStatus'] == 'COMPLETED':
            break
        elif response['CompilationJobStatus'] == 'FAILED':
            # FIXME : should include more detailed failure reason in exception
            raise RuntimeError('Compilation failed')
        print('Compiling ...')
        time.sleep(30)

    print('Done!')


def create_app(name, description, manifest, role, device):
    if role is None:
        role = default_app_role()
    return client.create_application_instance(
        Name=name,
        Description=description,
        ManifestPayload={
            'PayloadData': manifest
        },
        RuntimeRoleArn=role,
        DefaultRuntimeContextDevice=device
    )


def describe_app(app):
    return client.describe_application_instance(
        ApplicationInstanceId=app)


def remove_app(app):
    return client.remove_application_instance(
        ApplicationInstanceId=app)


def list_app_instances(device):
    
    # FIXME : should use next-token and get all the result
    
    return client.list_application_instances(
        DeviceId=device,
        MaxResults=25)


def deploy_app(device_id, app_name, role):
    """Deploy app"""

    app_description = ''

    with open('./{}/graphs/{}/graph.json'.format(app_name, app_name), 'r') as file_object:
        my_app_manifest_from_file = json.load(file_object)
    my_app_manifest = json.dumps(my_app_manifest_from_file)
    app_created = create_app(
        app_name,
        app_description,
        my_app_manifest,
        role,
        device_id)
    assert app_created['ResponseMetadata']['HTTPStatusCode'] == 200
    my_app_id = app_created['ApplicationInstanceId']
    print(f'App ID: {my_app_id}')
    i=0
    while True:
        print(f'Request: #{i + 1}')
        response = describe_app(my_app_id)
        app_status = response['Status']
        print(f'App status #{i + 1}: {app_status}')
        if app_status in ['DEPLOYMENT_SUCCEEDED','DEPLOYMENT_FAILED']:
            # logger.info(app_status)  # "HealthStatus" : "RUNNING"
            # ,"NOT_AVAILABLE"
            print(app_status)
            break
        i+=1
        time.sleep(150)
    assert app_status == 'DEPLOYMENT_SUCCEEDED'
    
    return response


def remove_application( device_id, application_instance_id ):
    """Remove app"""

    response = remove_app(application_instance_id)
    print(f'Response: {response}')
    remove_status_code = response['ResponseMetadata']['HTTPStatusCode']
    # listApplicationInstances "Status" : "REMOVAL_PENDING","Status" :
    # "REMOVAL_SUCCEEDED",
    if remove_status_code == 200:
        removed = False
        i=0
        while True:
            if not removed:
                print(f'Request: {i + 1}')
                response = list_app_instances(device_id)
                app_instances = response['ApplicationInstances']
                print(f'app_instances: {app_instances}')
                #logger.info(f'app_instances: {app_instances}')
                for app in app_instances:
                    print(f'app: {app}')
                    app_inst = app['ApplicationInstanceId']
                    print(f'app_inst_id: {app_inst}')
                    if app['ApplicationInstanceId'] == application_instance_id:
                        status = app['Status']
                        print(f'Status: {status}')
                        if status == 'REMOVAL_SUCCEEDED':
                            print('Removed')
                            removed = True
                            break
                i+=1
                time.sleep(150)
            else:
                break
        assert removed
    else:
        print('App Not Removed')


def update_descriptor(account_id, code_package_name, name_of_file):

    # update Descriptor
    descriptor_path = "./{}/packages/{}-{}-1.0/descriptor.json".format(_c.app_name, account_id, code_package_name)

    with open(descriptor_path, "r") as jsonFile:
        data = json.load(jsonFile)

    data["runtimeDescriptor"]["entry"]["name"] = "/panorama/" + name_of_file

    with open(descriptor_path, "w") as jsonFile:
        json.dump(data, jsonFile)
