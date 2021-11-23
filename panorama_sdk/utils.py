import sys
import os
import re
import time
import json
import tarfile
import platform
import urllib

import boto3

import panoramasdk

s3 = boto3.resource('s3')
client = boto3.client('panorama') # FIXME : should rename client -> panorama or panorama_client

# ---

class Config:
    
    def __init__( self, **args ):
        
        # Use *this* directory (where this python module exists) as the test-utility directory.
        self.test_utility_dirname, _ = os.path.split(__file__)

        # Set platform dependent parameters such as Neo compile options
        self._set_platform_dependent_parameters()

        self.__dict__.update(args)

    def _set_platform_dependent_parameters(self):

        self.compiled_model_suffix = None
        self.neo_target_device = None
        self.neo_target_platform = None

        os_name = platform.system()
        processor = platform.processor()
        
        if os_name == "Linux":
            if processor == "aarch64":
                self.compiled_model_suffix = "LINUX_ARM64"
                self.neo_target_platform = { 'Os': 'LINUX', 'Arch': 'ARM64' }
            elif processor == "x86_64":
                self.compiled_model_suffix = "LINUX_X86_64"
                self.neo_target_platform = { 'Os': 'LINUX', 'Arch': 'X86_64' }
            else:
                assert False, f"Processor type {processor} not supported"
        elif os_name == "Darwin":
            assert False, "MacOS is not yet supported"
            self.compiled_model_suffix = "COREML"
            self.neo_target_device = "coreml"
        else:
            assert False, f"OS {os_name} not supported"

_c = Config()

# ---

def configure( config ):
    
    global _c
    _c = config
        
    panoramasdk._configure(config)


class ProgressDots:
    def __init__(self):
        self.previous_status = None
    def update_status(self,status):
        if status == self.previous_status:
            print( ".", end="", flush=True )
        else:
            if self.previous_status : print("")
            if status : print( status + " " , end="", flush=True)
            self.previous_status = status
            

def split_s3_path( s3_path ):
    re_pattern_s3_path = "s3://([^/]+)/(.*)"
    re_result = re.match( re_pattern_s3_path, s3_path )
    bucket = re_result.group(1)
    key = re_result.group(2)
    return bucket, key


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


def _http_download( url, dst ):
    with urllib.request.urlopen( url ) as fd:
        data = fd.read()
    with open( dst, "wb" ) as fd:
        fd.write(data)


def download_sample_model( model_name, dst_dirname=None ):

    if dst_dirname is None:
        dst_dirname = f"{_c.test_utility_dirname}/models"

    src_url = f"https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/{model_name}.tar.gz"
    dst_path = f"{dst_dirname}/{model_name}.tar.gz"

    os.makedirs( dst_dirname, exist_ok=True )

    print( "Downloading", src_url )

    _http_download( src_url, dst_path )

    print( "Downloaded to", dst_path )


def compile_model(
        region,
        s3_model_location,
        data_shape,
        framework,
        target_device,
        target_platform,
        s3_output_location,
        compile_job_role):

    # Create a SageMaker client so you can submit a compilation job
    sagemaker_client = boto3.client('sagemaker', region_name=region)

    # Give your compilation job a name
    compilation_job_name = 'Comp-Job' + time.strftime('%Y-%m-%d_%I-%M-%S-%p').replace('_', '').replace('-', '')

    # https://docs.aws.amazon.com/sagemaker/latest/dg/neo-job-compilation-cli.html
    params = {
        "CompilationJobName" : compilation_job_name,
        "RoleArn" : compile_job_role,
        "InputConfig" : {
            'S3Uri': s3_model_location,
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

    print( f'Created compilation job [{compilation_job_name}]' )

    # Poll every 30 sec to check completion status
    progress_dots = ProgressDots()
    while True:
        response = sagemaker_client.describe_compilation_job(
            CompilationJobName=compilation_job_name)

        progress_dots.update_status( "Compilation job status : " + response['CompilationJobStatus'] )

        if response['CompilationJobStatus'] == 'COMPLETED':
            break
        elif response['CompilationJobStatus'] == 'FAILED':
            failure_reason = response["FailureReason"]
            failure_reason = failure_reason.replace( "\\n", "\n" )
            failure_reason = failure_reason.replace( "\\'", "'" )
            raise RuntimeError( 'Model compilation failed \n' + failure_reason )

        time.sleep(30)

    progress_dots.update_status("")


# Upload raw model data to s3, compile it, download the compilation result, and extract it
def prepare_model_for_test(
    region,
    data_shape,
    framework,
    local_model_filepath,
    s3_model_location,
    compile_job_role ):

    s3_client = boto3.client('s3')

    local_model_dirname, model_filename = os.path.split(local_model_filepath)
    s3_bucket, s3_prefix = split_s3_path( s3_model_location )
    s3_prefix = s3_prefix.rstrip("/")

    model_filename_ext = ".tar.gz"
    if model_filename.lower().endswith( model_filename_ext ):
        model_name = model_filename[ : -len(model_filename_ext) ]
    else:
        assert False, f"Model filename has to end with [{model_filename_ext}]"

    compiled_model_filename = f"{model_name}-{_c.compiled_model_suffix}.tar.gz"

    # ---

    print( f"Uploading [{model_filename}] to [{s3_model_location}] ..." )

    s3_client.upload_file(
        f"{local_model_dirname}/{model_filename}",
        s3_bucket, f"{s3_prefix}/{model_filename}"
    )

    # ---

    print( f"Compiling [{model_filename}] ..." )

    compile_model(
        region = region,
        s3_model_location = f"s3://{s3_bucket}/{s3_prefix}/{model_filename}",
        data_shape = data_shape,
        framework = framework,
        target_device = _c.neo_target_device,
        target_platform = _c.neo_target_platform,
        s3_output_location = f"s3://{s3_bucket}/{s3_prefix}/",
        compile_job_role = compile_job_role
    )

    # ---

    print( f"Downloading compiled model to [{local_model_dirname}/{compiled_model_filename}] ..." )

    s3_client.download_file(
        s3_bucket, f"{s3_prefix}/{compiled_model_filename}",
        f"{local_model_dirname}/{compiled_model_filename}"
    )

    # ---

    print( f"Extracting compiled model to [{local_model_dirname}/{model_name}-{_c.compiled_model_suffix}] ..." )

    extract_targz( 
        f"{local_model_dirname}/{compiled_model_filename}", 
        f"{local_model_dirname}/{model_name}-{_c.compiled_model_suffix}"
    )

    print( "Done." )


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
    app_id = app_created['ApplicationInstanceId']
    
    print( "Application Instance Id :", app_id )

    progress_dots = ProgressDots()
    while True:
        response = describe_app(app_id)
        status = response['Status']
        progress_dots.update_status( f'{status} ({response["StatusDescription"]})' )
        if app_status in ['DEPLOYMENT_SUCCEEDED','DEPLOYMENT_FAILED']:
            break
        time.sleep(60)
    
    assert status == 'DEPLOYMENT_SUCCEEDED'
    
    return response


def remove_application( device_id, application_instance_id ):
    """Remove app"""

    response = remove_app(application_instance_id)
    print(f'Response: {response}')
    remove_status_code = response['ResponseMetadata']['HTTPStatusCode']
    # listApplicationInstances "Status" : "REMOVAL_PENDING","Status" :
    # "REMOVAL_SUCCEEDED",
    if remove_status_code == 200:
        i=0
        while True:
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
                        return
            i+=1
            time.sleep(150)
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
