import sys
import os
import re
import time
import json
import tarfile
import platform
import urllib
import subprocess

import boto3

import panoramasdk

panorama_client = boto3.client('panorama') # FIXME : pass from sample notebook


# ---

class Config:
    
    def __init__( self, **args ):
        
        # By default, render output image on Jupyter notebook. This can be turned off in non-Jupyter environments.
        self.render_output_image_with_pyplot = True
        self.screenshot_dir = None
        
        self.video_range = range(0,30,1)

        # FIXME : Should set default values for other parameters as well

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
            

def preview_text_file( filename ):
    print( filename + ":" )
    print( "---" )
    try:
        result = subprocess.run( ["pygmentize", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE )
        if result.stdout : print( result.stdout.decode("utf-8") )
        if result.stderr : print( result.stderr.decode("utf-8") )
    except FileNotFoundError:
        with open(filename) as fd:
            print( fd.read() )


def split_s3_path( s3_path ):
    re_pattern_s3_path = "s3://([^/]+)/(.*)"
    re_result = re.match( re_pattern_s3_path, s3_path )
    bucket = re_result.group(1)
    key = re_result.group(2)
    return bucket, key


def extract_targz( targz_filename, dst_dirname ):
    with tarfile.open( targz_filename, "r" ) as tar_fd:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_fd, path=dst_dirname)


def resolve_sm_role():
    my_session = boto3.session.Session()
    my_region = my_session.region_name
    iam_client = boto3.client('iam', region_name=my_region)
    response_roles = iam_client.list_roles(
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
                            "s3.amazonaws.com",
                            "cloudwatch.amazonaws.com"]},
                    "Action": "sts:AssumeRole"}]}

        rolename = 'AWSPanoramaSMRole' + \
            time.strftime('%Y-%m-%d %H:%M:%S').replace('-', '').replace(" ", "").replace(':', "")

        role = iam_client.create_role(
            RoleName=rolename,
            AssumeRolePolicyDocument=json.dumps(role_policy_document),
        )

        iam_client.attach_role_policy(
            RoleName=rolename,
            PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess',
        )
        iam_client.attach_role_policy(
            RoleName=rolename,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        iam_client.attach_role_policy(
            RoleName=rolename,
            PolicyArn="arn:aws:iam::aws:policy/CloudWatchFullAccess"
        )
        return role['Role']['Arn']

    except Exception as e:
        for role in response_roles['Roles']:
            if role['RoleName'].startswith('AWSPanoramaSMRole'):
                print('Resolved SageMaker IAM Role to: ' + str(role))
                return role['Arn']


def default_app_role( app_name ):
    my_session = boto3.session.Session()
    my_region = my_session.region_name
    iam_client = boto3.client('iam', region_name=my_region)
    response_roles = iam_client.list_roles(
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

        rolename = 'AWSPanoramaSamplesDeploymentRoleTest_{}'.format(app_name)

        role = iam_client.create_role(
            RoleName=rolename,
            AssumeRolePolicyDocument=json.dumps(role_policy_document),
        )

        return role['Role']['Arn']

    except Exception as e:
        for role in response_roles['Roles']:
            if role['RoleName'].startswith('AWSPanoramaSamplesDeploymentRoleTest_{}'.format(app_name)):
                print('Resolved App IAM Role to: ' + str(role))
                return role['Arn']


def _http_download( url, dst ):
    with urllib.request.urlopen( url ) as fd:
        data = fd.read()
    with open( dst, "wb" ) as fd:
        fd.write(data)


def download_sample_model( model_name, dst_dirname ):

    src_url = f"https://panorama-starter-kit.s3.amazonaws.com/public/v2/Models/{model_name}.tar.gz"
    dst_path = f"{dst_dirname}/{model_name}.tar.gz"

    os.makedirs( dst_dirname, exist_ok=True )

    print( "Downloading", src_url )

    _http_download( src_url, dst_path )

    print( "Downloaded to", dst_path )
    
    
def download_artifacts_gpu_sample(sample, account_id):
    if sample.upper() == 'ONNX':
        print('Downloading Source Code')
        os.system("wget -P ./onnx_37_app/packages/{}-onnx_37_app-1.0/ https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/ONNX_Sample/src.zip".format(str(account_id)))
        os.system("unzip -o ./onnx_37_app/packages/{}-onnx_37_app-1.0/src.zip -d ./onnx_37_app/packages/{}-onnx_37_app-1.0".format(str(account_id), str(account_id)))
        os.system("rm ./onnx_37_app/packages/{}-onnx_37_app-1.0/src.zip".format(str(account_id)))
        
        print('Downloading Dependencies')
        os.system("wget -P . https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/ONNX_Sample/dependencies.zip")
        os.system("unzip -o dependencies.zip -d . ")
        os.system("rm dependencies.zip")
        
    elif sample.upper() == 'PYTORCH':
        print('Downloading Source Code')
        os.system("wget -P ./yolov5s_37_2_app/packages/{}-yolov5s_37_2_app-1.0/ https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/PT_Sample/src.zip".format(str(account_id)))
        os.system("unzip -o ./yolov5s_37_2_app/packages/{}-yolov5s_37_2_app-1.0/src.zip -d ./yolov5s_37_2_app/packages/{}-yolov5s_37_2_app-1.0".format(str(account_id), str(account_id)))
        os.system("rm ./yolov5s_37_2_app/packages/{}-yolov5s_37_2_app-1.0/src.zip".format(str(account_id)))
        
        print('Downloading Dependencies')
        os.system("wget -P . https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/PT_Sample/dependencies.zip")
        os.system("unzip -o dependencies.zip -d . ")
        os.system("rm dependencies.zip")
        
    elif sample.upper() == 'TENSORFLOW':
        print('Downloading Source Code')
        os.system("wget -P ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0/ https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/TF_Sample/src.zip".format(str(account_id)))
        os.system("unzip -o ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0/src.zip -d ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0".format(str(account_id), str(account_id)))
        os.system("rm ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0/src.zip".format(str(account_id)))
        
        print('Downloading Model')
        os.system("wget -P ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0/ https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/TF_Sample/saved_model_trt_fp16.zip".format(str(account_id)))
        os.system("unzip -o ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0/saved_model_trt_fp16.zip -d ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0".format(str(account_id), str(account_id)))
        os.system("rm ./tf2_4_trt_app/packages/{}-tf2_4_trt_app-1.0/saved_model_trt_fp16.zip".format(str(account_id)))
        
        print('Downloading Dependencies')
        os.system("wget -P . https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/TF_Sample/dependencies.zip")
        os.system("unzip -o dependencies.zip -d . ")
        os.system("rm dependencies.zip")

    elif sample.upper() == 'TENSORRT':
        print('Downloading Source Code')
        os.system("wget -P ./trtpt_36_2_app/packages/{}-trtpt_36_2_app-1.0/ https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/TRT_Sample/src.zip".format(str(account_id)))
        os.system("unzip -o ./trtpt_36_2_app/packages/{}-trtpt_36_2_app-1.0/src.zip -d ./trtpt_36_2_app/packages/{}-trtpt_36_2_app-1.0".format(str(account_id), str(account_id)))
        os.system("rm ./trtpt_36_2_app/packages/{}-trtpt_36_2_app-1.0/src.zip".format(str(account_id)))
        
        print('Downloading Dependencies')
        os.system("wget -P . https://panorama-starter-kit.s3.amazonaws.com/public/v2/opengpusamples/TRT_Sample/dependencies.zip")
        os.system("unzip -o dependencies.zip -d . ")
        os.system("rm dependencies.zip")
        
        


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
    input_model_filepath,
    output_model_dir,
    s3_model_location,
    compile_job_role ):

    s3_client = boto3.client('s3')

    input_model_dirname, model_filename = os.path.split(input_model_filepath)
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
        input_model_filepath,
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

    print( f"Downloading compiled model to [{output_model_dir}/{compiled_model_filename}] ..." )

    os.makedirs( output_model_dir, exist_ok=True )

    s3_client.download_file(
        s3_bucket, f"{s3_prefix}/{compiled_model_filename}",
        f"{output_model_dir}/{compiled_model_filename}"
    )

    # ---

    print( f"Extracting compiled model in [{output_model_dir}/{model_name}-{_c.compiled_model_suffix}] ..." )

    extract_targz(
        f"{output_model_dir}/{compiled_model_filename}", 
        f"{output_model_dir}/{model_name}-{_c.compiled_model_suffix}"
    )

    print( "Done." )


def create_app(name, description, manifest, role, device):
    if role is None:
        role = default_app_role( name )
    return panorama_client.create_application_instance(
        Name=name,
        Description=description,
        ManifestPayload={
            'PayloadData': manifest
        },
        RuntimeRoleArn=role,
        DefaultRuntimeContextDevice=device
    )


def list_app_instances( device_id = None ):
    
    apps = []
    
    next_token = None
    
    while True:
        
        params = {
            "MaxResults" : 25,
        }

        if device_id:
            params["DeviceId"] = device_id
        
        if next_token:
            params["NextToken"] = next_token
    
        response = panorama_client.list_application_instances( **params )
        
        apps += response["ApplicationInstances"]

        if "NextToken" in response:
            next_token = response["NextToken"]
            continue
        
        break
    
    return apps


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
        response = panorama_client.describe_application_instance( ApplicationInstanceId = app_id )
        status = response['Status']
        progress_dots.update_status( f'{status} ({response["StatusDescription"]})' )
        if status in ['DEPLOYMENT_SUCCEEDED','DEPLOYMENT_FAILED']:
            break
        time.sleep(60)
    
    return response


def remove_application( device_id, application_instance_id ):
    """Remove app"""

    response = panorama_client.remove_application_instance( ApplicationInstanceId = application_instance_id )
    assert response['ResponseMetadata']['HTTPStatusCode'] == 200

    progress_dots = ProgressDots()
    while True:
        response = panorama_client.describe_application_instance( ApplicationInstanceId = application_instance_id )
        status = response['Status']
        progress_dots.update_status( f'{status} ({response["StatusDescription"]})' )
        if status in ('REMOVAL_SUCCEEDED', 'REMOVAL_FAILED'):
            break
        time.sleep(60)

    return response


def update_package_descriptor( app_name, account_id, code_package_name, name_of_py_file ):

    # update Descriptor
    descriptor_path = "./{}/packages/{}-{}-1.0/descriptor.json".format(app_name, account_id, code_package_name)

    with open(descriptor_path, "r") as jsonFile:
        data = json.load(jsonFile)

    data["runtimeDescriptor"]["entry"]["name"] = "/panorama/" + name_of_py_file

    with open(descriptor_path, "w") as jsonFile:
        json.dump(data, jsonFile)


# get CloudWatch Logs URL to see application logs
def get_logs_url( region_name, device_id, application_instance_id ):
    log_group = f"/aws/panorama/devices/{device_id}/applications/{application_instance_id}"
    encoded_log_group = log_group.replace( "/", "$252F" )
    return f"https://console.aws.amazon.com/cloudwatch/home?region={region_name}#logsV2:log-groups/log-group/{encoded_log_group}"

