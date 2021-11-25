import sys
import os
import argparse

import boto3

import panorama_test_utility

# ---

# FIXME : make these parameters configurable

AWS_REGION = "us-east-1"
ML_MODEL_FNAME = "ssd_512_resnet50_v1_voc"
S3_BUCKET = "shimomut-panorama-test-us-east-1"
APP_NAME = "people_counter_app"
source_filepath = "./people_counter_app/packages/123456789012-PEOPLE_COUNTER_CODE-1.0/src/app.py"

c = panorama_test_utility.Config(

    # application name
    app_name = 'people_counter_app',

    ## package names and node names
    code_package_name = 'PEOPLE_COUNTER_CODE',
    model_package_name = 'SSD_MODEL',
    camera_node_name = 'abstract_rtsp_media_source',

    # models (model node name : compiled model path without platform dependent suffic)
    models = {
        "model_node" : "./models/" + ML_MODEL_FNAME,
    },

    # video file path to simulate camera stream
    videoname = '../common/test_utility/videos/TownCentreXVID.avi',

    # AWS account ID
    account_id = boto3.client("sts").get_caller_identity()["Account"],
)

# ---

def compile_model_as_needed( model_name ):

    raw_model_file = f"./models/{model_name}.tar.gz"
    compiled_model_file = f"./models/{model_name}-LINUX_X86_64.tar.gz" # FIXME : suffix

    need_model_compilation = False

    if not os.path.exists(raw_model_file):
        print( "Error : Raw model file doesn't exist" )
        return

    elif not os.path.exists(compiled_model_file):
        print( "Compiled model file doesn't exist. Compiling." )
        need_model_compilation = True

    else:
        raw_model_stat = os.stat( raw_model_file )
        compiled_model_stat = os.stat( compiled_model_file )
    
        if raw_model_stat.st_mtime > compiled_model_stat.st_mtime:
            print( "Raw model file has newer timestamp than compiled model file. Recompiling." )
            need_model_compilation = True
        else:
            print( "Compiled model file is up to date." )

    if need_model_compilation:

        # Upload the model to S3, compile it with SageMaker, download the result, and extract it
        panorama_test_utility.prepare_model_for_test(
            region = AWS_REGION,
            data_shape = '{"data":[1,3,512,512]}', # FIXME : parameterize
            framework = 'MXNET',                   # FIXME : parameterize
            local_model_filepath = f"./models/{model_name}.tar.gz",
            s3_model_location = f"s3://{S3_BUCKET}/{APP_NAME}/",
            compile_job_role = None,
        )

def run_simulation():

    name = os.path.basename(source_filepath)

    with open( source_filepath ) as fd:
        file_image = fd.read()

    namespace = {}
    code = compile( file_image, name, 'exec' )
    exec( code, namespace, namespace )
    

def main():
    
    panorama_test_utility.configure(c)
    
    # FIXME : support multiple models
    compile_model_as_needed()

    run_simulation()

main()
