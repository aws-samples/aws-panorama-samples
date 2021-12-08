import sys
import os
import datetime
import argparse

import boto3

import panorama_test_utility
import panoramasdk

# ---

argparser = argparse.ArgumentParser( description='Panorama Test-Utility' )
argparser.add_argument('--region', dest='region', action='store', default=None, help='Region name such as us-east-1')
argparser.add_argument('--app-name', dest='app_name', action='store', required=True, help='Application name')
argparser.add_argument('--code-package-name', dest='code_package_name', required=True, action='store', help='Code package name')
argparser.add_argument('--model-package-name', dest='model_package_name', required=True, action='store', help='Model package name')
argparser.add_argument('--camera-node-name', dest='camera_node_name', required=True, action='store', help='Camera node name')
argparser.add_argument('--s3-model-location', dest='s3_model_location', action='store', required=True, help='S3 location for model compilation. e.g. s3://mybucket/myapp/')
argparser.add_argument('--model-node-name', dest='model_node_names', action='append', required=True, help='Model node name')
argparser.add_argument('--model-file-basename', dest='model_file_basenames', action='append', required=True, help='Model filename excluding .tar.gz part')
argparser.add_argument('--model-data-shape', dest='model_data_shapes', action='append', required=True, help='Model input data shape. e.g. {"data":[1,3,512,512]}')
argparser.add_argument('--model-framework', dest='model_frameworks', action='append', required=True, help='Model framework name. e.g. MXNET')
argparser.add_argument('--video-file', dest='video_file', action='store', required=True, help='Video filename to simulate camera stream')
argparser.add_argument('--video-start', dest='video_start', action='store', default=0, help='Video start frame (default: 0)')
argparser.add_argument('--video-stop', dest='video_stop', action='store', default=30, help='Video stop frame (default: 30)')
argparser.add_argument('--video-step', dest='video_step', action='store', default=1, help='Video frame step (default: 1)')
argparser.add_argument('--screenshot-dir', dest='screenshot_dir', action='store', default=None, help="Directory name to save screenshot files. You can use Python's datetime format.")
argparser.add_argument('--py-file', dest='py_file', action='store', required=True, help='Python source path to execute')
args = argparser.parse_args()

if len(args.model_node_names) != len(args.model_file_basenames) or len(args.model_node_names) != len(args.model_data_shapes) or len(args.model_node_names) != len(args.model_frameworks):
    print( "Error: number of arguments have to be consistent between --model-node-name, --model-file-basenames, --model-data-shape, and --model-framework" )
    sys.exit(1)

# ---

model_node_and_file = {}
for model_node_name, model_file_basename, model_data_shape, model_framework in zip( args.model_node_names, args.model_file_basenames, args.model_data_shapes, args.model_frameworks ):
    model_file_dirname, model_file_basename = os.path.split(model_file_basename)
    compiled_model_basename = f"{model_file_dirname}/{model_node_name}/{model_file_basename}"
    model_node_and_file[model_node_name] = compiled_model_basename

screenshot_dir_dt_resolved = None
if args.screenshot_dir:
    screenshot_dir_dt_resolved = datetime.datetime.now().strftime(args.screenshot_dir)
    os.makedirs( screenshot_dir_dt_resolved, exist_ok=True )

c = panorama_test_utility.Config(

    # application name
    app_name = args.app_name,

    ## package names and node names
    code_package_name = args.code_package_name,
    model_package_name = args.model_package_name,
    camera_node_name = args.camera_node_name,

    # models (model node name : compiled model path without platform dependent suffic)
    models = model_node_and_file,

    # video file path to simulate camera stream
    videoname = args.video_file,
    video_range = range( int(args.video_start), int(args.video_stop), int(args.video_step) ),
    
    # Suppress rendering output by pyplot, and write screenshots in PNG files
    render_output_image_with_pyplot = False,
    screenshot_dir = screenshot_dir_dt_resolved,

    # AWS account ID
    account_id = boto3.client("sts").get_caller_identity()["Account"],
)

# ---

def compile_model_as_needed( model_node_name, model_file_basename, model_data_shape, model_framework ):

    print( f"Checking [{model_node_name}]" )

    model_file_dirname, model_file_basename = os.path.split(model_file_basename)

    raw_model_file = f"{model_file_dirname}/{model_file_basename}.tar.gz"
    compiled_model_dir = f"{model_file_dirname}/{model_node_name}"
    compiled_model_file = f"{compiled_model_dir}/{model_file_basename}-{c.compiled_model_suffix}.tar.gz"

    need_model_compilation = False

    if not os.path.exists(raw_model_file):
        print( f"Error : Raw model file [{raw_model_file}] doesn't exist" )
        return

    elif not os.path.exists(compiled_model_file):
        print( f"Compiled model file [{compiled_model_file}] doesn't exist. Need compilation." )
        need_model_compilation = True

    else:
        raw_model_stat = os.stat( raw_model_file )
        compiled_model_stat = os.stat( compiled_model_file )
    
        if raw_model_stat.st_mtime > compiled_model_stat.st_mtime:
            print( "Raw model file has newer timestamp than compiled model file. Need compilation." )
            need_model_compilation = True
        else:
            print( "Compiled model file is up to date. Skipping compilation." )

    if need_model_compilation:
        panorama_test_utility.prepare_model_for_test(
            region = args.region,
            data_shape = model_data_shape,
            framework = model_framework,
            input_model_filepath = raw_model_file,
            output_model_dir = compiled_model_dir,
            s3_model_location = args.s3_model_location + "/" + model_node_name,
            compile_job_role = panorama_test_utility.resolve_sm_role(),
        )

def run_simulation():

    name = os.path.basename(args.py_file)

    with open( args.py_file ) as fd:
        file_image = fd.read()

    print("---")

    try:
        namespace = {}
        code = compile( file_image, name, 'exec' )
        exec( code, namespace, namespace )
    except panoramasdk.TestUtilityEndOfVideo:
        print( "Reached end of video. Stopped simulation." )

def main():
    
    panorama_test_utility.configure(c)
    
    # TODO : parallelize
    for model_node_name, model_file_basename, model_data_shape, model_framework in zip( args.model_node_names, args.model_file_basenames, args.model_data_shapes, args.model_frameworks ):
        compile_model_as_needed( model_node_name, model_file_basename, model_data_shape, model_framework )

    run_simulation()

main()
