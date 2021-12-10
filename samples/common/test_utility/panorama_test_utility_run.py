import sys
import os
import datetime
import argparse

import boto3

import panorama_test_utility

# ---

argparser = argparse.ArgumentParser( description='Panorama Test-Utility' )
argparser.add_argument('--region', dest='region', action='store', default=None, help='Region name such as us-east-1')
argparser.add_argument('--app-name', dest='app_name', action='store', required=True, help='Application name')
argparser.add_argument('--code-package-name', dest='code_package_name', required=True, action='store', help='Code package name')
argparser.add_argument('--model-package-name', dest='model_package_name', required=True, action='store', help='Model package name')
argparser.add_argument('--camera-node-name', dest='camera_node_name', required=True, action='store', help='Camera node name')
argparser.add_argument('--model-node-name', dest='model_node_names', action='append', required=True, help='Model node name')
argparser.add_argument('--model-file-basename', dest='model_file_basenames', action='append', required=True, help='Model filename excluding .tar.gz part')
argparser.add_argument('--video-file', dest='video_file', action='store', required=True, help='Video filename to simulate camera stream')
argparser.add_argument('--video-start', dest='video_start', action='store', default=0, help='Video start frame (default: 0)')
argparser.add_argument('--video-stop', dest='video_stop', action='store', default=30, help='Video stop frame (default: 30)')
argparser.add_argument('--video-step', dest='video_step', action='store', default=1, help='Video frame step (default: 1)')
argparser.add_argument('--screenshot-dir', dest='screenshot_dir', action='store', default=None, help="Directory name to save screenshot files. You can use Python's datetime format.")
argparser.add_argument('--py-file', dest='py_file', action='store', required=True, help='Python source path to execute')
args = argparser.parse_args()

if len(args.model_node_names) != len(args.model_file_basenames):
    print( "Error: number of arguments have to be consistent between --model-node-name and --model-file-basenames" )
    sys.exit(1)

# ---

model_node_and_file = {}
for model_node_name, model_file_basename in zip( args.model_node_names, args.model_file_basenames):
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

def run_simulation():

    name = os.path.basename(args.py_file)

    with open( args.py_file ) as fd:
        file_image = fd.read()

    try:
        namespace = {}
        code = compile( file_image, name, 'exec' )
        exec( code, namespace, namespace )
    except panorama_test_utility.panoramasdk.TestUtilityEndOfVideo:
        print( "Reached end of video. Stopped simulation." )

def main():
    
    panorama_test_utility.configure(c)
    
    run_simulation()

main()
