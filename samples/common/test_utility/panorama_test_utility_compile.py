import sys
import os
import argparse

import panorama_test_utility

# ---

# Enclose all variables locally in this function, to avoid contaminating Jupyter notebook namespace when %run is used.
def test_utility_compile_main():

    argparser = argparse.ArgumentParser( description='Panorama Test-Utility compile command' )
    argparser.add_argument('--region', dest='region', action='store', default=None, help='Region name such as us-east-1')
    argparser.add_argument('--s3-model-location', dest='s3_model_location', action='store', required=True, help='S3 location for model compilation. e.g. s3://mybucket/myapp/')
    argparser.add_argument('--model-node-name', dest='model_node_names', action='append', required=True, help='Model node name')
    argparser.add_argument('--model-file-basename', dest='model_file_basenames', action='append', required=True, help='Model filename excluding .tar.gz part')
    argparser.add_argument('--model-data-shape', dest='model_data_shapes', action='append', required=True, help='Model input data shape. e.g. {"data":[1,3,512,512]}')
    argparser.add_argument('--model-framework', dest='model_frameworks', action='append', required=True, help='Model framework name. e.g. MXNET')
    args = argparser.parse_args()

    if len(args.model_node_names) != len(args.model_file_basenames) or len(args.model_node_names) != len(args.model_data_shapes) or len(args.model_node_names) != len(args.model_frameworks):
        print( "Error: number of arguments have to be consistent between --model-node-name, --model-file-basenames, --model-data-shape, and --model-framework" )
        sys.exit(1)

    # slash character at the end of S3 prefix is optional.
    args.s3_model_location = args.s3_model_location.rstrip("/")

    # ---

    c = panorama_test_utility.Config()

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

    panorama_test_utility.configure(c)

    # TODO : parallelize
    for model_node_name, model_file_basename, model_data_shape, model_framework in zip( args.model_node_names, args.model_file_basenames, args.model_data_shapes, args.model_frameworks ):
        compile_model_as_needed( model_node_name, model_file_basename, model_data_shape, model_framework )

test_utility_compile_main()

