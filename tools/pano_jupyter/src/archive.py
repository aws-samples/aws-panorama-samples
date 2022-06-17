import sys
import os
import glob
import re
import argparse
import shutil

argparser = argparse.ArgumentParser( description='PanoJupyter archive script' )
argparser.add_argument('application_name', help='Application name to create archive ( pano_jupyter_tf, ... )')
args = argparser.parse_args()

assert args.application_name in ("pano_jupyter_tf",)

def remove_file_by_pattern( pattern ):
    
    filenames = glob.glob( pattern )
    assert len(filenames)<=1
    
    if len(filenames)>0:
    
        print( f"Removing : {filenames[0]}" )
        os.unlink( filenames[0] )
    

def replace_account_id_with_placeholder( filename ):

    print( f"Replacing account ids in : {filename}" )
    
    with open(filename, "r") as fd:
        d = fd.read()
        d = re.sub( r"[0-9]{12}\:\:", "123456789012::", d )
    
    with open(filename, "w") as fd:
        fd.write(d)
    

def rename_directory_by_pattern( src_pattern, dst ):
    dirnames = glob.glob( src_pattern )
    assert len(dirnames)==1
    
    src = os.path.normpath(dirnames[0])
    dst = os.path.normpath(dst)

    if src == dst: return

    print( "Renaming : %s -> %s" % (dirnames[0], dst) )
    os.rename( dirnames[0], dst )


def create_zipfile( archive_filename, dir_name ):

    print( f"Creating archive file : {dir_name} -> {archive_filename}" )

    shutil.make_archive( archive_filename, 'zip', ".", dir_name )


if args.application_name == "pano_jupyter_tf":

    remove_file_by_pattern( f"{args.application_name}/packages/*-{args.application_name}_code-1.0/tensorflow-2.4.4-cp37-cp37m-linux_aarch64.whl" )

    replace_account_id_with_placeholder( f"{args.application_name}/graphs/{args.application_name}/graph.json" )
    replace_account_id_with_placeholder( f"{args.application_name}/graphs/{args.application_name}/override.json" )

    rename_directory_by_pattern( f"{args.application_name}/packages/*-{args.application_name}_code-1.0", f"{args.application_name}/packages/123456789012-{args.application_name}_code-1.0" )
    
    create_zipfile( f"{args.application_name}", f"{args.application_name}" )
