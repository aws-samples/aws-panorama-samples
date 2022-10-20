import sys
import os
import shutil
import argparse
import subprocess


argparser = argparse.ArgumentParser( description = 'Panorama sideloading installation script' )
argparser.add_argument('--expiry', action='store', type=int, default=30, help='Expiry of certificates in days (default: 30 days)')
argparser.add_argument('--app-src-dir', dest="app_src_dir", action='store', required=True, help="Path to Panorama application's src directory")
args = argparser.parse_args()


def copy_file_and_permission( src, dst_dir ):
    
    if not os.path.isdir(dst_dir):
        raise ValueError( f"{dst_dir} is not a directory" )

    print( f"Copying {src} to {dst_dir}" )

    shutil.copy( src, dst_dir )

def delete_file( filename ):

    print( f"Deleting {filename}" )
    os.unlink( filename )

def install_certs_keys():

    # Generate certs and keys
    subprocess.run( [ "openssl", "req", "-x509", "-new", "-days", str(args.expiry), "-nodes", "-out", "sideloading_server.cert.pem", "-keyout", "sideloading_server.key.pem", "-subj", "/CN=pan-sideloading-server" ] )
    subprocess.run( [ "openssl", "req", "-x509", "-new", "-days", str(args.expiry), "-nodes", "-out", "sideloading_client.cert.pem", "-keyout", "sideloading_client.key.pem", "-subj", "/CN=pan-sideloading-client" ] )

    # Install cert and key to server side (Panorama application side)
    copy_file_and_permission( "sideloading_server.cert.pem", args.app_src_dir )
    copy_file_and_permission( "sideloading_server.key.pem",  args.app_src_dir )
    copy_file_and_permission( "sideloading_client.cert.pem", args.app_src_dir )

    # Delete unnecessary files (Other pem files are needed by CLI)
    delete_file( "sideloading_server.key.pem" )

def install_agent_script():

    agent_script_path = os.path.join( os.path.dirname(__file__), "sideloading_agent.py" )
    copy_file_and_permission( agent_script_path, args.app_src_dir )

install_certs_keys()
install_agent_script()


