import sys
import os
import json
import urllib
import pprint
import datetime
import fnmatch
import http.client
import ssl
import argparse


python_version_tuple = ( sys.version_info.major, sys.version_info.minor, sys.version_info.micro )
min_python_version_tuple = (3,7,0)
if python_version_tuple < min_python_version_tuple:
    print( "Warning : Python versions older than {min_python_version_tuple[0]}.{min_python_version_tuple[1]} is not supported." )


ignore_patterns = [
    "sideloading_server.cert.pem",
    "sideloading_server.key.pem",
    "sideloading_client.cert.pem",
    "sideloading_client.key.pem",
    "sideloading_agent.py",
]

class SideloadingClient:

    def __init__( self, server_address, port, cert_key_dir="." ):
    
        self.ssl_context = ssl.create_default_context( purpose=ssl.Purpose.SERVER_AUTH )

        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED

        # Server certificate
        self.ssl_context.load_verify_locations( os.path.join( cert_key_dir, "sideloading_server.cert.pem" ) )

        # Client certificate & secret-key
        self.ssl_context.load_cert_chain( 
            os.path.join( cert_key_dir, "sideloading_client.cert.pem" ), 
            os.path.join( cert_key_dir, "sideloading_client.key.pem" ), 
        )

        self.https_conn = http.client.HTTPSConnection( server_address, port, context=self.ssl_context)

    def sendFile( self, src_top, src_filename ):
    
        src_top = os.path.normpath(src_top).replace("\\","/")
    
        src_filename = src_filename.replace("\\","/")

        src_fullpath = os.path.join( src_top, src_filename )
        src_fullpath = os.path.normpath(src_fullpath).replace("\\","/")
        assert src_fullpath.startswith(src_top), f"{src_fullpath} doesn't start with {src_top}"
        
        st = os.stat(src_fullpath)
        dt_mtime = datetime.datetime.fromtimestamp( st.st_mtime )
        
        with open(src_fullpath,"rb") as src_fd:
            d = src_fd.read()
        
        headers = {
            "mtime" : dt_mtime.isoformat()
        }
        
        print( f"Sending {src_filename}" )
        
        self.https_conn.request( 'PUT', '/files/' + urllib.parse.quote(src_filename), body = d, headers=headers )

        response = self.https_conn.getresponse()
        d = json.loads(response.read())
        return d

    def deleteFile( self, src_filename ):
    
        src_filename = src_filename.replace("\\","/")

        print( f"Deleting {src_filename}" )

        self.https_conn.request( 'DELETE', '/files/' + urllib.parse.quote(src_filename) )

        response = self.https_conn.getresponse()
        d = json.loads(response.read())
        return d

    def listFiles( self ):
    
        self.https_conn.request( 'GET', '/files' )

        response = self.https_conn.getresponse()
        d = json.loads(response.read())
        return d

    def sync( self, src_top ):
        
        src_top = os.path.normpath(src_top).replace("\\","/")

        # List all files in destination
        dst_files_list = self.listFiles()

        # List all files in source
        src_files_list = []
        for place, dirs, files in os.walk( src_top ):
            for filename in files:
                
                filepath = os.path.join( place, filename ).replace("\\","/")
                assert filepath.startswith(src_top)
                filepath_relative = filepath[ len(src_top) : ]
                filepath_relative = filepath_relative.lstrip("/\\")
                filepath_relative = filepath_relative.replace("\\","/")

                ignore = False
                for ignore_pattern in ignore_patterns:
                    if fnmatch.fnmatch( filepath_relative, ignore_pattern ):
                        ignore = True
                        break
                if ignore:
                    continue
                
                st = os.stat(filepath)
                dt_mtime = datetime.datetime.fromtimestamp( st.st_mtime )
                
                src_files_list.append( {
                    "filepath" : filepath_relative,
                    "mtime" : dt_mtime.isoformat(),
                    "size" : st.st_size,
                })
        
        src_files_table = {}
        for src_file in src_files_list:
            src_files_table[ src_file["filepath"] ] = src_file

        dst_files_table = {}
        for dst_file in dst_files_list:
            dst_files_table[ dst_file["filepath"] ] = dst_file
        
        # Copy all files            
        for src_file in src_files_list:
            
            filepath = src_file["filepath"]
            
            if filepath in dst_files_table:
                
                dst_file = dst_files_table[filepath]

                datetime_isoformat = "%Y-%m-%dT%H:%M:%S.%f"
                src_mtime = datetime.datetime.strptime( src_file["mtime"], datetime_isoformat ).timestamp()
                dst_mtime = datetime.datetime.strptime( dst_file["mtime"], datetime_isoformat ).timestamp()

                # Skip unchanged file
                if abs(src_mtime - dst_mtime)<1 and src_file["size"]==dst_file["size"]:
                    print( f"Unchanged {filepath}" )
                    continue
                
            # Send changed file
            self.sendFile( src_top, filepath )
                
        # Delete files which disappeard in source
        for dst_file in dst_files_list:
            
            filepath = dst_file["filepath"]
            
            if filepath not in src_files_table:
                self.deleteFile(filepath)
        
        return {}

    def runApplication(self):
    
        self.https_conn.request( 'POST', '/application' )

        response = self.https_conn.getresponse()
        d = json.loads(response.read())
        return d

    def killApplication(self):
    
        self.https_conn.request( 'DELETE', '/application' )

        response = self.https_conn.getresponse()
        d = json.loads(response.read())
        return d


# ---


def send_file( argv ):

    argparser = argparse.ArgumentParser( description='Send single file' )
    argparser.add_argument('--addr', dest='addr', action='store', required=True, help='IP address or hostname of Panorama device')
    argparser.add_argument('--port', dest='port', action='store', default=8123, help='Port number (default:8123)')
    argparser.add_argument('--cert-key-dir', dest='cert_key_dir', action='store', default=".", help='Directory where PEM files are stored (default: current directory)')
    argparser.add_argument('--src-top', dest='src_top', action='store', type=str, required=True, help='Top directory (where main.py / main.sh exists) in development host.')
    argparser.add_argument('filepath', action='store', type=str, help='Relative filepath from src-top directory')
    args = argparser.parse_args( argv )
    
    sideloading_client = SideloadingClient(
        server_address = args.addr,
        port = args.port,
        cert_key_dir = args.cert_key_dir,
    )

    sideloading_client.sendFile( src_top = args.src_top, src_filename = args.filepath )
    

def delete_file( argv ):

    argparser = argparse.ArgumentParser( description='Delete single file' )
    argparser.add_argument('--addr', dest='addr', action='store', required=True, help='IP address or hostname of Panorama device')
    argparser.add_argument('--port', dest='port', action='store', default=8123, help='Port number (default:8123)')
    argparser.add_argument('--cert-key-dir', dest='cert_key_dir', action='store', default=".", help='Directory where PEM files are stored (default: current directory)')
    argparser.add_argument('filepath', action='store', type=str, help='Relative filepath ')
    args = argparser.parse_args( argv )
    
    sideloading_client = SideloadingClient(
        server_address = args.addr,
        port = args.port,
        cert_key_dir = args.cert_key_dir,
    )

    sideloading_client.deleteFile( args.filepath )
    

def list_files( argv ):

    argparser = argparse.ArgumentParser( description='List all sideloaded files' )
    argparser.add_argument('--addr', dest='addr', action='store', required=True, help='IP address or hostname of Panorama device')
    argparser.add_argument('--port', dest='port', action='store', default=8123, help='Port number (default:8123)')
    argparser.add_argument('--cert-key-dir', dest='cert_key_dir', action='store', default=".", help='Directory where PEM files are stored (default: current directory)')
    args = argparser.parse_args( argv )
    
    sideloading_client = SideloadingClient(
        server_address = args.addr,
        port = args.port,
        cert_key_dir = args.cert_key_dir,
    )

    files = sideloading_client.listFiles()
    
    max_filepath_length = 10
    max_mtime_length = 10
    for f in files:
        max_filepath_length = max( len(f["filepath"]), max_filepath_length )
        max_mtime_length = max( len(f["mtime"]), max_mtime_length )

    format = f"%{max_filepath_length}s  %-{max_mtime_length}s  %s"
    print( format % ("filepath", "mtime", "size") )
    for f in files:
        print( format % (f["filepath"], f["mtime"], f["size"]) )
    

def sync( argv ):

    argparser = argparse.ArgumentParser( description='List all sideloaded files' )
    argparser.add_argument('--addr', dest='addr', action='store', required=True, help='IP address or hostname of Panorama device')
    argparser.add_argument('--port', dest='port', action='store', default=8123, help='Port number (default:8123)')
    argparser.add_argument('--cert-key-dir', dest='cert_key_dir', action='store', default=".", help='Directory where PEM files are stored (default: current directory)')
    argparser.add_argument('--src-top', dest='src_top', action='store', type=str, required=True, help='Top directory (where main.py / main.sh exists) in development host.')
    args = argparser.parse_args( argv )
    
    sideloading_client = SideloadingClient(
        server_address = args.addr,
        port = args.port,
        cert_key_dir = args.cert_key_dir,
    )

    sideloading_client.sync( src_top = args.src_top )
    
    
def run_app( argv ):

    argparser = argparse.ArgumentParser( description='List all sideloaded files' )
    argparser.add_argument('--addr', dest='addr', action='store', required=True, help='IP address or hostname of Panorama device')
    argparser.add_argument('--port', dest='port', action='store', default=8123, help='Port number (default:8123)')
    argparser.add_argument('--cert-key-dir', dest='cert_key_dir', action='store', default=".", help='Directory where PEM files are stored (default: current directory)')
    args = argparser.parse_args( argv )
    
    sideloading_client = SideloadingClient(
        server_address = args.addr,
        port = args.port,
        cert_key_dir = args.cert_key_dir,
    )

    sideloading_client.runApplication()
    

def kill_app( argv ):

    argparser = argparse.ArgumentParser( description='List all sideloaded files' )
    argparser.add_argument('--addr', dest='addr', action='store', required=True, help='IP address or hostname of Panorama device')
    argparser.add_argument('--port', dest='port', action='store', default=8123, help='Port number (default:8123)')
    argparser.add_argument('--cert-key-dir', dest='cert_key_dir', action='store', default=".", help='Directory where PEM files are stored (default: current directory)')
    args = argparser.parse_args( argv )
    
    sideloading_client = SideloadingClient(
        server_address = args.addr,
        port = args.port,
        cert_key_dir = args.cert_key_dir,
    )

    sideloading_client.killApplication()
    

command_table = {
    "send-file" : send_file,
    "delete-file" : delete_file,
    "list-files" : list_files,
    "sync" : sync,
    "run-app" : run_app,
    "kill-app" : kill_app,
}


argparser = argparse.ArgumentParser( description = 'Panorama sideloading commands' )
command_names = ", ".join( command_table.keys() ).strip(" ,") 
argparser.add_argument('command', help=f'Command to run ( {command_names} )')
args = argparser.parse_args( sys.argv[1:2] )

command_func = command_table[args.command]
command_func( sys.argv[2:] )

