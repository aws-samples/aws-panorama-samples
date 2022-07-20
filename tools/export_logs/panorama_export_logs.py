import sys
import os
import re
import time
import datetime
import argparse
import tempfile
import shutil
import gzip
import boto3
        
argparser = argparse.ArgumentParser( description='Export Panorama device level / application level logs in a Zip file' )
argparser.add_argument('--region', dest='region', action='store', default=None, help='Region name such as us-east-1')
argparser.add_argument('--device-id', dest='device_id', action='store', default=None, help='Panorama device-id')
argparser.add_argument('--app-id', dest='app_id', action='store', default=None, help='Panorama application instance id')
argparser.add_argument('--s3-path', dest='s3_path', action='store', default=None, help='S3 path as a working place')
argparser.add_argument('--start-datetime', dest='start_datetime', action='store', default=None, help='Start date-time in UTC, in YYYYMMDD_HHMMSS format')
argparser.add_argument('--end-datetime', dest='end_datetime', action='store', default=None, help='End date-time in UTC, in YYYYMMDD_HHMMSS format')
args = argparser.parse_args()

if args.device_id is None:
    print( "Missing argument --device-id" )
    argparser.print_help()
    sys.exit(1)

if args.s3_path is None:
    print( "Missing argument --s3-path" )
    argparser.print_help()
    sys.exit(1)

if args.start_datetime is None:
    print( "Missing argument --start-datetime" )
    argparser.print_help()
    sys.exit(1)

if args.end_datetime is None:
    print( "Missing argument --end-datetime" )
    argparser.print_help()
    sys.exit(1)

# ---

def splitS3Path( s3_path ):
    re_pattern_s3_path = "s3://([^/]+)/(.*)"
    re_result = re.match( re_pattern_s3_path, s3_path )
    bucket = re_result.group(1)
    key = re_result.group(2)
    key = key.rstrip("/")
    return bucket, key


def getLogGroup( device_id, application_instance_id=None ):
    if application_instance_id is None:
        log_group = f"/aws/panorama/devices/{device_id}"
    else:
        log_group = f"/aws/panorama/devices/{device_id}/applications/{application_instance_id}"
    return log_group


def exportSingleLogGroup( log_group, local_dirname ):

    # Export to S3

    logs = boto3.client( "logs", region_name = args.region )
    
    s3_bucket, s3_prefix = splitS3Path(args.s3_path)
    
    start_datetime_utc = datetime.datetime.strptime( args.start_datetime, "%Y%m%d_%H%M%S" )
    end_datetime_utc = datetime.datetime.strptime( args.end_datetime, "%Y%m%d_%H%M%S" )

    response = logs.create_export_task(
        logGroupName = log_group,
        fromTime = int( start_datetime_utc.timestamp() * 1000 ),
        to = int( end_datetime_utc.timestamp() * 1000 ),
        destination = s3_bucket,
        destinationPrefix = s3_prefix,
    )
    
    export_task_id = response["taskId"]
    
    while True:
        
        completed = False
        response = logs.describe_export_tasks( taskId = export_task_id )
        
        for export_task in response["exportTasks"]:
            if export_task["taskId"] == export_task_id:
                status_code = export_task["status"]["code"]
                if "message" in export_task["status"]:
                    status_message = export_task["status"]["message"]
                else:
                    status_message = ""
                print( "Export task status :", status_code, status_message )
                if status_code in ("COMPLETED", "CANCELLED", "FAILED"):
                    completed = True
        
        if completed: break
        
        time.sleep(10)
        
    # Download files to local
    
    s3 = boto3.client( "s3" )

    exported_s3_prefix = s3_prefix + "/" + export_task_id
    response = s3.list_objects_v2( Bucket = s3_bucket, Prefix=exported_s3_prefix )
    for s3_object in response["Contents"]:
        
        exported_s3_key = s3_object["Key"]
        
        assert exported_s3_key.startswith( exported_s3_prefix )
        log_stream_name_and_filename = exported_s3_key[ len(exported_s3_prefix) : ].lstrip("/")
        
        downloaded_local_filepath = os.path.join( local_dirname, log_stream_name_and_filename )
        
        os.makedirs( os.path.split(downloaded_local_filepath)[0], exist_ok=True )
        
        print( "Downloading", exported_s3_key )
        
        s3.download_file(
            Bucket = s3_bucket,
            Key = exported_s3_key,
            Filename = downloaded_local_filepath,
        )

def convertToPlainTextAndNormalize( src_dirname, dst_dirname ):

    for place, dirs, files in os.walk( src_dirname ):
        for filename in files:
            if filename.endswith(".gz"):
                
                src_filepath = os.path.join( place, filename )

                assert src_filepath.startswith( src_dirname )
                dst_filepath = os.path.join( dst_dirname, src_filepath[len(src_dirname):].lstrip("/\\") )
                dst_filepath = os.path.splitext(dst_filepath)[0] + ".log"
                
                print( "Converting to plain text :", dst_filepath )
                
                with gzip.open( src_filepath ) as fd_gz:
                    d = fd_gz.read()
                
                # Sort
                lines = d.splitlines()
                line_group_list = []
                for line in lines:
                    # 2022-06-24T16:50:57.033Z
                    re_result = re.match( rb"[0-9]{4}\-[0-9]{2}\-[0-9]{2}T[0-9]{2}\:[0-9]{2}\:[0-9]{2}\.[0-9]{3}Z .*", line )
                    if re_result is not None:
                        line_group_list.append( [ line ] )
                    else:
                        assert len(line)==0 or line.startswith(b"\t"), str([ line ])
                        line_group_list[-1].append(line)
                line_group_list.sort()
                lines = []
                for line_group in line_group_list:
                    lines += line_group
                d = b"\n".join(lines)               
                
                # Normalize                
                d = d.replace( b"\0", b"\\0" )
                
                os.makedirs( os.path.split(dst_filepath)[0], exist_ok=True )
                with open( dst_filepath, "wb" ) as fd_log:
                    fd_log.write(d)
    

def createZipFile( dirname_to_zip, zip_filename_wo_ext ):

    print( "Creating a Zip file", zip_filename_wo_ext + ".zip" )
    shutil.make_archive( zip_filename_wo_ext, 'zip', dirname_to_zip )


def exportLogsAndCreateZip():

    utcnow = datetime.datetime.utcnow()
    
    with tempfile.TemporaryDirectory() as export_dir:
        with tempfile.TemporaryDirectory() as plaintext_dir:
    
            # export application level logs
            if args.app_id is not None:
                exportSingleLogGroup( getLogGroup( device_id=args.device_id, application_instance_id=args.app_id ), local_dirname = os.path.join( export_dir, args.app_id ) )

            # export device level logs
            exportSingleLogGroup( getLogGroup( device_id=args.device_id ), local_dirname = os.path.join( export_dir, args.device_id ) )
        
            # convert to plain text, sort, and normalize
            convertToPlainTextAndNormalize( export_dir, plaintext_dir )
            
            # create a Zip file from plain text files        
            createZipFile( plaintext_dir, "./panorama_exported_logs_%s" % utcnow.strftime("%Y%m%d_%H%M%S") )
            

exportLogsAndCreateZip()

