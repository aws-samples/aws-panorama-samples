# FIXME : merge to existing util module

import subprocess

class ProgressDots:
    def __init__(self):
        self.previous_status = None
    def update_status(self,status):
        if status == self.previous_status:
            print( ".", end="", flush=True )
        else:
            if self.previous_status : print("")
            print( status + " " , end="", flush=True)
            self.previous_status = status
            
# get CloudWatch Logs URL to see application logs
def get_logs_url( region_name, device_id, application_instance_id ):
    log_group = f"/aws/panorama/devices/{device_id}/applications/{application_instance_id}"
    encoded_log_group = log_group.replace( "/", "$252F" )
    return f"https://console.aws.amazon.com/cloudwatch/home?region={region_name}#logsV2:log-groups/log-group/{encoded_log_group}"

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
