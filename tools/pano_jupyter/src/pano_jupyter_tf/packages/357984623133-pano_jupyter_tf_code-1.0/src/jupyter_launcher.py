import os
import subprocess

def main():
    
    print( "Launching Jupyter", flush=True )
    
    # /root is readonly, so need to replace jupyter config and data directories
    os.environ["JUPYTER_CONFIG_DIR"] = "/opt/aws/panorama/storage/.jupyter"
    os.environ["JUPYTER_DATA_DIR"] = "/opt/aws/panorama/storage/.local/share/jupyter"
    
    # launching jupyter-lab in sub-process
    cmd = "jupyter-lab --no-browser --allow-root --ip 0.0.0.0 --port 8888 --notebook-dir /opt/aws/panorama/storage"
    result = subprocess.run( cmd, shell=True, capture_output=False )

    print( "Jupyter server ended", flush=True )    

main()

