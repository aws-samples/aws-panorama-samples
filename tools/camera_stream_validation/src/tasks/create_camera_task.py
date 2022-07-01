'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
from tasks.panorama_task import PanoramaTask
from tasks.describe_camera_task import CameraDescribeTask

DESCRIBE_CAMERA_TERMINAL_STATES = ["SUCCEEDED"]
DESCRIBE_CAMERA_DELAY_IN_SEC = 10
DESCRIBE_CAMERA_TIMEOUT = 20 * 10

class CameraRegisterTask(PanoramaTask):
    """
    A task to register camera package
    """
    def __init__(self):
        super().__init__()

    def run(self, camera_name, camera_credential, version=1.0):

        create_camera_response = self.panorama.create_node_from_template_job(camera_name, camera_credential, version)
        if create_camera_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise RuntimeError("Error: Device does not exist with response [{}]".format(create_camera_response))

        job_id = create_camera_response["JobId"]

        check_state = lambda response: True if response['Status'] in DESCRIBE_CAMERA_TERMINAL_STATES else False
        describe_camera_response = self.wait_for_response(self.panorama.describe_node_from_template_job, check_state,
                                                          DESCRIBE_CAMERA_DELAY_IN_SEC, 
                                                          DESCRIBE_CAMERA_TIMEOUT, job_id)
        
        if describe_camera_response['Status'] != "SUCCEEDED":
            raise RuntimeError("Error registering camera with response [{}]".format(describe_camera_response))

        return create_camera_response
