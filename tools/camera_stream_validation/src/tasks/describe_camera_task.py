'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
from tasks.panorama_task import PanoramaTask

class CameraDescribeTask(PanoramaTask):
    """
    A task to check camera registered or not
    """
    def __init__(self):
        super().__init__()
    
    def run(self, data_source_name):
        try:
            describe_camera_response = self.panorama.describe_package(data_source_name)
        except self.panorama.get_client().exceptions.ResourceNotFoundException:
            raise RuntimeError("Data source: {} does not exist. We will create this data source".format(data_source_name))

        return describe_camera_response