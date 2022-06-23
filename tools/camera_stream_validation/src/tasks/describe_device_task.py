'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
from tasks.panorama_task import PanoramaTask


class DescribeDeviceTask(PanoramaTask):
    """A task to check device status"""

    def __init__(self):
        super().__init__()

    def run(self, device, status='ONLINE'):
        device_response = self.panorama.describe_device(device)
        if device_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise RuntimeError("Error: Device does not exist with response [{}]".format(device_response))
        return device_response