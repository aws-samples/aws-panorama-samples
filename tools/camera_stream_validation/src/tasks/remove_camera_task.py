'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
from tasks.panorama_task import PanoramaTask

class RemoveCameraTask(PanoramaTask):

    """A task to remove camera package"""
    def __init__(self):
        super().__init__()

    def run(self, data_source):
        self.panorama.remove_node_package(data_source)

