'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
from tasks.panorama_task import PanoramaTask

class AppNodeStatusTask(PanoramaTask):
    """ A task to check nodes status from app """

    def __init__(self):
        super().__init__()

    def run(self, app):
        list_node_instances_response = self.panorama.list_node_instances(app)
        if list_node_instances_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise RuntimeError("Error listing node instance status with status [{}]".format(list_node_instances_response))

        return list_node_instances_response['NodeInstances']