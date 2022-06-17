'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
from tasks.panorama_task import PanoramaTask

class AppDeployedQueryTask(PanoramaTask):
    def __init__(self):
        super().__init__()

    def run(self, device_id):
        list_deployed_app_instances_response = self.panorama.list_app_instances(device_id, "DEPLOYMENT_SUCCEEDED")
        if list_deployed_app_instances_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise RuntimeError("Error listing deployed app with response [{}]".format(list_deployed_app_instances_response))

        return [app['ApplicationInstanceId'] for app in list_deployed_app_instances_response["ApplicationInstances"] if app['Name'] == 'panorama_camera_stream_validation']