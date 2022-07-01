'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
import logging

from models.app import App
from tasks.panorama_task import PanoramaTask

logger = logging.getLogger(__name__)

DESCRIBE_APP_TERMINAL_STATES = ["DEPLOYMENT_SUCCEEDED", "DEPLOYMENT_FAILED", "REMOVAL_SUCCEEDED", "REMOVAL_FAILED"]
DESCRIBE_APP_DELAY_IN_SEC = 60
DESCRIBE_APP_TIMEOUT = 20 * 60


class AppDeploymentTask(PanoramaTask):
    """
    A task to deploy given app to device
    """

    def __init__(self):
        super().__init__()

    def run(self, device, app: App):
        create_app_response = self.panorama.create_app(app.get_name(),
                                                       app.get_manifest_as_str(),
                                                       app.get_override_as_str(),
                                                       device)
        if create_app_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise RuntimeError("Error creating app with response [{}]".format(create_app_response))
        app_id = create_app_response['ApplicationInstanceId']

        check_state = lambda response: True if response['Status'] in DESCRIBE_APP_TERMINAL_STATES else False
        describe_app_response = self.wait_for_response(self.panorama.describe_app, check_state,
                                                       DESCRIBE_APP_DELAY_IN_SEC,
                                                       DESCRIBE_APP_TIMEOUT, app_id)
        if describe_app_response['Status'] != "DEPLOYMENT_SUCCEEDED":
            raise RuntimeError("Error deploying app with response [{}]".format(describe_app_response))
        return describe_app_response