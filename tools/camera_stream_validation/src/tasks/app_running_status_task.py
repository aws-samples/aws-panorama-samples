'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
import logging

from tasks.panorama_task import PanoramaTask

APP_RUNNING_STATUS = ["RUNNING", "ERROR"]
DESCRIBE_APP_DELAY_IN_SEC = 30
DESCRIBE_APP_TIMEOUT = 6 * 30
logger = logging.getLogger(__name__)


class AppRunningStatusTask(PanoramaTask):
    """
    A task to check if given app is running on device
    """

    def __init__(self):
        super().__init__()

    def run(self, app):
        describe_app_response = self.panorama.describe_app(app)
        if describe_app_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise RuntimeError("Error describing app with response [{}]".format(describe_app_response))

        check_state = lambda response: True if response['HealthStatus'] in APP_RUNNING_STATUS else False
        describe_app_response = self.wait_for_response(self.panorama.describe_app, check_state,
                                                       DESCRIBE_APP_DELAY_IN_SEC,
                                                       DESCRIBE_APP_TIMEOUT, app)

        if not describe_app_response['HealthStatus'] in APP_RUNNING_STATUS:
            raise RuntimeError("Error app is not running with response [{}]".format(describe_app_response))

        return True