'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
import logging

from tasks.panorama_task import PanoramaTask

logger = logging.getLogger(__name__)

DESCRIBE_APP_TERMINAL_STATES = ["DEPLOYMENT_SUCCEEDED", "DEPLOYMENT_FAILED", "REMOVAL_SUCCEEDED", "REMOVAL_FAILED"]
REMOVE_APP_DELAY_IN_SEC = 60
REMOVE_APP_TIMEOUT = 9 * 60


class AppDeletionTask(PanoramaTask):
    """
    A task to delete given app from device
    """

    def __init__(self):
        super().__init__()

    def run(self, device, app_id):
        if not app_exists(self.panorama, app_id):
            return False

        remove_app_response = self.panorama.remove_app(app_id)
        if remove_app_response['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise RuntimeError("Error deleting app with response [{}]".format(remove_app_response))

        check_removal_state = lambda response: True if any(
            appInstance['ApplicationInstanceId'] == app_id and appInstance['Status'] == 'REMOVAL_SUCCEEDED' for
            appInstance in response['ApplicationInstances']) else False
        self.wait_for_response(self.panorama.list_app_instances, check_removal_state, REMOVE_APP_DELAY_IN_SEC,
                               REMOVE_APP_TIMEOUT, device)
        return check_removal_state


def app_exists(panorama, app_id):
    """
    Returns true if app exists
    """

    app_response = panorama.describe_app(app_id)
    application_exists = True if app_response['ResponseMetadata']['HTTPStatusCode'] == 200 else False

    return application_exists