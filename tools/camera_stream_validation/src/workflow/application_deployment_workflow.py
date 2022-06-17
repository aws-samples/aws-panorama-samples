'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
import json
import sys

from models.app import App

from tasks.app_deployment_task import AppDeploymentTask
from tasks.app_running_status_task import AppRunningStatusTask
from tasks.app_deletion_task import AppDeletionTask
from tasks.describe_camera_task import CameraDescribeTask
from tasks.create_camera_task import CameraRegisterTask
from tasks.app_node_status_task import AppNodeStatusTask
from tasks.remove_camera_task import RemoveCameraTask
from tasks.app_deployed_query_task import AppDeployedQueryTask

import time
from enum import Enum
from workflow.utils import Logger

logger = Logger().get_logger()

DEFAULT_APP_TTL_MINUTES = 2

class Terminator(Enum):
    Register = 1
    Deploy = 2
    Remove = 3

def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(e)

    return inner_function

OUTPUT_TITLE = "ValidationResult"

class ApplicationDeploymentWorkflow:

    def __init__(self, account_id, device_id, output):
        self.account_id = account_id
        self.device = device_id
        self.output = output
    
    @exception_handler
    def check_data_source(self, name) -> bool:
        logger.info("Data source: {}".format(name))
        describe_camera_task = CameraDescribeTask()
        try:
            describe_camera_task.run(name)
        except RuntimeError as e:
            logger.warning(e)
            return False

        return True

    @exception_handler
    def create_data_source(self, name, username, password, stream_url, version=1.0) -> str:
        logger.info("Creating data source: {}. This may take a minute...".format(name))

        camera_credential = {}
        if username:
            camera_credential["Username"] = username
        if password:
            camera_credential["Password"] = password
        if stream_url:
            camera_credential["StreamUrl"] = stream_url

        create_camera_package_task = CameraRegisterTask()
        create_camera_package_response = create_camera_package_task.run(name, camera_credential, version)

        return create_camera_package_response

    @exception_handler
    def remove_camera_packapge(self, data_sources):
        if len(data_sources) == 0:
            return
        logger.info("Removing camera packages...")
        task = RemoveCameraTask()
        for data_source_to_be_removed in data_sources:
            logger.info("To remove data source: {}".format(data_source_to_be_removed))
            task.run(data_source_to_be_removed)


    @exception_handler
    def deploy_application(self, data_sources) -> str:
        # TODO download the manifest file from S3 bucket
        logger.info("Deploying application...")
        app = App(
            account_id=self.account_id,
            name="panorama_camera_stream_validation"
        )
        if len(data_sources) != 0:
            app.generate_override(data_sources)
        app_deployment_task = AppDeploymentTask()
        app_deployed_response = app_deployment_task.run(self.device, app)

        return app_deployed_response['ApplicationInstanceId']

    @exception_handler
    def verify_application_status(self, app_id) -> bool:
        logger.info("Verifying application status")
        app_describe_task = AppRunningStatusTask()
        return app_describe_task.run(app_id)

    @exception_handler
    def remove_application(self, app_id) -> bool:
        logger.info("Removing application")
        remove_task = AppDeletionTask()
        return remove_task.run(self.device, app_id)

    @exception_handler
    def verify_node_status(self, app_id):
        logger.info("Validation status from app {}".format(app_id))
        task = AppNodeStatusTask()
        verification_result = task.run(app_id)

        json_result = { app_id: {} }
        for node in verification_result:
            json_result[app_id][node['NodeInstanceId']] = node['CurrentStatus']
            if node['CurrentStatus'] == 'RUNNING':
                logger.info("=> Camera [{}] status: <{}>".format(node['NodeInstanceId'], node['CurrentStatus']))
            else:
                logger.info('''=> Camera [{}] status: <{}>
                                 Please verify that the device can access the camera, verify credentials and RTSP url'''.format(node['NodeInstanceId'], node['CurrentStatus']))

        if self.output != "":
            list_obj = []
            with open(self.output) as fd:
                list_obj = json.load(fd)
            with open(self.output, 'w') as fd:
                list_obj[OUTPUT_TITLE].append(json_result)
                json.dump(list_obj, fd, indent=4)
                logger.info("Successfully dump validation result to the output {}".format(self.output))

    def data_source_check_and_create(self, data_sources):
        logger.info("List of data sources to validate for this deployment")
        # datasource check
        for data_source in data_sources:
            if not self.check_data_source(data_source[0]):
                self.create_data_source(data_source[0], # name
                                        data_source[1], # Username
                                        data_source[2], # Password
                                        data_source[3], # StreamUrl
                                        data_source[4]) # version

    def data_source_removal(self, data_sources):
        remove_list = []
        for data_source in data_sources:
            if data_source[-1]:
                remove_list.append(data_source[0])

        self.remove_camera_packapge(remove_list)

    @exception_handler
    def list_deployed_application(self):
        task = AppDeployedQueryTask()
        app_instance_id_list = task.run(self.device)
        for app_id in app_instance_id_list:
            self.verify_node_status(app_id)

    def run(self, termination, data_sources=[]):

        data_source_list = []
        data_source_remove_list = []
        for data_source in data_sources:
            data_source_list.append(data_source[0])
            if data_source[-1]:
                data_source_remove_list.append(data_source[0])

        if termination <= Terminator.Register.value:
            logger.info("Terminate flow after camera package created")
            return
        # Application Deployment
        app_id = self.deploy_application(data_source_list)

        if app_id:
            logger.info("Application deployment finished with app id {}".format(str(app_id)))
            # Verify deployment status
            logger.info("Prepare to validate camera list {}...".format(data_sources))

            # Wating for verify app running status, return when app is Running/Error
            start_time = time.time()
            while time.time() - start_time < DEFAULT_APP_TTL_MINUTES * 60:
                time.sleep(60)

            # Return node status
            logger.info("Camera verification result:")
            self.verify_node_status(app_id)

            if termination <= Terminator.Deploy.value:
                logger.info("Terminate flow after validation application deployed, keep it on device")
                return

            # Application deletion
            self.remove_application(app_id)

