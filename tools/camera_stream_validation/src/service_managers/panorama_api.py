'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''

import boto3

class Panorama:
    def __init__(self):
        self.client = boto3.client(service_name='panorama')
    
    def get_client(self):
        return self.client

    def describe_device(self, device):
        return self.client.describe_device(DeviceId=device)

    def describe_app(self, app):
        app_id = app if type(app) == str else app[0]
        return self.client.describe_application_instance(
            ApplicationInstanceId=app_id
        )

    def list_node_instances(self, app):
        app_id = app if type(app) == str else app[0]
        return self.client.list_application_instance_node_instances(
            ApplicationInstanceId=app_id
        )

    def remove_app(self, app):
        return self.client.remove_application_instance(
            ApplicationInstanceId=app
        )

    def list_app_instances(self, device, status=""):
        dev_id = device if type(device) == str else device[0]
        if status != "":
            return self.client.list_application_instances(
                DeviceId=dev_id,
                MaxResults=25,
                StatusFilter=status
            )
        return self.client.list_application_instances(
            DeviceId=dev_id,
            MaxResults=25
        )

    def create_app(self, name, manifest, override, device):
        payload_data_manifest = manifest if type(manifest) == str else manifest[0]
        payload_data_override = override if type(override) == str else override[0]
        return self.client.create_application_instance(
            Name=name,
            ManifestPayload={
                'PayloadData': payload_data_manifest
            },
            ManifestOverridesPayload={
                'PayloadData': payload_data_override
            },
            DefaultRuntimeContextDevice=device
        )

    def describe_package(self, package_name):
        return self.client.describe_package(
            PackageId="packageName/"+package_name
        )

    def describe_package_version(self, package_name, package_version):
        return self.client.describe_package_version(
            PackageId="packageName/" + package_name,
            PackageVersion=str(package_version)
        )

    def create_node_from_template_job(self, camera_name, camera_credential, version):
        return self.client.create_node_from_template_job(
            TemplateType='RTSP_CAMERA_STREAM',
            OutputPackageName=camera_name,
            OutputPackageVersion=str(version),
            NodeName=camera_name,
            TemplateParameters=camera_credential
        )

    def describe_node_from_template_job(self, job_id):
        return self.client.describe_node_from_template_job(
            JobId=job_id
        )

    def remove_node_package(self, package_name):
        return self.client.delete_package(
            ForceDelete=True,
            PackageId="packageName/" + package_name,
        )

    def get_devices(self):
        return self.client.list_devices()