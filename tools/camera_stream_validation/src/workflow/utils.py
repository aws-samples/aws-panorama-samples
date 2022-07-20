"""
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
"""
import logging
import itertools
import os
import sys
import threading
import time
import boto3
import base64
from botocore.exceptions import ClientError

from service_managers.panorama_api import Panorama

DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION")


class Logger:
    def __init__(self):
        self.logger = logging.getLogger("AwsPanoramaCSVLogger")

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s:  %(message)s")
            stream = logging.StreamHandler()
            stream.setLevel(logging.INFO)
            stream.setFormatter(formatter)
            self.logger.addHandler(stream)
            self.logger.setLevel(logging.INFO)

    def get_logger(self):
        return self.logger

class LoadingAnimator:
    def __init__(self):
        self.done = False
        self.thread = threading.Thread(target=self.animate)

    def animate(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if self.done:
                break
            sys.stdout.write('\r\t\t\t\t' + c)
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        self.thread.start()

    def end(self):
        self.done = True
        self.thread.join()
        print('\n')

class DeviceValidator:
    def __init__(self):
        self.device_list = [ (device['Name'], device['DeviceId']) for device in Panorama().get_devices()['Devices'] if device['ProvisioningStatus'] == 'SUCCEEDED' ]

        self.logger = Logger().get_logger()

    def select_device(self):
        self.logger.info("There are following devices provisioned at account")
        for n, device in enumerate(self.device_list):
            self.logger.info("[{}] => {}".format(n+1, device))

        self.logger.info("[q] => Quit")
        self.logger.info("Please type corresponding number to select device for validation...")

        while True:
            number = input()
            if number == 'q':
                self.logger.info("Exit...")
                sys.exit(0)

            try:
                number = int(number)
            except Exception as e:
                self.logger.error("Please type valid number to select device")
                sys.exit(-1)

            if number > len(self.device_list) or number <= 0:
                self.logger.error("Please select available device listed above and type again...")
                continue

            self.logger.info("Select device {}".format(self.device_list[number-1]))
            break

        return self.device_list[number-1]




def get_secret(secret_name, region_name=None):
    """
    gets a secret from AWS Secrets Manager

    :param secret_name str: Secret Name
    :param region_name str: Optional AWS region for the client to query
    :raises ClientError: DecryptionFailureException
    :raises ClientError: InternalServiceErrorException
    :raises ClientError: InvalidParameterException
    :raises ClientError: InvalidRequestException
    :raises ClientError: ResourceNotFoundException
    """

    logger = Logger().get_logger()

    # Create a Secrets Manager client
    client = boto3.client(
        service_name="secretsmanager",
        region_name=region_name if region_name is not None else DEFAULT_REGION,
    )

    try:
        logger.info(f"Retrieving [{secret_name}] from AWS Secrets Manager")
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side.
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            # You provided an invalid value for a parameter.
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            # You provided a parameter value that is not valid for the current state of the resource.
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            # We can't find the resource that you asked for.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            logger.info("Secret retrieved. Reading [SecretString]")
            secret = get_secret_value_response["SecretString"]
            return secret
        else:
            logger.info("Secret retrieved. Reading [SecretBinary]")
            decoded_binary_secret = base64.b64decode(
                get_secret_value_response["SecretBinary"]
            )
            return decoded_binary_secret

