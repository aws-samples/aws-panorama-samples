'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
import logging
import time
from abc import abstractmethod
from service_managers.panorama_api import Panorama

logger = logging.getLogger(__name__)


class PanoramaTask:
    """
    A Panorama task is a unit of task achieved by calling either a single
    or multiple service APIs
    """

    def __init__(self):
        self.panorama = Panorama()

    @abstractmethod
    def run(self, **kwargs):
        pass

    def wait_for_response(self, func, check_response, delay_in_sec, timeout_in_sec, *args):
        """
        Many of the service APIs like OTA, App deployment are long running async jobs and provide a
        read api to check the status. This helper method provides a way to run the read api at a certain
        cadence and provides a callback to check if a terminal state has been achieved.

        This method will either return a successful response or will throw a TimeoutError
        """
        start_time = time.time()
        while time.time() - start_time < timeout_in_sec:
            response = func(args[0])
            if check_response(response):
                return response
            time.sleep(delay_in_sec)

        logger.error("Timeout after {} seconds when calling {} and no terminal condition could be reached.".format(
            timeout_in_sec, str(func)))
        raise TimeoutError(
            "Timeout after {} seconds when calling {} and no terminal condition could be reached.".format(
                timeout_in_sec, str(func)))