'''
  Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

  PROPRIETARY/CONFIDENTIAL

  Use is subject to Panorama Beta program conditions, see README for more details
'''
import logging
import itertools
import sys
import threading
import time

from service_managers.panorama_api import Panorama

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