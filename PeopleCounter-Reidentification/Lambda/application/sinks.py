import json
import os
import datetime
import logging
import boto3

COLNAMES = ["time", "location", "enter", "exit"]

class Record(object):
    def __init__(self, location, time, enter, exit):
        """
        :param location: location (signifies the device in this case)
        :param time: time of the event
        :param enter: number of unique people entered
        :param exit: number of unique people exited
        """
        self.record = dict()
        self.record['time'] = time
        self.record["location"] = location
        self.record['enter'] = enter
        self.record['exit'] = exit


    def __eq__(self, other):
        return self == other


    def __repr__(self):
        """str representation used to for output to MQTT"""
        return json.dumps(self.record)


    def to_list(self):
        """to list with order of column"""
        return [self.record[x.lower()] for x in COLNAMES]


    def eq_ignore_time(self, other):
        """only used to determined current record is exactly with previous one"""
        original_record = dict(self.record)
        record_cpy = dict(other.record)
        del original_record['time']
        del record_cpy['time']
        return original_record == record_cpy


class Sink(object):
    def __init__(
        self, 
        aws_iot_endpoint_url,
        enable_mqtt, 
        enable_kinesis):
        """
        :param enable_mqtt: True/False
        :param enable_kinesis: True/False
        """
        # Indicates if MQTT sink is enabled
        self.enable_mqtt = enable_mqtt 
        # Indicates if Kinesis video sink is enabled
        self.enable_kinesis = enable_kinesis
        # Get IoT Green grass name. Can be used as location
        self.device_name = self.fetch_device_name()
        # Greengrass SDK client
        self.iot_boto_client = boto3.client(
            'iot-data',
            endpoint_url=aws_iot_endpoint_url)
        # IoT topic
        self.iot_topic = f"topic/{self.device_name}/inference".format(self.device_name)


    def fetch_device_name(self):
        """
        :return: return device name which is set
        during the provisioning, expects to be the
        location name
        """
        thing = os.environ['AWS_IOT_THING_NAME']
        arr = thing.split("_")
        return "_".join(arr[1:-2])

    def sink(self, enter, exit):
        """
        :param enter: number of unique people entered
        :param exit: number of unique people exited
        :return:
        """
        record = Record(
            location=self.device_name, 
            time=self._fetch_time(), 
            enter=enter,
            exit=exit)

        if self.enable_mqtt:
            self._mqtt_sink(record)


    def _fetch_time(self, millisecond=False):
        """fetch human readable time with precision to seconds"""
        if not millisecond:
            return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


    def _mqtt_sink(self, record):
        """send real time record to iot topic
        :param record: record
        :return:
        """
        logging.info(f'Publishing payload to {self.iot_topic}:{str(record)}')
        print(f'Publishing payload to {self.iot_topic}:{str(record)}')
        boto_response = self.iot_boto_client.publish(
            topic=self.iot_topic,
            qos=0,
            payload=str(record).encode())
        logging.info(f'Boto Response:{boto_response}')
        print(f'Boto Response:{boto_response}')


