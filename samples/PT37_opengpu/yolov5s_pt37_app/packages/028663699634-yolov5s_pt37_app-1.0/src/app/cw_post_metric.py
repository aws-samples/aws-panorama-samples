import boto3
from botocore.exceptions import ClientError
import time
from threading import Thread
import multiprocessing
from multiprocessing import Process
import logging
import struct
import json
import os
import select
from log_utils import get_logger
from message_utils import encode_msg_size, decode_msg_size, create_msg, get_message

BATCH_SIZE = 150
IPC_FIFO_NAME = "/opt/aws/panorama/logs/post_metrics_ipc"

log = get_logger()


class PostMetricProcess(Process):

    def __init__(self, namespace, fifoName):
        super(PostMetricProcess, self).__init__()
        self.namespace = namespace
        #self.q = q
        self.metrics_buffer = list()
        self.cw_client = boto3.client('cloudwatch', region_name='us-west-2')
        self.fifoName = fifoName


    def run(self):
        os.mkfifo(self.fifoName)
        fifo = os.open(self.fifoName, os.O_RDONLY | os.O_NONBLOCK)
        poll = select.poll()
        poll.register(fifo, select.POLLIN)
        while(True):
            try:
                if(fifo, select.POLLIN) in poll.poll(1000):
                    msg = get_message(fifo)
                    metric_data = json.loads(msg)
                    self.metrics_buffer.append(metric_data)
                    if(len(self.metrics_buffer)>=BATCH_SIZE):
                        self.post_metric_data(self.metrics_buffer)
                        self.metrics_buffer = list()
            except Exception as err:
                log.exception("Exception reading from fifo", err)
                pass

    def make_cw_metric_datum(self, metric_data):
        cw_metric_datum = dict()
        cw_metric_datum['MetricName'] = metric_data['MetricName']
        cw_metric_datum['Timestamp'] = metric_data['Timestamp']
        cw_metric_datum['Values'] = [metric_data['Value']]
        cw_metric_datum['Counts'] = [metric_data['Count']]
        cw_metric_datum['Unit'] = metric_data['Unit']
        cw_metric_datum['Dimensions'] = metric_data['Dimensions']
        return cw_metric_datum

    def post_metric_data(self, metric_data_list):
        # Metrics aggregated by name and timestamp\
        aggregated_metrics = dict()
        for metric_data in metric_data_list:
            metric_name = metric_data['MetricName']
            timestamp = metric_data['Timestamp']
            if metric_name in aggregated_metrics:
                if timestamp in aggregated_metrics[metric_name]:
                    # CW does not accept more than 150 values in a single PutMetricData call
                    if len(aggregated_metrics[metric_name][timestamp][-1]['Values']) <= 150:
                        cw_metric_datum = aggregated_metrics[metric_name][timestamp][-1]
                        cw_metric_datum['Values'].append(metric_data['Value'])
                        cw_metric_datum['Counts'].append(metric_data['Count'])
                    else:
                        aggregated_metrics[metric_name][timestamp].append(self.make_cw_metric_datum(metric_data))
                else:
                   aggregated_metrics[metric_name][timestamp] = [self.make_cw_metric_datum(metric_data)]
            else:
                aggregated_metrics[metric_name] = dict()
                aggregated_metrics[metric_name][timestamp] = [self.make_cw_metric_datum(metric_data)]

        cw_metric_data = list()
        for metric_name in aggregated_metrics:
            value = aggregated_metrics[metric_name]
            for timestamp in value:
                cw_metric_data.extend(value[timestamp])
        try:
            if len(cw_metric_data)>0:
                response = self.cw_client.put_metric_data(
                    Namespace = self.namespace,
                    MetricData = cw_metric_data
                )
                #log.debug('CW response [{}]'.format(response))
        except Exception as err:
            log.exception("Unable to post metrics for namespace [{}], err: {}".format(self.namespace, err))
            pass

class MetricsHandler:
    """
    Allows one to publish metrics to Cloudwatch.

    Usage:
    metricsHandler = MetricsHandler()
    metricsHandler.put_metric(metric_data)

    metric_data is a dictionay and should be of this form
    {
      "MetricName" : "MyMetric",
      "Timestamp" : "2021"
      "Value": 60.0,
      "Unit": "Seconds"
      "Dimension" : [{
          "Name" : "AppFunction",
          "Value": "ObjectDetection"
          },{
          "Name" : "NumStreams",
          "Value": "4"
          },{
          "Name" : "ModelType",
          "Value": "Squeezenet"
          },{
          "Name" : "StreamResolution",
          "Value": "720"
        }
      ]
    }

    Valid values of ```Unit``` are  Seconds | Microseconds | Milliseconds | Bytes | Kilobytes | Megabytes | Gigabytes | Terabytes | Bits | Kilobits | Megabits | Gigabits | Terabits | Percent | Count | Bytes/Second | Kilobytes/Second | Megabytes/Second | Gigabytes/Second | Terabytes/Second | Bits/Second | Kilobits/Second | Megabits/Second | Gigabits/Second | Terabits/Second | Count/Second | None
    Value values of ```Value``` are double within the range -2^360 to 2^360
    Timestamp should be used as datetime.datetime.now().replace(microsecond=0)
    """

    def __init__(self, namespace, metrics_factory):
        # Run this in an environment where boto3 can fetch credentials for writing to cloudwatch
        self.namespace = namespace
        self.metrics_factory = metrics_factory
        #self.q = Queue()
        #self.post_thread = Thread(target=self.run, args=(self.q,), daemon=True)
        if os.path.exists(IPC_FIFO_NAME):
            #log.info('IPC file exists, deleting')
            os.remove(IPC_FIFO_NAME)
        #log.info('Creating IPC file')
        try:
           self.post_process = PostMetricProcess(self.namespace, IPC_FIFO_NAME)
           self.post_process.start()
        except Exception as err:
            log.exception(err)
            pass
        while(not os.path.exists(IPC_FIFO_NAME)):
            time.sleep(1)
        self.fifo = os.open(IPC_FIFO_NAME, os.O_WRONLY)

    def terminate(self):
        self.post_process.terminate()

    def kill(self):
        self.post_process.kill()

    def get_metric(self, name):
        return self.metrics_factory.get_metric_object(name)

    def put_metric(self, metric):
        msg = create_msg(json.dumps(metric.get_cw_metric_object(), default=str).encode('utf-8'))
        try:
            os.write(self.fifo, msg)
        except Exception as err:
            log.exception('Unable to write to fifo', err)
            pass
        #self.q.put(metric.get_cw_metric_object())

    def put_metric_value(self, name, value, unit):
        metric = self.metrics_factory.get_metric_object(name)
        metric.add_value(value, unit)
        self.put_metric(metric)

    def put_metric_count(self, name, value):
        metric = self.metrics_factory.get_metric_object(name)
        metric.add_count(value, 1)
        self.put_metric(metric)

