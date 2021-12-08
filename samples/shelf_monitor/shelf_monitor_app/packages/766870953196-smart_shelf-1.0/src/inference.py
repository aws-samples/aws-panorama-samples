import json
import os
import logging
import time
import datetime
from logging.handlers import RotatingFileHandler

import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import panoramasdk

s3 = boto3.client('s3', region_name='us-east-1')
sqs = boto3.client('sqs', region_name='us-east-1')

class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.MODEL_NODE = "model_node"
        self.MODEL_DIM = 320
        self.frame_num = 0
        self.threshold = 1.
        # Desired class
        self.classids = [39.]
        # Number of Bottle
        self.number_bottle = 0
        # Aditional parameter input to model
        sqs_url = self.inputs.sqs_url.get()
        s3_bucket = self.inputs.s3_bucket.get()

        logger.info('Initialiation complete.')

    def process_streams(self):
        """Processes one frame of video from one or more video streams."""
        self.frame_num += 1
        logger.debug(self.frame_num)

        # Loop through attached video streams
        streams = self.inputs.video_in.get()
        for stream in streams:
            self.process_media(stream)

        self.outputs.video_out.put(streams)

    def process_media(self, stream):
        """Runs inference on a frame of video."""
        image_data = preprocess(stream.image, self.MODEL_DIM)
        logger.debug(image_data.shape)

        # Run inference
        inference_results = self.call({"data":image_data}, self.MODEL_NODE)

        # Process results (object deteciton)
        self.process_results(inference_results, stream)

    def process_results(self, inference_results, stream):
        """Processes output tensors from a computer vision model and annotates a video frame."""
        if inference_results is None:
            logger.warning("Inference results are None.")
            return

        # Make a copy image to oiverlay bounding box
        image_to_s3 = stream.image.copy()
        w, h, c = image_to_s3.shape
        num_bottle = 0

        class_data = None # Class Data
        bbox_data = None # Bounding Box Data
        conf_data = None # Confidence Data

        # Pulls data from the class holding the results
        # inference_results is a class, which can be iterated through
        # but inference_results has no index accessors (cannot do inference_results[0])

        k = 0
        for det_data in inference_results:
            if k == 0:
                class_data = det_data[0]
            if k == 1:
                conf_data = det_data[0]
            if k == 2:
                bbox_data = det_data[0]
                for a in range(len(conf_data)):
                    if conf_data[a][0] * 100 > self.threshold and class_data[a][0] in self.classids:
                        (left, top, right, bottom) = np.clip(det_data[0][a]/self.MODEL_DIM,0,1)
                        stream.add_rect(left, top, right, bottom)
                        start_point = (int(left*h), int(top*w))
                        end_point = (int(right*h), int(bottom*w))
                        cv2.rectangle(image_to_s3, start_point, end_point, (0,255,0), 2)
                        # Adding confidence score on S3 image
                        print('Adding confidence score on S3 image')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        org = start_point
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(image_to_s3,'bottle: {}'.format(str(conf_data[a][0])), org, font, fontScale, color, thickness)
                        print('confidence score added')
                        num_bottle += 1
                    else:
                        continue
            k += 1

        # sending predicted frame S3 and metadata to SQS
        print('Sending data to cloud')
        self.push_to_cloud(image_to_s3, num_bottle)

        logger.info('# bottle {}'.format(str(num_bottle)))
        stream.add_label('# Number of Bottle {}'.format(str(num_bottle)), 0.1, 0.1)

    def push_to_cloud(self, image_to_send, bottle_count):
        try:
            print('Entering into push_to_cloud block...')
            index = 0
            timestamp = int(time.time())
            now = datetime.datetime.now()
            key = "frames/bottle_{}_{}_{}_{}_{}_{}.jpg".format(now.month, now.day,
                                                   now.hour, now.minute,
                                                   timestamp, index)


            jpg_data = cv2.imencode('.jpg', image_to_send)[1].tostring()

            s3_bucket = self.inputs.s3_bucket.get()
            sqs_url = self.inputs.sqs_url.get()

            # Writing predection frame to S3
            print("Writing prediction frame to S3")
            response = s3.put_object(ACL='private',
                                    Body=jpg_data,
                                    Bucket=s3_bucket,
                                    Key=key,
                                    ContentType='image/JPG')

            print('S3 key: {}'.format(key))

            #Sending S3 metadata to SQS
            print("Sending S3 metadata to SQS")
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                # dd/mm/YY H:M:S
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

                data = {
                        "ProductType": "bottle",
                        "StockCount": bottle_count,
                        "TimeStamp": dt_string,
                        "S3Uri": 's3://' + s3_bucket +'/' + key
                        }

                res = sqs.send_message(
                    QueueUrl=sqs_url,
                    MessageBody=json.dumps(data),
                    MessageGroupId='BottleCounterGroup'
                    )
            else:
                print("Unable to write {} to S3 bucket: {}".format (key, bucket))

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

def preprocess(img, size):
    """Resizes and normalizes a frame of video."""
    resized = cv2.resize(img, (size, size))
    mean = [0.485, 0.456, 0.406]  # RGB
    std = [0.229, 0.224, 0.225]  # RGB
    img = resized.astype(np.float32) / 255.  # converting array of ints to floats
    r, g, b = cv2.split(img)
    # normalizing per channel data:
    r = (r - mean[0]) / std[0]
    g = (g - mean[1]) / std[1]
    b = (b - mean[2]) / std[2]
    # putting the 3 channels back together:
    x1 = [[[], [], []]]
    x1[0][0] = r
    x1[0][1] = g
    x1[0][2] = b
    return np.asarray(x1)

def get_logger(name=__name__,level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=100000000, backupCount=2)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    try:
        logger.info("INITIALIZING APPLICATION")
        app = Application()
        logger.info("PROCESSING STREAMS")
        while True:
            app.process_streams()
    except:
        logger.exception('Exception during processing loop.')

logger = get_logger(level=logging.INFO)
main()
