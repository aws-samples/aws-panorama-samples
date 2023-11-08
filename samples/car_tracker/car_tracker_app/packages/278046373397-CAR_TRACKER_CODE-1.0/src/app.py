import tensorflow as tf
import json
import logging
import time
from logging.handlers import RotatingFileHandler

import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import panoramasdk
import datetime
from CentroidTracker import CentroidTracker

class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.MODEL_NODE = "model_node"
        self.MODEL_DIM = 300
        self.frame_num = 0
        self.tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
        self.tracked_objects = []
        self.tracked_objects_start_time = dict()
        self.tracked_objects_duration = dict()

        root = tf.saved_model.load("/panorama/ssd_mobilenet_v2_coco_tf_trt")
        self.model = root.signatures['serving_default']

        try:
            # Parameters
            logger.info('Configuring parameters.')
            self.threshold = self.inputs.threshold.get()
            
            # Desired class
            self.classids = [3.]

        except:
            logger.exception('Error during initialization.')
        finally:
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
        inference_results = [x.numpy() for x in self.model(tf.convert_to_tensor(image_data)).values()]

        # Process results (object deteciton)
        self.process_results(inference_results, stream)

    def process_results(self, inference_results, stream):
        """Processes output tensors from a computer vision model and annotates a video frame."""
        if inference_results is None:
            logger.warning("Inference results are None.")
            return
        
        w,h,c = stream.image.shape

        conf_scores = None
        classes = None
        bboxes = None
        rects = []

        for det in inference_results:
            if det.shape[-1] == 4:
                bboxes = det[0]
            elif det.shape[-1] == 100:
                if det[0][0] >= 1:
                    classes = det[0]
                else:
                    conf_scores = det[0]
        
        for a in range(len(conf_scores)):
            if conf_scores[a] * 100 > self.threshold and classes[a] in self.classids:
                (top, left, bottom, right) = bboxes[a]
                rects.append([left*w, top*h, right*w, bottom*h])
                stream.add_rect(left, top, right, bottom)
                
        rects = np.array(rects)
        rects = rects.astype(int)
        objects = self.tracker.update(rects)
        
        logger.info('Tracking {} cars'.format(len(objects)))
        
        for (objectID, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            if objectID not in self.tracked_objects:
                self.tracked_objects.append(objectID)
                self.tracked_objects_start_time[objectID] = datetime.datetime.now()
                self.tracked_objects_duration[objectID] = 0
            else:
                time_diff = datetime.datetime.now() - self.tracked_objects_start_time[objectID]
                sec = time_diff.total_seconds()
                self.tracked_objects_duration[objectID] = sec
            
            duration = self.tracked_objects_duration[objectID]
            
            logger.info('CarId: {} at ({},{}) for {}'.format(objectID, x1, y1, duration))
            stream.add_rect(x1/w, y1/h, x2/w, y2/h)
            stream.add_label('{}s'.format(str(duration)), x1/w, y1/h)

def preprocess(img, size):
    """Resizes and normalizes a frame of video."""
    resized = cv2.resize(img, (size, size))
    x1 = np.asarray(resized)
    x1 = np.expand_dims(x1, 0)
    return x1

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
