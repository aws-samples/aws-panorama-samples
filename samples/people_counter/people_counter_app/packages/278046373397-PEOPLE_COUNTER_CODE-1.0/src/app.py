
import json
import logging
import time
from logging.handlers import RotatingFileHandler

import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import panoramasdk


class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.MODEL_NODE = "model_node"
        self.MODEL_DIM = 512
        self.frame_num = 0
        self.threshold = 50.
        # Desired class
        self.classids = [14.]
        
        try:
            # Parameters
            logger.info('Getting parameters')
            self.threshold = self.inputs.threshold.get()
        except:
            logger.exception('Error during initialization.')
        finally:
            logger.info('Initialiation complete.')
            logger.info('Threshold: {}'.format(self.threshold))
            

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

        num_people = 0

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
                        num_people += 1
                    else:
                        continue
            k += 1
        
        logger.info('# people {}'.format(str(num_people)))
        stream.add_label('# people {}'.format(str(num_people)), 0.1, 0.1)


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
