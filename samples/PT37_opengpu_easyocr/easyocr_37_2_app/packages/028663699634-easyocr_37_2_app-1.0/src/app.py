import logging
from logging.handlers import RotatingFileHandler

import easyocr
import numpy as np
import panoramasdk


class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.ocr_detector = easyocr.Reader(["en"], gpu=True)

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
        image_data = stream.image
        logger.debug(image_data.shape)

        # Cropping image to focus on region to read OCR (top left region with 100*100 pixels)
        cropped_image = image_data[:100, :100, :]

        # Process results (object deteciton)
        self.process_results(cropped_image, stream)

    def process_results(self, cropped_image, stream):
        """Processes output tensors from a computer vision model and annotates a video frame."""
        if any(dim_size==0 for dim_size in cropped_image.shape):
            logger.warning("Image size too small")
            return

        list_of_words_detected = self.ocr_detector.readtext(cropped_image, detail=0)
        
        drift = 0.05

        # Logging and printing first 5 words
        for idx, word in enumerate(list_of_words_detected[:5]):
            logger.info('word #{} = {}'.format(str(idx), word))
            stream.add_label('word #{} = {}'.format(str(idx), word), 0.1, 0.1*(drift*idx))

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
