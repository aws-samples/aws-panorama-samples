import torch
import torch_tensorrt

import json
import logging
import time
from logging.handlers import RotatingFileHandler

import boto3
import cv2
import numpy as np
import panoramasdk

import os
os.environ["GST_DEBUG"] = "2"
os.environ["GST_PLUGIN_PATH"] = "$GST_PLUGIN_PATH:/usr/local/lib/gstreamer-1.0/:/amazon-kinesis-video-streams-producer-sdk-cpp/build"

from types import SimpleNamespace
from bytetracker.byte_tracker import BYTETracker
from yolox_postprocess import demo_postprocess, multiclass_nms

from datetime import datetime, timezone

class Application(panoramasdk.node):
    def __init__(self):
        """Initializes the application's attributes with parameters from the interface, and default values."""
        self.VIDEO_RECORDING = False
        self.STREAM_ID = 0
        self.MODEL_NODE = "model_node"
        self.MODEL_INPUT = (640, 640)  #YOLOX
        self.source_fnum = 0
        self.target_fnum = 0
        
        converted_model_path = '/opt/aws/panorama/storage/converted_model.ts'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(converted_model_path):
            logger.info('Converting model for use with TRT...')
            yolovx = torch.jit.load('/panorama/yolox_m_neo.pth', map_location=torch.device(self.device))
            model = yolovx.eval()
            inputs = [
                torch_tensorrt.Input(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640],
                    dtype=torch.float,
                )
            ]
            enabled_precisions = {torch.float}
            self.trt_model = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)
            torch.jit.save(self.trt_model, converted_model_path)
        else:
            logger.info('Loading saved TRT-compatible model...')
            self.trt_model = torch.jit.load(converted_model_path, map_location=torch.device(self.device))
        logger.info('TRT-compatible model ready for use!')

        #for uploading still-shot each start day
        self.refresh = True
        self.lastday = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.today = self.lastday
        
        #Origin size
        streams = self.inputs.video_in.get()
        stream = streams[0]
        width = stream.image.shape[1]
        height = stream.image.shape[0]
        
        self.CAMERA_INPUT = (height, width)
        
        # Parameters
        logger.info('Getting parameters')
        self.service_region = self.inputs.service_region.get()
        self.bucket_name = self.inputs.bucket_name.get()
        self.kinesis_name = self.inputs.kinesis_name.get()
        self.kinesis_video_name = self.inputs.kinesis_video_name.get()
        
        session = boto3.Session(region_name=self.service_region)
        self.s3 = session.client('s3')
        self.firehose = session.client('firehose')
            
        self.SOURCE_FPS = self.inputs.source_fps.get() #30
        self.TARGET_FPS = self.inputs.target_fps.get() #10
        self.CATEGORY = compile(self.inputs.yolox_category.get(), '<string>', 'eval') #[0,1,2]
        self.VERTICAL_RATIO = round(self.inputs.vertical_ratio.get(), 2) #1.6
        
        self.args = SimpleNamespace(**{
                    'nms': round(self.inputs.nms.get(),2), #0.45
                    'track_thresh': round(self.inputs.track_thresh.get(), 2), #0.65
                    'track_buffer': self.inputs.track_buffer.get(), #30
                    'match_thresh': round(self.inputs.match_thresh.get(), 2), #0.9
                    'min_box_area': self.inputs.min_box_area.get(), #100 w*h
                    'mot20': False})
            
        self.trackers = [BYTETracker(self.args, frame_rate=self.TARGET_FPS) for _ in range(len(streams))]
        
        gst_out = self.inputs.gstreamer_encoder.get()
        if len(gst_out) > 0:
            self.VIDEO_RECORDING = True
            kvssecret = session.client('secretsmanager')                
            aksk = json.loads(kvssecret.get_secret_value(SecretId='KVSSecret')['SecretString'])
            gst_out += f" ! kvssink log-config=/amazon-kinesis-video-streams-producer-sdk-cpp stream-name={self.kinesis_video_name} framerate={self.TARGET_FPS} access-key={aksk['accesskey']} secret-key={aksk['secretkey']} aws-region={self.service_region} "
            self.videowriter = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(self.TARGET_FPS), (width, height))
        
        logger.info('Initialiation complete.')
        logger.info('Args: {}'.format(self.args))

    def stop(self):
        if self.VIDEO_RECORDING == True:
            self.videowriter.release()
        logger.info('Terminated.')
        
    def resetstate(self):
        streams = self.inputs.video_in.get()
        for stream, tracker in zip(streams, self.trackers):
            image = cv2.imencode('.png', stream.image)[1].tostring()
            self.s3.put_object(Body=image, Bucket=self.bucket_name, 
                          Key=f"dailycapture/{stream.stream_id}/{self.today}.png", ContentType='image/PNG')
            #Refresh byte_track
            tracker.reset()

    def process_streams(self):
        """Processes one frame of video from one or more video streams."""
        self.source_fnum += 1
        
        #Check every minute to refresh check, 30 * 60 is 60 seconds
        if self.source_fnum % 1800 == 0:
            self.today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if self.lastday != self.today:
                self.lastday = self.today
                self.refresh = True
        
        if self.refresh == True:
            self.refresh = False
            self.resetstate()
            
        #For processing partial frames to improve performance
        if self.source_fnum % (self.SOURCE_FPS / self.TARGET_FPS) != 0:
            return
        
        self.target_fnum += 1

        # Loop through attached video streams
        streams = self.inputs.video_in.get()
        for stream, tracker in zip(streams, self.trackers):
            self.process_media(stream, tracker)

        #TODO: Currently send only stream 0 to KVS, additional implemendation required to switch stream id by using iot channel
        if self.VIDEO_RECORDING == True:
            self.videowriter.write(streams[self.STREAM_ID].image)
        
        self.outputs.video_out.put(streams)
    
    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def process_media(self, stream, tracker):
        """Runs inference on a frame of video."""
        image_data, ratio = self.preproc(stream.image, self.MODEL_INPUT)

        image_data_torch = torch.unsqueeze(torch.from_numpy(image_data).to(self.device), 0)
        inference_results = self.trt_model(image_data_torch).cpu().detach().numpy()
        
        # Process results (object deteciton)
        num_people = 0
        if len(inference_results) > 0:
            num_people = self.process_results(inference_results, stream, tracker, ratio)            
        
        add_label(stream.image, f"{stream.stream_id} / # People {num_people} / {datetime.utcnow().strftime('%H:%M:%S.%f')[:-5]}", 30, 50)
    
    def process_results(self, inference_results, stream, tracker, ratio):
        boxes, scores, class_indices = self.postprocess(inference_results, self.MODEL_INPUT, ratio)        
        if boxes is None:
            return 0
        
        media_height, media_width, _ = stream.image.shape
        media_scale = np.asarray([media_width, media_height, media_width, media_height])
        
        candidates = []
        for box, score, category_id in zip(boxes, scores, class_indices):
            if category_id in eval(self.CATEGORY):
                w = box[2] - box[0]
                h = box[3] - box[1]
                if w * h < self.args.min_box_area:
                    continue
                horizontal = w / h > self.VERTICAL_RATIO
                if category_id == 0 and horizontal:
                    continue
                candidates.append([box[0], box[1], box[2], box[3], score, category_id])
        
        num_people = len(candidates)
        if num_people == 0:
            return 0
        
        online_targets = tracker.update(self.target_fnum, torch.tensor(candidates))
        jsonlist = []
        ts = stream.time_stamp
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tcid = t.category_id
            tscore = t.score
            age = round((self.target_fnum - t.start_frame)/self.TARGET_FPS, 1)
            jsonlist.append({"Data":f'{{"sid":"{stream.stream_id}","ts":{ts[0] + (0.1) * ts[1]},"fnum":{self.target_fnum},"cid":{tcid},"tid":{tid},"age":{age},"left":{tlwh[0]/self.CAMERA_INPUT[1]},"top":{tlwh[1]/self.CAMERA_INPUT[0]},"w":{tlwh[2]/self.CAMERA_INPUT[1]},"h":{tlwh[3]/self.CAMERA_INPUT[0]}}}'})
            add_rect(stream.image, tlwh[0], tlwh[1], tlwh[2], tlwh[3])
            add_label(stream.image, f'{tid}/{tcid}/{age}', tlwh[0], tlwh[1] - 10)

        num_people = len(jsonlist)
        if num_people == 0:
            return 0
        
        self.firehose.put_record_batch(DeliveryStreamName=self.kinesis_name, Records=jsonlist)
        
        return num_people
        
    def postprocess(self, result, input_shape, ratio):        
        # source: https://github.com/Megvii-BaseDetection/YOLOX/blob/2c2dd1397ab090b553c6e6ecfca8184fe83800e1/demo/ONNXRuntime/onnx_inference.py#L73
        input_size = input_shape[-2:]
        predictions = demo_postprocess(result, input_size)
        predictions = predictions[0] # TODO: iterate through eventual batches
                
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms, score_thr=0.1)
        if dets is None:
            return None, None, None
        
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        boxes = final_boxes
        scores = final_scores
        class_indices = final_cls_inds.astype(int)
        return boxes, scores, class_indices
    
def add_label(image, text, x1, y1):
    # White in BGR
    color = (255, 255, 255)
    # Using cv2.putText() method
    return cv2.putText(image, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def add_rect(image, x1, y1, x2, y2):
    # Red in BGR
    color = (0, 0, 255)        
    return cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + x2), int(y1 + y2)), color, 2)

def get_logger(name=__name__,level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=100000000, backupCount=2)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    while True:
        try:
            logger.info("INITIALIZING APPLICATION")
            app = Application()
            logger.info("PROCESSING STREAMS")
            while True:
                app.process_streams()
        except:
            logger.exception('Exception during processing loop.')
        finally:
            app.stop()
        
        #TODO: What about the failover?
        break
        #time.sleep(10)

logger = get_logger(level=logging.INFO)
main()
