import panoramasdk
import cv2
import numpy as np

import time
import traceback

import boto3
import sinks

from deep_sort import build_tracker
from utils.draw import draw_boxes, letterbox_image, preprocess
from utils.logger import log


# GLOBAL DECLARATIONS
DEFAULT_AWS_DATA_ENDPOINT_URL = "greengrass-ats.iot.us-west-2.amazonaws.com"
DEFAULT_DETECTION_INPUT_SIZE_H = 512
DEFAULT_DETECTION_INPUT_SIZE_W = 512
DEFAULT_TRACKING_INPUT_SIZE_H = 256
DEFAULT_TRACKING_INPUT_SIZE_W = 128
DEFAULT_MAX_DIST = 0.5
DEFAULT_MIN_CONFIDENCE = 0.3
DEFAULT_NMS_MAX_OVERLAP = 0.8
DEFAULT_MAX_IOU_DISTANCE = 0.7
DEFAULT_MAX_AGE = 70
DEFAULT_N_INIT = 5
DEFAULT_NN_BUDGET = 20


class app(panoramasdk.base):
    def interface(self):
    # defines the parameters that interface with other services from Panorama
        return {
                "parameters":
                (
                    ("float", "threshold", "Detection threshold", 0.30),
                    ("model", "people_detection_model", "People detection model", "ssd_512_resnet50_voc_v02"),
                    ("model", "reid_tracking_model", "People re-identification model", "reid_model_v1_v02"),
                    ("int", "detection_input_size_h", "Detection model input size (actual shape will be [1, 3, detection_input_size_h, detection_input_size_w])", DEFAULT_DETECTION_INPUT_SIZE_H),
                    ("int", "detection_input_size_w", "Detection model input size (actual shape will be [1, 3, detection_input_size_h, detection_input_size_w])", DEFAULT_DETECTION_INPUT_SIZE_W),
                    ("int", "tracking_input_size_h", "Tracking model input size (actual shape will be [1, 3, tracking_input_size_h, tracking_input_size_w])", DEFAULT_TRACKING_INPUT_SIZE_H),
                    ("int", "tracking_input_size_w", "Tracking model input size (actual shape will be [1, 3, tracking_input_size_h, tracking_input_size_w])", DEFAULT_TRACKING_INPUT_SIZE_W),
                    ("int", "batch_size", "Model batch size", 1),
                    ("int", "person_class_index", "Person class index based on the model", 14),
                    ("float", "tracking_max_dist", "Euclidean distance separated from the previous distribution", DEFAULT_MAX_DIST),
                    ("float", "tracking_min_confidence", "Confidence threshold", DEFAULT_MIN_CONFIDENCE),
                    ("float", "tracking_nms_max_overlap", "Threshold for non-max supression", DEFAULT_NMS_MAX_OVERLAP),
                    ("float", "tracking_max_iou_distance", "Threshold for Area of Intersection/Area of Union of actual & predicted bounding box", DEFAULT_MAX_IOU_DISTANCE),
                    ("float", "tracking_max_age", "Number of frames to survive before assigning new identity", DEFAULT_MAX_AGE),
                    ("float", "tracking_n_init", "Number of frames before assigning identity or relinquishing it", DEFAULT_N_INIT),
                    ("float", "tracking_nn_budget", "Number of samples", DEFAULT_NN_BUDGET),
                    ("int", "top_y", "Top Y co-ordinate to indicate the entry/exit point", 200),
                    ("int", "bottom_y", "Bottom Y co-ordinate to indicate the entry/exit point", 800),
                    ("string", "aws_iot_endpoint_url", "AWS IoT data endpoint url", DEFAULT_AWS_DATA_ENDPOINT_URL),
                    ("int", "enable_overlay", "Enable video overlay", 1),
                    ("int", "enable_mqtt", "Enable MQTT publication", 0),
                    ("int", "enable_kinesis", "Enable Kinesis Video Stream publication", 0),
                ),
                "inputs":
                (
                    ("media[]", "video_in", "Camera input stream"),
                ),
                "outputs":
                (
                    ("media[video_in]", "video_out", "Camera output stream"),
                ) 
            }


    def init(self, parameters, inputs, outputs):
        # defines the attributes such as arrays and model objects that will be used in the application

        # Detection probability threshold.
        self.threshold = parameters.threshold
        # Input size of the image for detection
        self.detection_input_size_h = parameters.detection_input_size_h
        self.detection_input_size_w = parameters.detection_input_size_w
        # Person class index
        self.person_class_index = parameters.person_class_index
        # Inidicates the top Y co-ordinate entry/exit point
        self.top_y = parameters.top_y
        # Inidicates the bottom Y co-ordinate entry/exit point
        self.bottom_y = parameters.bottom_y
        # AWS IoT data endpoint URL
        self.aws_iot_endpoint_url = 'https://' + parameters.aws_iot_endpoint_url
        # Indicates if panorama sdk overlay is enabled
        self.enable_overlay = parameters.enable_overlay
        # Indicates if MQTT relay is enabled
        self.enable_mqtt = parameters.enable_mqtt 
        # Indicates if Kinesis relay is enabled
        self.enable_kinesis = parameters.enable_kinesis
        # Frame Number Initialization
        self.frame_num = 0
        # Count the number of people entered and exited
        self.enter = set()
        self.exit = set()

        try:  
            log('Loading object detection model')
            start = time.time()
            self.model = panoramasdk.model()
            self.model.open(parameters.people_detection_model, parameters.batch_size)
            log(f'Model loaded in {int(time.time() - start)} seconds')
            log('Object detection model loaded')
            class_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)
            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            self.rect_array = np.empty(rect_info.get_dims(), dtype=rect_info.get_type())

            # Setting up tracking model
            log('Loading object tracking model')
            start = time.time()
            self.tracking_model = panoramasdk.model()
            self.tracking_model.open(parameters.reid_tracking_model, parameters.batch_size)
            log(f'Object Tracking Model loaded in {int(time.time() - start)} seconds')
            # Setting up deep sort
            cfg = {}
            cfg["INPUT_SIZE_H"] = parameters.tracking_input_size_h
            cfg["INPUT_SIZE_W"] = parameters.tracking_input_size_w
            cfg["MAX_DIST"] = parameters.tracking_max_dist
            cfg["MIN_CONFIDENCE"] = parameters.tracking_min_confidence
            cfg["NMS_MAX_OVERLAP"] = parameters.tracking_nms_max_overlap
            cfg["MAX_IOU_DISTANCE"] = parameters.tracking_max_iou_distance
            cfg["MAX_AGE"] = parameters.tracking_max_age
            cfg["N_INIT"] = parameters.tracking_n_init
            cfg["NN_BUDGET"] = parameters.tracking_nn_budget
            self.deepsort = build_tracker(self.tracking_model, cfg, True)
            # Setting up the sinks
            self.sink = sinks.Sink(
                aws_iot_endpoint_url=self.aws_iot_endpoint_url,
                enable_mqtt=self.enable_mqtt, 
                enable_kinesis=self.enable_kinesis)

            return True
        except Exception as e:
            print("Exception: {}".format(e))
            return False


    def entry(self, inputs, outputs):
        # defines the application logic responsible for predicting using the inputs and handles what to do
        # with the outputs
        try:
            for i in range(len(inputs.video_in)):
                stream = inputs.video_in[i]
                original_frame = stream.image
                log(f'Shape: {original_frame.shape}, {original_frame.shape[:2]}')
                self.frame_num += 1
                # if self.frame_num % 3 != 0:
                #     log(f'Skipping frame {self.frame_num}')
                #     return True
                # if self.frame_num >= 150:
                #     self.frame_num = 0

                # Object detection inference
                transformed_image = self.run_inference(original_frame)
                log(f'Transformed Shape: {transformed_image.shape}, {transformed_image.shape[:2]}')
                class_data = self.class_array[0]
                prob_data = self.prob_array[0]
                rect_data = self.rect_array[0]
                # Get Indices of classes that correspond to Person
                person_indices = self.get_persons(class_data, prob_data)
                if person_indices is not None and len(person_indices):
                    class_data = np.compress(person_indices, class_data, axis=0)
                    prob_data = np.compress(person_indices, prob_data, axis=0)
                    rect_data = np.compress(person_indices, rect_data, axis=0)
                    bboxes = []
                    # Now we know the list of people, let's 
                    # do something with each person
                    if class_data is not None and len(class_data):
                        for index in range(len(class_data)):
                            log(f'Confidence: {prob_data[index]}, Left:{rect_data[index][0]} Top: {rect_data[index][1]} Right: {rect_data[index][2]} Bottom: {rect_data[index][3]}')
                        try:
                            outs = self.track(
                                rect_data,
                                prob_data,
                                class_data,
                                original_frame,
                                transformed_image)
                        except Exception as e:
                            traceback.print_exc()

                        # Now we are successfully tracking
                        if outs is not None and len(outs) > 0:
                            bbox_tlwh = []
                            bbox_xyxy = outs[:, :4]
                            identities = outs[:, -3]
                            direction = outs[:, -2:]
                            # Let's draw all the trackers
                            if self.enable_overlay: 
                                draw_boxes(stream.image, bbox_xyxy, identities)
                            # Counter entry/exit
                            for idx in range(len(bbox_xyxy)):
                                bb_xyxy = bbox_xyxy[idx]
                                log(f'BB_XYXY - Left:{bb_xyxy[0]} Top: {bb_xyxy[1]} Right: {bb_xyxy[2]} Bottom: {bb_xyxy[3]}')
                                self.count_enter_exit(identities[idx], north_south=direction[idx][0], east_west=direction[idx][1], top=bb_xyxy[1], bottom=bb_xyxy[3])
                                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                height, width = original_frame.shape[:2]
                log(f'Before rect height:{height} width:{width}')
                left = np.clip(1 / np.float(width), 0, 1)
                top = np.clip(self.top_y / np.float(height), 0, 1)
                right = np.clip((width-1) / np.float(width), 0, 1)
                bottom = np.clip(self.bottom_y / np.float(height), 0, 1)
                stream.add_rect(left, top, right, bottom)
                stream.add_label(f'Enter:{len(self.enter)} Exit: {len(self.exit)}', 0.1, 0.20)
                outputs.video_out[i] = stream

            return True
        except Exception as e:
            log("Exception: {}".format(e))
            return False


    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
    
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        coords = self.clip_coords(coords, img0_shape)
        return coords


    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2
        return boxes


    def run_inference(self, image):
        result = None
        log("Running inference")
        start = time.time()
        try:
            transformed_image = preprocess(
                image, 
                self.detection_input_size_h, 
                self.detection_input_size_w,
                True)
            self.model.batch(0, transformed_image)
            self.model.flush()
            result = self.model.get_result()
        except Exception as ex:
            traceback.print_exc()
        inf_time = time.time() - start
        log(f'Inference completed in {int(inf_time * 1000):,} msec')
        batch_0 = result.get(0)
        batch_1 = result.get(1)
        batch_2 = result.get(2)
        batch_0.get(0, self.class_array)
        batch_1.get(0, self.prob_array)
        batch_2.get(0, self.rect_array)
        self.model.release_result(result)

        return transformed_image


    def get_persons(self, class_data, prob_data):
        # Returns an array the size of class data
        # Default false unless the identified object
        # is a person with probablity greater than 
        # the threshold
        person_indices = None
        number_of_people = 0
        if class_data is not None and len(class_data):
            person_indices = np.full(len(class_data), False)
            for x in range(len(class_data)):
                if int(class_data[x]) == self.person_class_index and prob_data[x] >= self.threshold:
                    person_indices[x] = True
                    number_of_people += 1
        log(f'Number of people detected: {number_of_people}')
        return person_indices


    def track(self, rect_data, prob_data, class_data, original_frame, transformed_image):
        # do tracking
        outputs = None
        start = time.time()
        try:
            log(f'Transformed image shape {transformed_image.shape}/{transformed_image.shape[2:]}')
            log(f'Original image shape {original_frame.shape}/{original_frame.shape[:2]}')
            start_scale = time.time()
            rect_data = self.scale_coords(
                transformed_image.shape[2:], 
                rect_data, 
                original_frame.shape[:2])
            rect_data = self.xyxy_to_xywh(rect_data)
            log(f'Scaling and conversion completed in {int((time.time() - start_scale) * 1000)} msec')
            outputs = self.deepsort.update(
                rect_data, 
                prob_data, 
                original_frame)
        except Exception as ex:
            traceback.print_exc()
        end_time = time.time() - start
        log(f'Deep sort completed in {int(end_time * 1000)} msec')
        return outputs


    def xyxy_to_xywh(self, boxes_xyxy):
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        for index in range(len(boxes_xywh)):
            log(f'X:{boxes_xywh[index][0]} Y: {boxes_xywh[index][1]} W: {boxes_xywh[index][2]} H: {boxes_xywh[index][3]}')
        return boxes_xywh


    def count_enter_exit(self, track_id, north_south, east_west, top, bottom):
        # North = 1
        # South = 2
        # East = 3
        # West = 4
        if north_south == 1:
            if top > self.bottom_y:
                self.enter.add(track_id)
            if top > self.top_y:
                self.exit.add(track_id)
        else: 
            # if north_south == 2
            if bottom > self.top_y:
                self.enter.add(track_id)
            if bottom > self.bottom_y:
                self.exit.add(track_id)

        if self.enable_mqtt:
            self.sink.sink(len(self.enter), len(self.exit))
        log(f'Enter: {len(self.enter)}, Exit: {len(self.exit)}')


def main():
    app().run()


main()


