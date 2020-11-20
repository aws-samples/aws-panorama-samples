from __future__ import division
from __future__ import print_function

import panoramasdk
import numpy as np
import time
from utils import update_x, update_y, update_ids, crop_resize_normalize, preprocess, upscale_bbox_fn, \
    detector_to_simple_pose, get_max_pred, heatmap_to_coord, reset_counts, reset_tracker, fall_detection


class fall_detector(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("float", "conf_thresh", "Detection threshold", 0.10),
                    ("model", "object_detector", "Model for detecting people", "ssd-coco"),
                    ("model", "pose_model", "Model for pose estimation", "pose-net2"),
                    ("int", "batch_size", "Model batch size", 1),

                    ("int", "img_size", "img size", 512),
                    ("float", "person_index", "person index based on dataset used", 0),
                    ("int", "box_size_thresh", "min bbox dimension", 20),

                    ("int", "min_non_dets", "reset trackers after no detection", 20),
                    ("int", "anks_shdr_thresh", "min ankle-shoulder dist", 10),
                    ("int", "dist_count", "min frame count for low anks-shdr distance", 5),
                    ("int", "dist_hist", "min ankle-shoulder dist", 50),
                    ("int", "fall_interval", "number of frames to skip, to detect next fall", 1000)
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
        try:
            ### data parameters
            self.img_size = (parameters.img_size, parameters.img_size)

            # Detection probability threshold.
            self.conf_thresh = parameters.conf_thresh
            # Number of People
            self.number_people = 0
            # Person Index for Model
            self.person_index = parameters.person_index

            ### Fall parameters
            self.min_non_dets = parameters.min_non_dets
            self.box_size_thresh = (parameters.box_size_thresh, parameters.box_size_thresh)
            self.anks_shdr_thresh = parameters.anks_shdr_thresh
            self.dist_hist = parameters.dist_hist
            self.dist_count = parameters.dist_count
            self.fall_interval = parameters.fall_interval
            self.fall_time = -1

            # Load model from the specified directory.
            print("loading the model...")
            self.model = panoramasdk.model()
            self.model.open(parameters.object_detector, 1)
            print("Detector loaded")
            self.pose_model = panoramasdk.model()
            self.pose_model.open(parameters.pose_model, 1)

            print("Pose model loaded")
            # Create input and output arrays.
            class_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)

            # Create pose output arrays
            heatmap_info = self.pose_model.get_output(0)

            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            self.rect_array = np.empty(rect_info.get_dims(), dtype=rect_info.get_type())
            self.heatmaps_array = np.empty(heatmap_info.get_dims(), dtype=heatmap_info.get_type())

            # Fall tracking variables
            self.xpart_tracker, self.ypart_tracker = reset_tracker()
            self.frame_num, self.frame_prev, self.frame_curr, self.zero_dets = reset_counts()
            self.fall_idx = -1

            self.master_idx = -1

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def get_person_data(self, class_data, prob_data, rect_data):
        # get indices of people detections in class data
        person_indices = [i for i in range(len(class_data)) if int(class_data[i]) == self.person_index]
        # filter detections below the confidence threshold
        prob_person_indices = [i for i in person_indices if prob_data[i] >= self.conf_thresh]
        return prob_person_indices, class_data[person_indices], prob_data[person_indices], rect_data[person_indices]

    def entry(self, inputs, outputs):
        self.master_idx += 1

        for i in range(len(inputs.video_in)):

            stream = inputs.video_in[i]
            person_image = stream.image
            x1, orig_img = preprocess(person_image, output_size=self.img_size)

            # Do inference on the new frame.
            self.model.batch(0, x1)
            self.model.flush()

            # Get the results.
            resultBatchSet = self.model.get_result()
            class_batch = resultBatchSet.get(0)
            prob_batch = resultBatchSet.get(1)
            rect_batch = resultBatchSet.get(2)
            class_batch.get(0, self.class_array)
            prob_batch.get(0, self.prob_array)
            rect_batch.get(0, self.rect_array)
            class_data = self.class_array[0]
            prob_data = self.prob_array[0]
            rect_data = self.rect_array[0]

            self.model.release_result(resultBatchSet)
            x_min, y_min = rect_data[0][0], rect_data[0][1]
            x_max, y_max = rect_data[0][2], rect_data[0][3]
            w, h = x_max - x_min, y_max - y_min
            try:
                # Filter predictions of only "person-class"
                person_indices, class_data, prob_data, rect_data = self.get_person_data(class_data, prob_data,
                                                                                        rect_data)
            except Exception as e:
                print("Exception: {}".format(e))
            try:
                self.number_people = len(person_indices)
            except:
                self.number_people = 0

            # Draw Bounding Boxes
            if (self.number_people > 0) and (w > self.box_size_thresh[0]) and (h > self.box_size_thresh[1]):
                try:
                    # Crop the bbox area from detector output from original image, transform it for pose model
                    pose_input, upscale_bbox = detector_to_simple_pose(orig_img, class_data[None, :, :][:, 0:1, :],
                                                                       prob_data[None, :, :][:, 0:1, :],
                                                                       rect_data[None, :, :][:, 0:1, :],
                                                                       person_index=self.person_index,
                                                                       thr=self.conf_thresh)
                    if len(pose_input) > 0:
                        self.pose_model.batch(0, pose_input)
                        self.pose_model.flush()
                        # Get the results.
                        PresultBatchSet = self.pose_model.get_result()
                        heatmaps_batch = PresultBatchSet.get(0)
                        heatmaps_batch.get(0, self.heatmaps_array)
                        predicted_heatmap = self.heatmaps_array[0]
                        self.pose_model.release_result(PresultBatchSet)
                        # process pose model output to get key point coordinates
                        pred_coords, confidence = heatmap_to_coord(predicted_heatmap[None, :, :, :], upscale_bbox)
                        pred_coords = np.round(pred_coords, 3)

                        self.xpart_tracker = update_x(pred_coords[0][:, 0], self.xpart_tracker)
                        self.ypart_tracker = update_y(pred_coords[0][:, 1], self.ypart_tracker)

                        result = fall_detection(self.ypart_tracker, self.anks_shdr_thresh, self.dist_hist,
                                                self.dist_count)

                        if result:
                            # Flag next fall after fall_interval frames
                            if self.fall_idx == -1 or (self.master_idx - self.fall_idx) >= (self.fall_interval):
                                self.fall_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
                                print('Fall Detected at : {}'.format(self.fall_time))
                                self.fall_idx = self.master_idx

                except Exception as e:
                    print('Pose model exception')
                    print("Exception: {}".format(e))

            else:
                ### Reset tracker if no person is detected for more than `min_non_dets` continuous frames
                if self.zero_dets > self.min_non_dets:
                    self.xpart_tracker, self.ypart_tracker = reset_tracker()
                    self.frame_num, self.frame_prev, self.frame_curr, self.zero_dets = reset_counts()

                    outputs.video_out[i] = stream
                    continue

                # Track consecutive non detections
                self.frame_prev, self.frame_curr = self.frame_curr, self.frame_num
                if self.frame_curr - self.frame_prev == 1:
                    self.zero_dets += 1
                else:
                    self.zero_dets = 0

            if len(person_indices) > 0:
                # currently single person fall detector, choosing the top prediction.
                index = 0
                left = np.clip(rect_data[index][0] / np.float(512), 0, 1)
                top = np.clip(rect_data[index][1] / np.float(512), 0, 1)
                right = np.clip(rect_data[index][2] / np.float(512), 0, 1)
                bottom = np.clip(rect_data[index][3] / np.float(512), 0, 1)

                stream.add_rect(left, top, right, bottom)
                stream.add_label(str(prob_data[index][0]), left, top)
                stream.add_label('Current Frame : ' + str(self.master_idx), 0.1, 0.1)
                stream.add_label('Fall frame : ' + str(self.fall_idx), 0.1, 0.15)
                stream.add_label('Last Fall at : ' + str(self.fall_time), 0.1, 0.2)

            outputs.video_out[i] = stream
            self.frame_num += 1

        return True


def main():
    fall_detector().run()


main()