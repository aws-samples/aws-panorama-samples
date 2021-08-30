# Imports
from __future__ import division
import panoramasdk
import cv2
import numpy as np
import urllib
import boto3
import os
import math 
import datetime
from scipy import misc


# SD based imports
import ModelOutput as jm
import socialDistance as sd
import socialDistanceUtils as sdu

red    = (0,0,255)
green  = (0,255,0)
black  = (0,0,0)
white  = (255,255,255)
text_color = black

num_frames = 1000000000
mask_frequency = 10

class AwspanoramaSD(panoramasdk.base):
    def interface(self):
        return {
            "parameters":
            (
                ("float", "threshold",      "Detection threshold",        0.1),
                ("model", "people_counter", "Model for people counting", "SSD-VOC-Orig"),
                ("int",   "batch_size",     "Model batch size",           1),
                ("float", "person_index", "person index based on dataset used", 14)
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
        """
        This is the init method where you can declare variables that will be used in this class and initialize any objects if necessary.

        Args: 
            Input parameters from the application configuration on the console.
            Input stream object that is created and passed in from mediapipeline.
            Output stream object that send data stream to panoramasdk data sink.
        Returns:
            Boolean.
        """
        try:
            # Detection probability threshold.
            self.threshold = parameters.threshold
            self.person_index = parameters.person_index
            
            self.boxes = []
            self.frame_num = 0
            self.send_images = True
            
            # SD Code 2
            self._batch_frame_count = [0] * len(inputs.video_in)
            self._frame_count = [0] * len(inputs.video_in)
            self._cam_standing_people = [[]] * len(inputs.video_in)
            self._curr_refs = [[]] * len(inputs.video_in)
            self._total_size_mask_count = [0] * len(inputs.video_in)
            self._cam_left = [-1] * len(inputs.video_in)
            self._size_mask = [[]] * len(inputs.video_in)
            
            # Load model from the specified directory.
            self.model = panoramasdk.model()
            self.model.open(parameters.people_counter, 1)

            class_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)

            self.class_array = np.empty(class_info.get_dims(), dtype=class_info.get_type())
            self.prob_array = np.empty(prob_info.get_dims(), dtype=prob_info.get_type())
            self.rect_array = np.empty(rect_info.get_dims(), dtype=rect_info.get_type())
            return True
        except Exception as e:
            print("Exception: {}".format(e))
            return False
            
    def preprocess(self, img):
        
        resized = cv2.resize(img, (512, 512))
        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]  # RGB
        img = resized.astype(np.float32) / 255.  # converting array of ints to floats
        img_a = img[:, :, 0]
        img_b = img[:, :, 1]
        img_c = img[:, :, 2]
        # Extracting single channels from 3 channel image
        # The above code could also be replaced with cv2.split(img) << which will return 3 numpy arrays (using opencv)
        # normalizing per channel data:
        img_a = (img_a - mean[0]) / std[0]
        img_b = (img_b - mean[1]) / std[1]
        img_c = (img_c - mean[2]) / std[2]
        # putting the 3 channels back together:
        x1 = [[[], [], []]]
        x1[0][0] = img_a
        x1[0][1] = img_b
        x1[0][2] = img_c
        x1 = np.asarray(x1)
        return x1
      
    def blur_bounding_box(self, person_image, bbox,sigma = 3.5): 
        img = person_image
        img_arr = img.copy()
        img_w, img_h = img.shape[1],img.shape[0]
        nominal_box_area = img_h * img_w * 0.1
        blur_sigma = sigma
        for boxes in bbox:
            try:
                xmin, ymin, xmax, ymax = [int(x) for x in boxes]
                print('BBox input into function {}'.format([xmin, ymin, xmax, ymax]))
                ymin = int(((ymin/512.0)*img_h))
                xmin = int(((xmin/512.0)*img_w))
                ymax = int(((ymax/512.0)*img_h))
                xmax = int(((xmax/512.0)*img_w))
                print('BBox input into function after scaling {}'.format([xmin, ymin, xmax, ymax]))    
                # blurring 
                box_area = (xmax - xmin) * (ymax - ymin)
                sigma_scaled = blur_sigma * math.sqrt(box_area/nominal_box_area)
                sigma_clamped = min(5.5, max(1.0, sigma_scaled))
                bbox_img = img_arr[ymin:ymax, xmin:xmax, :]
                img_arr[ymin:ymax, xmin:xmax, :] = cv2.GaussianBlur(bbox_img, (0, 0), sigma_clamped)
            except Exception as e:
                pass
        return img_arr
        
    
    def get_number_persons(self, class_data, prob_data):
        # get indices of people detections in class data
        person_indices = [i for i in range(len(class_data)) if int(class_data[i]) == self.person_index]
        # use these indices to filter out anything that is less than 95% threshold from prob_data
        prob_person_indices = [i for i in person_indices if prob_data[i] >= self.threshold]
        return prob_person_indices

    def different_enough(self, a, b):
        try:
            a_bb = a['BoundingBox']
            b_bb = b['BoundingBox']
        
            h_diff = abs(a_bb['Height'] - b_bb['Height'])
            w_diff = abs(a_bb['Width']  - b_bb['Width'])
            t_diff = abs(a_bb['Top']    - b_bb['Top'])
            l_diff = abs(a_bb['Left']   - b_bb['Left'])
        
            total_other_diff = h_diff + w_diff + l_diff
            if (t_diff > 0.10) and (total_other_diff > 0.30):
                return True
            else:
                return False
        except Exception as e:
            print('Different enough exception is {}'.format(e))
            return False
    
    def add_distinct_people(self, all_people, new_people):
        if len(all_people) == 0:
            return new_people 
        else:
            tmp_people = all_people.copy()
            for np in new_people:
                is_diff = False
                for ap in all_people:
                    is_this_one_diff = self.different_enough(np, ap)
                    if is_this_one_diff:
                        is_diff = True
                        break
                if is_diff:
                    tmp_people.append(np)
            return tmp_people


    def entry(self, inputs, outputs):
        self.frame_num += 1
        _cam_height = 20
        cam_order = {}

        for i in range(len(inputs.video_in)):
            cam_order[i] = 'camera_number_' + str(i+1)

        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
        
            if self.send_images == True:

                person_image = stream.image
                sending_image = person_image.copy()
                redacted_image = person_image.copy()
                
                # SD Code 1
                _frame = person_image
                _image_shape, _size_mask_shape = sdu.get_shapes(_frame)
                           
                x1 = self.preprocess(person_image)
                    
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
                
                
                class_data2 = self.class_array
                prob_data2 = self.prob_array
                rect_data2 = self.rect_array               
                
                person_indices = self.get_number_persons(class_data,prob_data)
                                    
                # find people in this frame
                _img_all_people, _img_standing_people = sdu.get_standing_people(_frame, _image_shape, _size_mask_shape, x1, class_data2, prob_data2, rect_data2)
                _num_total_people = len(_img_all_people)
                _min_distance = sd.MAX_DISTANCE
                _safe_cat = 'LessThanTwoPeople'
               
                # every N seconds, if we still need a better size mask, try to improve it
                if self._batch_frame_count[i] < mask_frequency:
                    self._batch_frame_count[i] += 1
                elif (len(self._cam_standing_people[i]) < sdu.MAX_STANDING_REFS_NEEDED) or (self._total_size_mask_count[i] == 0):
                    self._batch_frame_count[i] = 0
                    # 1. add to cumulative list of standing people for this camera
                    print('encountered add_distict_people')
                    self._cam_standing_people[i] = self.add_distinct_people(self._cam_standing_people[i], _img_standing_people)
                    print('Distinct standing people so far {}'.format(len(self._cam_standing_people[i]))) 
                    # 2. once we have at least MIN_STANDING_REFS_PER_CAM people in our running list
                    if (len(self._cam_standing_people[i]) > sdu.MIN_STANDING_REFS_PER_CAM):
                        # 2a. generate the best size mask for the standing people thus far
                        _curr_camera_config, _curr_best_rmse, _curr_size_mask_count = \
                            sdu.gen_best_size_mask(self._cam_standing_people[i], _image_shape, _size_mask_shape)
                                                
                        if _curr_size_mask_count > 0:
                            self._curr_refs[i] = _curr_camera_config['MaskReferenceSizes']
                        self._total_size_mask_count[i] += _curr_size_mask_count
            
                        print('total masks: {}, latest rmse: {}, refs: {}'.format(self._total_size_mask_count[i],_curr_best_rmse,self._curr_refs[i]))
                        if _curr_size_mask_count > 0:
                            self._cam_left[i]   = _curr_camera_config['CameraLeft']
                            _cam_height = _curr_camera_config['CameraHeight']
                            self._size_mask[i] = np.asarray(_curr_camera_config['SizeMask'])
                   
                
                if (self._total_size_mask_count[i] > 0):
                    _verbose = False
                    _likely_people, _proximity_list = sd.detect_distances(_img_all_people, self._size_mask[i], _image_shape,
                                                                          _cam_height, _verbose)
                    _min_distance     = sdu.min_distance_from_list(_proximity_list)
                    _num_unsafe_pairs = sdu.get_num_unsafe_pairs(_proximity_list)
                    _num_total_people = len(_likely_people)
                    
                    if _num_total_people < 2:
                        _safe_cat = 'LessThanTwoPeople'
                    elif _min_distance > sdu.MIN_SAFE_DISTANCE:
                        _safe_cat = 'AppropriateDistance'
                    else:
                        _safe_cat = 'ReducedDistance'
                    
                    print('[_likely_people, _proximity_list] is {}'.format([_likely_people, _proximity_list]))
                        
                else:
                    _proximity_list = [] 
                    self._curr_refs[i]      = []
                    _likely_people  = []
                    self._curr_refs[i].append({'HeightImageRatio': 0, 'AspectRatio': 0, 'FromImage': '', 'GridPos': [0,0]})
                    self._curr_refs[i].append({'HeightImageRatio': 0, 'AspectRatio': 0, 'FromImage': '', 'GridPos': [0,0]})
                    
                    
                # update saftey banner at bottom of output stream
                #   green banner if safe, red if unsafe, count of people in the frame
                if _safe_cat in ['LessThanTwoPeople', 'AppropriateDistance']:
                    color = green
                    text_color = black
                else:
                    color = red
                    text_color = white
                     
                # Draw a color-coded banner at the bottom of the frame showing safety, num people, min distance
                cv2.rectangle(_frame, (0, _frame.shape[0] - 30), (_frame.shape[1], _frame.shape[0]), color, -1)
                if (_min_distance >= sd.MAX_DISTANCE):
                    banner_text = '{} people'.format(_num_total_people)
                else:
                    banner_text = '{}, {} people, Min: {} ft'.format(_safe_cat, _num_total_people, _min_distance)
            
                cv2.putText(_frame, banner_text, 
                            (20, _frame.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, #SIMPLEX, 
                            1, text_color, 1, cv2.LINE_AA)
            
                
                print('_likely_people {}'.format(_likely_people))
                print('_proximity_list {}'.format(_proximity_list))
                print('self._curr_refs[i] {}'.format(self._curr_refs[i]))
                print('_size_mask_shape {}'.format(_size_mask_shape))
            
                # overlay color-coded bounding boxes on output stream, including yellow boxes for reference people
                sdu.add_bboxes(_frame, _likely_people, _proximity_list, self._curr_refs[i], _size_mask_shape)
                
                self._frame_count[i] += 1

                self.boxes = []
                if len(person_indices) > 0:
                    # get index of only people from the class list
                    for index in person_indices:
                               
                        left   = rect_data[index][0]
                        top    = rect_data[index][1]
                        right  = rect_data[index][2]
                        bottom = rect_data[index][3]
                            
                        boxes = [left,top,right,bottom]
                        self.boxes.append(boxes)

                        img_w, img_h = person_image.shape[1],person_image.shape[0]
                        xmin, ymin, xmax, ymax = [int(x) for x in boxes]
                        #print('BBox input into function {}'.format([xmin, ymin, xmax, ymax]))
                        left = int(((left/512.0)*img_w))
                        top = int(((top/512.0)*img_h))
                        right = int(((right/512.0)*img_w))
                        bottom = int(((bottom/512.0)*img_h))
                    
                self.model.release_result(resultBatchSet)
                
            outputs.video_out[i] = stream
        return True

def main():
    AwspanoramaSD().run()

main()