from __future__ import division
import numpy as np
import cv2

def get_mobilenet_people_from_frame(frame, x1, class_data, prob_data, rect_data):
 
  #class_IDs, scores, bounding_boxes = net1(mx.nd.array(x1))
  dims       = x1.shape
  #print('dims {}'.format(dims))
    
  class_IDs = class_data
  scores = prob_data
  bounding_boxes = rect_data
  
  img_height = dims[2]
  img_width  = dims[3]
  
  try:
    people = []
    for i in range(len(scores[0])):
        #print(class_IDs[0][i][0])
        if class_IDs[0][i][0] == 14:
            bbox2 = np.array(bounding_boxes[0][i])
            
            left, top, right, bottom = bbox2[0],bbox2[1],bbox2[2],bbox2[3]
            left, top, right, bottom = max(left,0), max(top,0), max(right,0), max(bottom,0)
    
            height = bottom - top
            width = right - left
   
            height = height / img_height
            width = width / img_width
            top = top / img_height
            left = left / img_width
  
            dict1 = {'Confidence':scores[0][i][0] * 100,'BoundingBox':{'Height':height,'Width':width,'Top':top,'Left':left}}
            people.append(dict1)
  except Exception as e:
    print('Exception is {}'.format(e))
    
  return people