from __future__ import division
import json
import boto3
import math 
import sys
import numpy as np 
import cv2
# TODO: kill off use of Pillow. CV2 can do the job
#from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import io
import os
#from sklearn.metrics import mean_squared_error

import socialDistance  as sd
import ModelOutput as jm

MIN_STANDING_REFS_PER_CAM = 2
MAX_STANDING_REFS_NEEDED  = 10
MIN_SAFE_DISTANCE         = 6.0

MAX_LABELS = 50

TEMP_PHOTO_FILENAME = 'tmp.jpg'

BBOX_SAFE_COLOR   = (0,255,0)   #'#00d400'
BBOX_UNSAFE_COLOR = (0,0,255)   #'#FF0000'
BBOX_NUMBER_COLOR = (255,255,255) #(0,0,0)     #'#00d400'
STANDING_COLOR    = (255,255,0) #'#FFFF00'

BBOX_COLOR_P        = '#00d400'
BBOX_NUMBER_COLOR_P = '#00d400'
BBOX_HEIGHT_COLOR_P = '#FF0000'
BBOX_WIDTH = 2
BBOX_NUMBER_OFFSET = 3

# def get_people_from_frame(frame):
#     img_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
#     rek_client=boto3.client('rekognition')
#     response = rek_client.detect_labels(Image={'Bytes':img_bytes}, MaxLabels=MAX_LABELS)
#     people = []
#     for label in response['Labels']:
#         if label['Name'] == 'Person':
#             people = label['Instances']

#     return people


def mean_squared_error(actuals, preds):
    return np.square(np.subtract(actuals,preds)).mean() 



def on_edge(pxTop, pxHeight, imgHeight):
    if pxTop <= 4:
        return True 
    elif (imgHeight - (pxTop + pxHeight)) <= 4:
        return True 
    else:
        return False

def get_standing_people(frame, image_shape, size_mask_shape, x1, class_data, prob_data, rect_data):
    all_people     = jm.get_mobilenet_people_from_frame(frame, x1, class_data, prob_data, rect_data)
    #all_people_mn  = jm.get_mobilenet_people_from_frame(frame)
    
    #print('all people len is {}'.format(len(all_people)))

    #print(f'    Rek: {len(all_people)}, MobileNetV2: {len(all_people_mn)}')

#    print(json.dumps(all_people_rek, indent=4, sort_keys=True))
#    print(json.dumps(all_people, indent=4, sort_keys=True))

    standing_people = []
    
    row_height = 1 / size_mask_shape[0]
    col_width  = 1 / size_mask_shape[1]


    for i in range(len(all_people)):
        
        conf = all_people[i]['Confidence']
    
        h = all_people[i]['BoundingBox']['Height']
        w = all_people[i]['BoundingBox']['Width']
        t = all_people[i]['BoundingBox']['Top']
        l = all_people[i]['BoundingBox']['Left']

#             print('Height is {}'.format(h))
#             print('Width is {}'.format(w))
#             print('Top is {}'.format(t))
#             print('Left is {}'.format(l))
#             print('Row Height is {}'.format(row_height))
#             print('Col width is {}'.format(col_width))

        row = int(t / row_height)  
        col = int(l / col_width)

        imgHeight = image_shape[1]
        imgWidth  = image_shape[0]

        pxTop    = t * imgHeight
        pxHeight = h * imgHeight
        pxWidth  = w * imgWidth
        pxAspect_ratio = pxHeight / pxWidth

        if (sd.likely_standing(conf, pxHeight, pxWidth)) and not on_edge(pxTop, pxHeight, imgHeight):
            all_people[i]['GridPos']     = [row, col]
            all_people[i]['AspectRatio'] = np.around(pxAspect_ratio, 3)
            all_people[i]['FromImage']   = ''
            standing_people.append(all_people[i])
            #print(f'  IS LIKELY STND,  pxAsp: {pxAspect_ratio:.2f}, h:{h:.2f}, w:{w:.2f}, c:{conf:.1f}, tl:[{t:.2f},{l:.2f}]')
            #print('IS LIKELY STND,  pxAsp: {}, h:{}, w:{}, c:{}, tl:[{},{}]'.format(pxAspect_ratio, h, w, conf, t, l))
        

        else:
            continue
            #print('NOT LIKELY STND, pxAsp: {}, h:{}, w:{}, i_h:{}, i_w:{}, c:{}'.format(pxAspect_ratio, h, w, image_shape[1], image_shape[0], conf))
            #print(f'  NOT LIKELY STND, pxAsp: {pxAspect_ratio:.2f}, h:{h:.2f}, w:{w:.2f}, i_h:{image_shape[1]}, i_w:{image_shape[0]}, c:{conf:.1f}')
    return all_people, standing_people

def get_people(bucket, photo, bbox_dir):
    # first try the cache
    rek_client=boto3.client('rekognition')

    found_in_cache = False
    people = []
    try:
        photo = photo.split('/')[2]
        fn = photo.replace('.jpg', '.json')
        bbox_json_fn = '{bbox_dir}/{fn}'.format(bbox_dir,fn)
        with open(bbox_json_fn) as f:
            bboxes_json   = json.load(f)
        people_json = bboxes_json['BoundingBoxes']
        found_in_cache = True
    except:
        #print(f'*****ERRROR: failed to open {bbox_json_fn}')
        pass

    # if no cache file, use Rekognition
    if not found_in_cache:
        print('Did NOT find people in cache for {}'.format(photo))
        response = rek_client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':photo}},
                                                        MaxLabels=MAX_LABELS)
        people = []
        for label in response['Labels']:
            if label['Name'] == 'Person':
                people = label['Instances']
        people_json = {'BoundingBoxes': {'People': people}}

    return people_json

def sort_ref_sizes(ref_list):
    return sorted(ref_list, key = lambda i: np.around(i['height'], 3)) 

def sort_people_by_ascending_height(people):
    return sorted(people, key = lambda i: np.around(i['BoundingBox']['Height'], 3)) 

def eval_size_mask2(size_mask, standing_people):
    np_size_mask = np.asarray(size_mask)
    actuals = [standing_people[s]['BoundingBox']['Height'] for s in range(len(standing_people))]
    preds   = []
    for i in range(len(standing_people)):
        which_row = standing_people[i]['GridPos'][0]
        which_col = standing_people[i]['GridPos'][1]
        preds.append(np_size_mask[which_row, which_col])
    try:
        rmse = math.sqrt(mean_squared_error(actuals, preds))
    except:
        rmse = sd.MAX_DISTANCE
        pass
    return rmse

def eval_size_mask(size_mask, ref_size_list):
    np_size_mask = np.asarray(size_mask)
    actuals = [ref_size_list[s]['height'] for s in range(len(ref_size_list))]
    preds   = []
    for i in range(len(ref_size_list)):
        which_row = ref_size_list[i]['grid_pos'][0]
        which_col = ref_size_list[i]['grid_pos'][1]
        preds.append(np_size_mask[which_row, which_col])
    try:
        rmse = math.sqrt(mean_squared_error(actuals, preds))
    except:
        rmse = sd.MAX_DISTANCE
        pass
    return rmse


# def overlay_people_height(photo, bucket, people, ref_list, out_fn):
#     s3_connection = boto3.resource('s3')
#     s3_object     = s3_connection.Object(bucket, photo)
#     s3_response   = s3_object.get()

#     stream = io.BytesIO(s3_response['Body'].read())
#     image  = Image.open(stream)
    
#     imgWidth, imgHeight = image.size  
#     draw = ImageDraw.Draw(image)  
#     font = ImageFont.truetype('/usr/share/fonts/default/Type1/a010013l.pfb', 24)
# #    font = ImageFont.truetype('Arial Narrow Bold.ttf', 24)
# #    height_font = ImageFont.truetype('Arial Narrow Bold.ttf', 24)
#     height_font = font
    
#     for p in range(len(people)):
#         r = ref_list[p]

#         box    = people[p]['BoundingBox']
#         left   = imgWidth  * box['Left']
#         top    = imgHeight * box['Top']
#         width  = imgWidth  * box['Width']
#         height = imgHeight * box['Height']
#         conf   = people[p]['Confidence']

#         height_ratio = r['height']

#         points = (
#             (left, top),
#             (left + width, top),
#             (left + width, top + height),
#             (left , top + height),
#             (left, top)
#         )
#         draw.line(points, fill=BBOX_COLOR_P, width=BBOX_WIDTH)
#         # if there is room for adding conf on top, do so
#         if top > 0.05:
#             conf_text = '{}'.format(np.round(conf,1))
#             draw.text((left, top - (8*BBOX_NUMBER_OFFSET)), 
#                       conf_text, font=font, fill=BBOX_NUMBER_COLOR)
#         draw.text((left + BBOX_NUMBER_OFFSET, top + BBOX_NUMBER_OFFSET), 
#                   str(p), font=font, fill=BBOX_NUMBER_COLOR_P)
                   
#         height_ratio_text = '{}'.format(np.round(height_ratio,2))
        
#         draw.text((left + BBOX_NUMBER_OFFSET, top + (BBOX_NUMBER_OFFSET * 8)), 
#                   height_ratio_text, font=height_font, fill=BBOX_HEIGHT_COLOR_P)

#     # Alternatively can draw rectangle. However you can't set line width.
#     #draw.rectangle([left,top, left + width, top + height], outline='#00d400') 

#     image.save(out_fn)
#     del image
#     return

# def get_image_shape_from_s3(photo, bucket):
#     s3_connection = boto3.resource('s3')
#     s3_object     = s3_connection.Object(bucket, photo)
#     s3_response   = s3_object.get()

#     stream = io.BytesIO(s3_response['Body'].read())
#     image  = Image.open(stream)

#     shape = image.size
#     del image

#     return shape

def suggest_size_mask_shape(img_shape, multiple=1):
    aspect_ratio = img_shape[0] / img_shape[1]
#    print(f'raw image was shape: {img_shape}, aspect ratio: {aspect_ratio:.1f}')
    rows = 6
    cols = int(round(aspect_ratio * rows))
    shape = [rows * multiple, cols * multiple]
#    print(f'suggesting size mask shape: {shape}')
    return shape

def get_ref_list(people, size_mask_shape, image_shape):
    likely_people = sd.keep_likely_people(people, image_shape)

    row_height = 1 / (size_mask_shape[0])
    col_width  = 1 / (size_mask_shape[1])

    ref_list = []

    for which_person in range(len(likely_people)):
        person   = likely_people[which_person]
        top_pos  = person['BoundingBox']['Top']
        left_pos = person['BoundingBox']['Left']
        height   = person['BoundingBox']['Height']
        width    = person['BoundingBox']['Width']
        conf     = person['Confidence']

        row = int(top_pos / row_height)  
        col = int(left_pos / col_width)
        
        pxHeight = height * image_shape[1]
        pxWidth  = width * image_shape[0]
        asp = np.around(pxHeight / pxWidth, 2)

        ref_list.append({'person': which_person, 'grid_pos': [row, col], 
                         'height': height, 'width': width, 'confidence': conf,
                         'top_left': [top_pos, left_pos], 'width': width, 
                         'aspect_ratio': asp})
        which_person += 1
    return likely_people, ref_list 

def get_shapes(f):
    _image_shape = (f.shape[1], f.shape[0])
    size_mask_shape = suggest_size_mask_shape(_image_shape, multiple=2)
    return _image_shape, size_mask_shape

def draw_bbox(frame, bbox, imgWidth, imgHeight):
#     print('calling draw bbox')

    h = bbox['Height']
    w = bbox['Width']
    t = bbox['Top']
    l = bbox['Left']

    pxLeft   = int(imgWidth  * l)
    pxTop    = int(imgHeight * t)
    pxHeight = int(imgHeight * h)
    pxWidth  = int(imgWidth  * w)

    #print('[pxLeft, pxTop, pxLeft + pxWidth, pxTop + pxHeight] is {}'.format([pxLeft, pxTop, pxLeft + pxWidth, pxTop + pxHeight]))

    #cv2.rectangle(frame, (pxLeft, pxTop), (pxLeft + pxWidth, pxTop + pxHeight), STANDING_COLOR, 5)
    return

def add_bboxes(frame, people, proximity_list, camera_refs, size_mask_shape):
    
    #print(len(people))
    
    if len(people) < 1:
        return

    imgWidth  = frame.shape[1]
    imgHeight = frame.shape[0]  

    # display bounding boxes for the 2 ref people, to help with debugging size masks
    draw_bbox(frame, camera_refs[0]['BoundingBox'], imgWidth, imgHeight)
    draw_bbox(frame, camera_refs[1]['BoundingBox'], imgWidth, imgHeight)


    row_height = 1 / (size_mask_shape[0])
    col_width  = 1 / (size_mask_shape[1])

    ref0    = camera_refs[0]['GridPos'] 
    ref0_ht = np.around(camera_refs[0]['HeightImageRatio'], 3)

    ref1    = camera_refs[1]['GridPos']
    ref1_ht = np.around(camera_refs[1]['HeightImageRatio'], 3)

#    print(f'REFS -- r0:{ref0}:{ref0_ht},r1:{ref1}:{ref1_ht}')

    #    print(f'num people: {len(people)}, proxes: {len(proximity_list)}')
    for p in range(len(people)):
        prox = proximity_list[p]
#        print(f'prox property list len:{len(prox)}')
        if prox[2] > MIN_SAFE_DISTANCE:
            bbox_color = BBOX_SAFE_COLOR
        else:
            bbox_color = BBOX_UNSAFE_COLOR

        conf = people[p]['Confidence']
        box  = people[p]['BoundingBox']
        h    = np.around(box['Height'], 3)
        w    = np.around(box['Width'], 3)

        top_pos  = box['Top']
        left_pos = box['Left']
        row = int(top_pos / row_height)  
        col = int(left_pos / col_width)

        pxLeft   = int(imgWidth  * left_pos)
        pxTop    = int(imgHeight * top_pos)
        pxHeight = int(imgHeight * h)
        pxWidth  = int(imgWidth  * w)
        pxAspect_ratio = pxHeight / pxWidth

        text = '{}'.format(p)
        
        # Use a different color for STANDING people, to help with algo debugging
        if sd.likely_standing(conf, pxHeight, pxWidth):
            text_color = STANDING_COLOR
            
            same_grid0 = (ref0[0] == row) and (ref0[1] == col)
            same_ht0   = (ref0_ht == h)
            same_grid1 = (ref1[0] == row) and (ref1[1] == col)
            same_ht1   = (ref1_ht == h)

            if (same_grid0 and same_ht0) or (same_grid1 and same_ht1):
#                print(f'{p}) IS REF --        [{row},{col}]:{h:.2f},w:{w:.2f},[{prox[4]},{prox[5]}]')
                text = '{}**'.format(p)
#            else:
#                print(f'{p}) STND, NOT REF -- [{row},{col}]:{h:.2f},w:{w:.2f},[{prox[4]},{prox[5]}]')
        else:
#            print(    f'{p}) NOT STND --      [{row},{col}]:{h:.2f},w:{w:.2f},aspX:{pxAspect_ratio:.2f},minAsp:{sd.MIN_STANDING_PERSON_ASPECT_RATIO},[{prox[4]},{prox[5]}]')
            text_color = bbox_color

        # cv2.putText(frame, text, (pxLeft + 5, pxTop + 20), 
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, #SIMPLEX, 
        #             1, BBOX_NUMBER_COLOR, 1, cv2.LINE_AA)
        
        cv2.putText(frame, '{}'.format(str(np.round(conf,2))), (pxLeft + 5, pxTop + 20), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, #SIMPLEX, 
                    1, BBOX_NUMBER_COLOR, 1, cv2.LINE_AA)
        
        # if ref0_ht > 0:
        #     cv2.putText(frame, f'h:{h:.2f}, g:[{row},{col}], r0:{ref0_ht:.2f}, r1:{ref1_ht:.2f}', (pxLeft - 40, pxTop - 10), 
        #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, #SIMPLEX, 
        #                 1, bbox_color, 1, cv2.LINE_AA)
        # else:
        #     cv2.putText(frame, f'h:{h:.2f}, g:[{row},{col}]', (pxLeft - 20, pxTop - 10), 
        #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, #SIMPLEX, 
        #                 1, bbox_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (pxLeft, pxTop), (pxLeft + pxWidth, pxTop + pxHeight), bbox_color, 3)
        #cv2.imwrite('filename.jpg',frame)
    return 

def gen_best_size_mask(standing_people, image_shape, size_mask_shape):
    #print('Entering gen_best_size_mask')
    grid_rows     = int(size_mask_shape[0])
    grid_cols     = int(size_mask_shape[1])

    num_standing  = len(standing_people)
    sorted_people = sort_people_by_ascending_height(standing_people)

    big_idx   = num_standing - 1
    small_idx = 0
    mid_idx = big_idx // 2

    smallest_height = sorted_people[small_idx]['BoundingBox']['Height']
    biggest_height  = sorted_people[big_idx  ]['BoundingBox']['Height']

    best_rmse      = sd.MAX_DISTANCE
    best_size_mask = []
    camera_config  = {'CameraConfig': {}}
    num_masks_generated = 0

    if np.abs(biggest_height - smallest_height) <= sd.MIN_DIFF_THRESHOLD:
        print('Need to pick values that are at least {} apart. Instead used {}, {}.'.format(sd.MIN_DIFF_THRESHOLD,smallest_height,biggest_height))
        camera_config['CameraLeft']   = 0.1
        camera_config['CameraHeight'] = 20
        camera_config['SizeMask']     = []
    else:
        ignore_same_cell   = 0
        ignore_same_height = 0

        best_cam_left   = -1
        best_cam_height = 20
        best_sm_pos     = []
        best_big_pos    = []
        best_sm_height  = 0
        best_big_height = 0
        best_big_ref    = -1
        best_sm_ref     = -1

        # try all the standing reference people, using all combinations of smaller heights to bigger heights
        for big_ref in range(num_standing - 1, mid_idx, -1):
            big_pos    = sorted_people[big_ref]['GridPos']
            big_height = sorted_people[big_ref]['BoundingBox']['Height']
            #print('Using biggest height: {} at {}'.format(big_height, big_pos))

            # for a given big height, try all the small heights up to but not including the current big ref
            for sm_ref in range(0, big_ref - 1, 1):
                sm_pos    = sorted_people[sm_ref]['GridPos']
                sm_height = sorted_people[sm_ref]['BoundingBox']['Height']

                if sm_pos[0] == big_pos[0] and sm_pos[1] == big_pos[1]:
                    ignore_same_cell += 1
                    #print('ignoring option of 2 people in same grid cell {}'.format(sm_pos))
                elif np.abs(big_height - sm_height) <= sd.MIN_DIFF_THRESHOLD:
                    ignore_same_height += 1
                    #print('ignoring option of 2 people with heights too close ({} apart).'.format(sd.MIN_DIFF_THRESHOLD))
                else:
                    # for now, ignore cam height. consider as an extension.
                    #for cam_height in range(15, 30, 5): 
                    cam_height = 20

                    # guaranteed that big height is bigger than small height
                    for cam_step in range(5, 95, 5):
                        cam_from_left = cam_step / 100
                        #print('Cam: {}'.format(cam_step))

                        # Ensure that no one BIGGER is CLOSER. This helps eliminate cases where a bounding box
                        # is cutting off the bottom of a person, or they are bent over or sitting but still in an
                        # aspect ratio that passes for standing up.

                        sm_smaller_than_further = False
                        big_smaller_than_further = False
                        bad_sm_height = -1 
                        bad_big_height = -1

                        # First gather the distance from the camera for each "standing" ref person
                        all_dist_from_cam = []
                        for j in range(len(sorted_people)):
                            d = sd.dist_from_camera_by_exact_coords(cam_from_left, cam_height, 
                                                                    [sorted_people[j]['BoundingBox']['Top'],
                                                                     sorted_people[j]['BoundingBox']['Left']])
                            all_dist_from_cam.append(d)

                        # now ensure the current small ref is not BIGGER than anyone closer.
                        # walk up the list of sorted refs, and see if any distance is larger (further away from cam).
                        sm_dist = all_dist_from_cam[sm_ref]
                        for k in range(sm_ref, num_standing - 1, 1):
                            if all_dist_from_cam[k] > sm_dist:
                                bad_sm_height = k
                                sm_smaller_than_further = True
                                #print('ELIM sm:  {}, sm  h: {}, sm  d: {}, bad_k: {}'.format(sm_ref,sm_height,sm_dist,bad_sm_height))
                                break

                        # now ensure the current big ref is not BIGGER than anyone closer.
                        # walk up the list of sorted refs, and see if any distance is larger (further away from cam).
                        #print('big ref is {}'.format(big_ref))
                        big_dist = all_dist_from_cam[big_ref]
                        #print('big dist is {}'.format(big_dist))
                        for k in range(big_ref, num_standing - 1, 1):
                            # it can't be true that someone BIGGER is CLOSER. confirm for all bigger refs.
                            #print('k is {}'.format(k))
                            if all_dist_from_cam[k] > big_dist:
                                bad_big_height = k
                                big_smaller_than_further = True
                                #print('ELIM big: {}, big h: {}, big d: {}, bad_k: {}'.format(big_ref,sm_height,sm_dist,bad_big_height))
                                break
                        
                        #print('out of the big ref loop')
                        if  not (big_smaller_than_further or sm_smaller_than_further):
                            #print('entering sd.make_gaussian_mask')
                            #print(type(size_mask_shape))
                            
                            size_mask = sd.make_gaussian_mask(cam_from_left, cam_height, 
                                                              size_mask_shape, sm_pos, sm_height,
                                                              big_pos, big_height).tolist()
                            
                            rmse = eval_size_mask2(size_mask, standing_people)
                            num_masks_generated += 1

                            if rmse < best_rmse:
                                best_rmse       = rmse
                                best_cam_left   = cam_from_left
                                best_cam_height = cam_height
                                best_size_mask  = size_mask
                                best_sm_pos     = sm_pos 
                                best_big_pos    = big_pos
                                best_sm_ref     = sm_ref
                                best_big_ref    = big_ref
                                
                            #print('Best rmse: {}, at camera: {}'.format(best_rmse,best_cam))

            if num_masks_generated > 0:
                best_big_height = sorted_people[best_big_ref]['BoundingBox']['Height']
                best_sm_height  = sorted_people[best_sm_ref ]['BoundingBox']['Height']
                best_big_width  = sorted_people[best_big_ref]['BoundingBox']['Width']
                best_sm_width   = sorted_people[best_sm_ref ]['BoundingBox']['Width']
                best_big_image  = sorted_people[best_big_ref]['FromImage']
                best_sm_image   = sorted_people[best_sm_ref ]['FromImage']

                camera_config = {'CameraConfig': {}}
                camera_config['CameraLeft']   = best_cam_left
                camera_config['CameraHeight'] = best_cam_height
                camera_config['SizeMask']     = best_size_mask

                final_ref_grids = [{'GridPos': best_sm_pos,  
                                    'HeightImageRatio': np.around(best_sm_height, 3),
                                    'HeightActualFeet': sd.NORMALIZE_NUM_FEET,
                                    'WidthImageRatio' : np.around(best_sm_width, 3),
                                    'AspectRatio'     : sorted_people[best_sm_ref]['AspectRatio'],
                                    'FromImage'       : sorted_people[best_sm_ref]['FromImage'],
                                    'BoundingBox'     : sorted_people[best_sm_ref]['BoundingBox']},
                                   {'GridPos': best_big_pos, 
                                    'HeightImageRatio': np.around(best_big_height, 3),
                                    'HeightActualFeet': sd.NORMALIZE_NUM_FEET,
                                    'WidthImageRatio' : np.around(best_big_width, 3),
                                    'AspectRatio'     : sorted_people[best_big_ref]['AspectRatio'],
                                    'FromImage'       : sorted_people[best_big_ref]['FromImage'],
                                    'BoundingBox'     : sorted_people[best_big_ref]['BoundingBox']}
                                    ]
                camera_config['MaskReferenceSizes'] = final_ref_grids
                #print('End Reached')
                #print('Generated {num_masks_generated} masks. Best RMSE: {best_rmse}. camLeft: {best_cam_left}, camHt: {best_cam_height}, numStanding: {num_standing}, ref1:{best_sm_pos},{best_sm_image}, ref2:{best_big_pos},{best_big_image}')

    return camera_config, best_rmse, num_masks_generated

def min_distance_from_list(proximity_list):
    min_distance = sd.MAX_DISTANCE
    for p in proximity_list:
        if p[2] < min_distance:
            min_distance = p[2]
    return min_distance 

def get_num_unsafe_pairs(proximity_list):
    num_unsafe_pairs = 0
    for p in proximity_list:
        if p[2] < MIN_SAFE_DISTANCE:
            num_unsafe_pairs += 1
    return num_unsafe_pairs
