from __future__ import division
import math
import numpy as np

CONF_THRESHOLD = 10 # 48 # pct  68

MIN_LIKELY_PERSON_ASPECT_RATIO   = 1.0
MAX_LIKELY_PERSON_ASPECT_RATIO   = 4.0

MIN_STANDING_PERSON_ASPECT_RATIO = 1.6 # 1.9
MAX_STANDING_PERSON_ASPECT_RATIO = MAX_LIKELY_PERSON_ASPECT_RATIO

MASK_GRANULARITY_FACTOR    = 2

HEIGHT_ADJ_FACTOR          = 0.5
HEIGHT_ADJ_MIN_ABS_SLOPE   = 0.5

MAX_DISTANCE               = 999999.0
MIN_LIKELY_PERSON_AREA     = 0.0015
MIN_LIKELY_PERSON_AREA_PX  = 615
CAM_FROM_TOP               = 1 # always bottom edge of pic
MIN_DIFF_THRESHOLD         = 0.03
NORMALIZE_NUM_FEET         = 6   # each unit in the size mask represents this many feet

def slope_between_two_points(point1, point2):
    # calculate the slope, being sure not to divide by zero
    t1 = point1[0]; l1 = point1[1]
    t2 = point2[0]; l2 = point2[1]
    
    t_dist = abs(t2 - t1); l_dist = abs(l2 - l1)

    if l_dist == 0:
        slope = 0
    else:
        slope = t_dist / l_dist 

    return slope

def likely_standing(conf, pxHeight, pxWidth):
    aspect_ratio = pxHeight / pxWidth

    if ((aspect_ratio > MIN_STANDING_PERSON_ASPECT_RATIO) and 
        (aspect_ratio < MAX_STANDING_PERSON_ASPECT_RATIO)):
        good_person_rectangle = True
    else:
        good_person_rectangle = False

    total_bbox_area_px = pxHeight * pxWidth
#    print(f'  px area:{total_bbox_area_px:.2f}, pxH:{pxHeight:.2f}, pxW:{pxWidth:.2f}')
    
    if total_bbox_area_px > MIN_LIKELY_PERSON_AREA_PX:
        big_enough = True
    else:
        big_enough = False

    is_standing_person = (conf > CONF_THRESHOLD) and (big_enough) and (good_person_rectangle)
    
    return is_standing_person

def likely_a_person(conf, h, w, image_shape,
                    min_aspect_ratio=MIN_LIKELY_PERSON_ASPECT_RATIO, 
                    max_aspect_ratio=MAX_LIKELY_PERSON_ASPECT_RATIO):
    pxHeight = h * image_shape[1]
    pxWidth  = w * image_shape[0]

    aspect_ratio = pxHeight / pxWidth
    if (aspect_ratio > min_aspect_ratio) and (aspect_ratio < max_aspect_ratio):
        good_person_rectangle = True
    else:
        good_person_rectangle = False

#    total_bbox_area = h * w
#    if total_bbox_area > MIN_LIKELY_PERSON_AREA:

    total_bbox_area_px = pxHeight * pxWidth
#    print(f'  px area:{total_bbox_area_px:.2f}, pxH:{pxHeight:.2f}, pxW:{pxWidth:.2f}')
    if total_bbox_area_px > MIN_LIKELY_PERSON_AREA_PX:
        big_enough = True
    else:
        big_enough = False

    if (conf > CONF_THRESHOLD) and (big_enough) and (good_person_rectangle):
        return True
    else:
#        print(f'REJECT - Conf: {conf:.2f}, w/h: {w:.2f}/{h:.2f}, bbox area: {total_bbox_area:.3f}, big enough: {big_enough}, good rect: {good_person_rectangle}, asp: {h/w:.1f}')
        return False

def keep_likely_people(p, image_shape):
    likely_people = []
    for i in range(len(p)):
        person = p[i]
        conf   = person['Confidence']
        bb     = person['BoundingBox']

        if likely_a_person(conf, bb['Height'], bb['Width'], image_shape):
            likely_people.append(person)
    return likely_people

def adjusted_distance_between_two_points(pos1, pos2, size_mask, cam_height, verbose):
    if verbose:
        #print(f'  start: {pos1}, end: {pos2}')
        print('start {}, end {}'.format(pos1, pos2))

    img_portion_dist = euclidian_distance_between_two_points(pos1, pos2)

    t1 = pos1[0]; l1 = pos1[1]
    t2 = pos2[0]; l2 = pos2[1]
    
    # convert from img portion distance to feet by using factors from the size mask.
    # these size factors take into account the camera left position as well as the camera height
    row_height = 1 / size_mask.shape[0] 
    col_width  = 1 / size_mask.shape[1]

    which_row1 = int(t1 / row_height)  
    which_col1 = int(l1 / col_width)
    size1      = size_mask[which_row1, which_col1]

    which_row2 = int(t2 / row_height)  
    which_col2 = int(l2 / col_width)
    size2      = size_mask[which_row2, which_col2]

    # use the minimum mask between src and dst in the grid
    baseline = min(size1, size2)
    dist_ft  = img_portion_dist / baseline * NORMALIZE_NUM_FEET

    # make adjustment for height of camera. size mask calculations don't fully account
    # for longer distances between persons when one person is above another.
    slope = slope_between_two_points(pos1, pos2)
    if abs(slope) > HEIGHT_ADJ_MIN_ABS_SLOPE:
        dist_ft = dist_ft * (1 + HEIGHT_ADJ_FACTOR)
    
    if verbose:
        #print(f'  Dist: {dist_ft:.2f}, start mk: {size1:.2f}, end mk: {size2:.2f}, img dist: {img_portion_dist:.2f}')
        print('This is verbose')

    return dist_ft, img_portion_dist, abs(slope)

def distance_from_closest_person(which_person, people, size_mask, cam_height, verbose):
    start_pos = [people[which_person]['BoundingBox']['Top'],
                 people[which_person]['BoundingBox']['Left']]

    dist = MAX_DISTANCE
    img_portion_dist = MAX_DISTANCE
    closest_person = -1
    slope = 0
    
    for p in range(len(people)):
        if not p == which_person:
            end_pos = [people[p]['BoundingBox']['Top'],
                       people[p]['BoundingBox']['Left']]
            this_dist_ft, this_img_portion_dist, this_slope = \
                    adjusted_distance_between_two_points(start_pos, end_pos, size_mask, cam_height, verbose)
            if verbose:
                #print(f'     {which_person} to {p} is {this_dist_ft:.2f} ft, img d: {this_img_portion_dist:.2f}')
                print('This is verbose from distance_from_closest_person')
            if this_dist_ft < dist:
                dist = this_dist_ft
                img_portion_dist = this_img_portion_dist
                closest_person = p
                slope = this_slope
    return dist, img_portion_dist, closest_person, slope

def detect_distances(people, size_mask, image_shape, cam_height, verbose):
    likely_people = keep_likely_people(people, image_shape)

    row_height = 1 / (size_mask.shape[0])
    col_width  = 1 / (size_mask.shape[1])

    proximity_list = []

#    if len(likely_people) < 2:
#         proximity_list.append([0, 0, MAX_DISTANCE])
#    else:
    for which_person in range(len(likely_people)):
        person   = likely_people[which_person]
        top_pos  = person['BoundingBox']['Top']
        left_pos = person['BoundingBox']['Left']
        height   = person['BoundingBox']['Height']
        width    = person['BoundingBox']['Width']
        conf     = person['Confidence']
        row = int(top_pos  / row_height)  
        col = int(left_pos / col_width)

        pxHeight = height * image_shape[1]
        pxWidth  = width  * image_shape[0]
        pxAsp    = pxHeight / pxWidth

        if len(likely_people) >=2:
            distance, img_portion_dist, closest_person, slope = \
                    distance_from_closest_person(which_person, likely_people, 
                                                 size_mask, cam_height, verbose)
        else:
            distance = MAX_DISTANCE ; img_portion_dist = 0 ; closest_person = 0; slope = 0
            
        proximity_list.append([which_person, closest_person, distance, height, 
                               row, col, size_mask[row, col],
                               top_pos, left_pos, conf, pxAsp, 
                               img_portion_dist, width, slope])
        which_person += 1
    return likely_people, proximity_list

def dist_from_camera(cam_from_left, cam_height, grid_shape, r, c):
    # zero based r and c
    # calc distance from exact camera position to the top left corner of the target grid cell
    top_left = [(r / (grid_shape[0] - 1)),
                (c / (grid_shape[1] - 1))]
    return dist_from_camera_by_exact_coords(cam_from_left, cam_height, top_left)

def dist_from_camera_by_exact_coords(cam_from_left, cam_height, top_left):
    return euclidian_distance_between_two_points([CAM_FROM_TOP, cam_from_left], top_left)
#    return height_adjusted_distance_between_two_points(cam_height, [CAM_FROM_TOP, cam_from_left], top_left)

def height_adjusted_distance_between_two_points(cam_height, point1, point2):
    base_dist = euclidian_distance_between_two_points(point1, point2)
    
    # calculate the slope, being sure not to divide by zero
    t1 = point1[0]; l1 = point1[1]
    t2 = point2[0]; l2 = point2[1]
    
    t_dist = abs(t2 - t1); l_dist = abs(l2 - l1)

    if l_dist == 0:
        slope = 0
    else:
        slope = t_dist / l_dist 
    
    # make adjustment for height of camera. size mask calculations don't fully account
    # for longer distances between persons when one person is above another.
    if abs(slope) > HEIGHT_ADJ_MIN_ABS_SLOPE:
        base_dist = base_dist * (1 - HEIGHT_ADJ_DISCOUNT_FACTOR)

    return base_dist

def euclidian_distance_between_two_points(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + 
                     (point1[1] - point2[1])**2)

def find_ampl_beta(cam_from_left, cam_height, grid_shape, pos1, val1, pos2, val2):
    radius1 = dist_from_camera(cam_from_left, cam_height, grid_shape, pos1[0], pos1[1])
    radius2 = dist_from_camera(cam_from_left, cam_height, grid_shape, pos2[0], pos2[1])

    # avoid divide by 0 warning at runtime
    r_squared_diff = (radius1**2 - radius2**2)
    if r_squared_diff == 0:
        r_squared_diff = 0.001
    beta = -np.log(val1 / val2) / r_squared_diff
    ampl = val1 / np.exp(-beta * radius1**2)
    return ampl, beta

def make_gaussian_mask(cam_from_left, cam_height, grid_shape, pos1, val1, pos2, val2):
    print('grid shape is {}'.format(grid_shape))
    print('grid shape type is {}'.format(type(grid_shape)))
    
    
    
    mask = np.zeros([int(grid_shape[0]), int(grid_shape[1])], dtype = float)
    print('make_gaussian_mask mask is {}'.format(mask))
    ampl, beta = find_ampl_beta(cam_from_left, cam_height, grid_shape, pos1, val1, pos2, val2)
    print('make_gaussian_mask ampl is {}'.format(ampl))
    print('make_gaussian_mask beta is {}'.format(beta))
    
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            dist = dist_from_camera(cam_from_left, cam_height, grid_shape, r, c)
            gaussian_val = ampl * np.exp(-beta * (dist**2))
            mask[r, c] = np.around(gaussian_val, 3)
    return mask
