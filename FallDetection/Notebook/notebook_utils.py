import cv2
import numpy as np


def preprocess(img, output_size=(512, 512), resize=True):
    """
    Normalize pixel values and resize image
    :param img: input
    :param output_size: img size to be fed for model
    :param resize: boolean flag for resize
    :return:
    """
    resized = cv2.resize(img, (output_size[1], output_size[0])) if resize else img

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
    return x1, resized


def upscale_bbox_fn(bbox, img, scale=1.25):
    """
    Increase the size of bbox by "scale" value
    for better keypoint detection
    :param bbox:
    :param img:
    :param scale:
    :return:
    """

    new_bbox = []
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    w = (x1 - x0) / 2
    h = (y1 - y0) / 2
    center = [x0 + w, y0 + h]
    new_x0 = max(center[0] - w * scale, 0)
    new_y0 = max(center[1] - h * scale, 0)
    new_x1 = min(center[0] + w * scale, img.shape[1])
    new_y1 = min(center[1] + h * scale, img.shape[0])
    new_bbox = [new_x0, new_y0, new_x1, new_y1]
    return new_bbox


def crop_resize_normalize(img, bbox_list, output_size,
                          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    crop bbox from image and resize to the dimensions
    required by pose estimation model
    :param img:
    :param bbox_list:
    :param output_size:
    :param mean:
    :param std:
    :return:
    """
    output_list = []

    for bbox in bbox_list:
        x0 = max(int(bbox[0]), 0)
        y0 = max(int(bbox[1]), 0)
        x1 = min(int(bbox[2]), int(img.shape[1]))
        y1 = min(int(bbox[3]), int(img.shape[0]))
        w = x1 - x0
        h = y1 - y0
        res_img = cv2.resize(img[y0:y0 + h, x0:x0 + w, :], (output_size[1], output_size[0]))
        res_img, _ = preprocess(res_img, resize=False)
        output_list.append(res_img)
    output_array = np.vstack(output_list)
    return output_array


def detector_to_simple_pose(img, class_ids, scores, bounding_boxs,
                            output_shape=(256, 192), scale=1.25, thr=0.5,
                             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), person_id=0):
    """
    Convert original img for pose estimation input based on bbox predictions
    :param img:
    :param class_ids: bbox classes
    :param scores: bbox confidence
    :param bounding_boxs: (left, top, bottom, right predictions)
    :param output_shape: shape of pose model
    :param scale: crop bbox*scale from img
    :param thr: bbox confidence score threshold
    :param mean: normalize mean
    :param std: normalize std
    :return: pose_input, upscale_bbox
    """
    L = class_ids.shape[1]
    upscale_bbox = []
    for i in range(L):
        if class_ids[0][i] != person_id:
            continue
        if scores[0][i] < thr:
            continue
        bbox = bounding_boxs[0][i]
        upscale_bbox.append(upscale_bbox_fn(bbox.tolist(), img, scale=scale))
    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(img, upscale_bbox, output_shape, mean=mean, std=std)
    else:
        pose_input = None
    return pose_input, upscale_bbox


def get_max_pred(batch_heatmaps):
    """
    Process pose estimation output
    :param batch_heatmaps:pose heatmap
    :return: keypoint coordinates and confidence scores
    """
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def heatmap_to_coord(heatmaps, bbox_list):
    """
    Convert heatmap to keypoint coordinates
    """
    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]
    coords, maxvals = get_max_pred(heatmaps)
    preds = np.zeros_like(coords)

    for i, bbox in enumerate(bbox_list):
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        w = (x1 - x0) / 2
        h = (y1 - y0) / 2
        center = np.array([x0 + w, y0 + h])
        scale = np.array([w, h])

        w_ratio = coords[i][:, 0] / heatmap_width
        h_ratio = coords[i][:, 1] / heatmap_height
        preds[i][:, 0] = scale[0] * 2 * w_ratio + center[0] - scale[0]
        preds[i][:, 1] = scale[1] * 2 * h_ratio + center[1] - scale[1]
    return preds, maxvals


def reset_tracker():
    """
    Initialize keypoint trackers for x,y coordinates
    """
    xpart_tracker = {'eyes': np.array([]), 'hips': np.array([]), 'anks': np.array([]), 'shdr': np.array([]),
                     'anks-shdr': np.array([])}
    ypart_tracker = {'eyes': np.array([]), 'hips': np.array([]), 'anks': np.array([]), 'shdr': np.array([]),
                     'anks-shdr': np.array([])}

    return (xpart_tracker, ypart_tracker)


def reset_counts():
    """
    Change fall counter variables
    :return:
    """
    return (0, -1, 0, 0)


def fall_detection(ypart_tracker, anks_shdr_dist, dist_hist=50, dist_count=5):
    """
    :param ypart_tracker: y points tracker
    :param anks_shdr_dist: threshold for distance
    :param dist_hist: number of points in history to consider
    :param dist_count: threshold for number of occurrences of low anks-shdr distance
    :return: Fall result
    """
    dist_cndt = np.sum(ypart_tracker['anks-shdr'][-dist_hist:] <= anks_shdr_dist)

    if dist_cndt > dist_count:
        return True
    else:
        return False

# Keypoint coordinates chosen for Simple Pose model.
# Check keypoint mapping for the model of choice.

def update_x(pred_coords, xpart_tracker, history=1500):
    """
    Update x-tracker and store last 1500 detection.
    :param pred_coords: x coordinates
    :param xpart_tracker: choose keypoints based on input conditions
    :return: x-tracker
    """

    anks_val = (pred_coords[15] + pred_coords[16]) * 0.5
    shdr_val = (pred_coords[5] + pred_coords[6]) * 0.5

    xpart_tracker['anks'] = np.append(xpart_tracker['anks'], [anks_val], axis=0)
    xpart_tracker['shdr'] = np.append(xpart_tracker['shdr'], [shdr_val], axis=0)
    xpart_tracker['anks-shdr'] = np.append(xpart_tracker['anks-shdr'], [anks_val - shdr_val], axis=0)

    xpart_tracker = {k: v[-history:] for k, v in xpart_tracker.items()}

    return xpart_tracker


def update_y(pred_coords, ypart_tracker, history=1500):
    """
    Update y-tracker and store last 1500 detection
    :param pred_coords: y coordinates
    :param ypart_tracker: choose keypoints based on input conditions
    :return: y-tracker
    """
    anks_val = (pred_coords[15] + pred_coords[16]) * 0.5
    shdr_val = (pred_coords[5] + pred_coords[6]) * 0.5

    ypart_tracker['anks'] = np.append(ypart_tracker['anks'], [anks_val], axis=0)
    ypart_tracker['shdr'] = np.append(ypart_tracker['shdr'], [shdr_val], axis=0)
    ypart_tracker['anks-shdr'] = np.append(ypart_tracker['anks-shdr'], [anks_val - shdr_val], axis=0)

    ypart_tracker = {k: v[-history:] for k, v in ypart_tracker.items()}

    return ypart_tracker