from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
# Image processing
import cv2
# lambda helpers
from notebook_utils import preprocess, detector_to_simple_pose, heatmap_to_coord, reset_tracker, update_x, update_y
# Model libraries
import mxnet as mx
from gluoncv import model_zoo, data, utils

def to_np(mxnet_array):
    return mxnet_array.asnumpy()


def to_mx(np_array):
    return mx.nd.array(np_array)


def init_models(det_name, pose_name):
    """
    Return gluoncv models from model zoo
    :param det_name: object detection model name
    :param pose_name: pose estimation model name
    :return: 
    """
    detector = model_zoo.get_model(det_name, pretrained=True)
    pose_net = model_zoo.get_model(pose_name, pretrained=True)
    detector.reset_class(["person"], reuse_weights=['person'])

    return detector, pose_net


def get_video_stats(videopath, verbose=True):
    """
    Return video characteristics
    :param videopath: 
    :param verbose: print stats
    :return: 
    """
    vidcap = cv2.VideoCapture(videopath)
    ret, frame = vidcap.read()

    video_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    video_time = video_len / (video_fps * 60)
    video_mins = int(video_time)
    video_secs = int((video_time - int(video_time)) * 60)

    if verbose:
        print("Total Frames & FPS : ", video_len, ", ", round(video_fps))
        print("Video Duration     : ", video_mins, "mins ", video_secs, "secs")
        print("Video Resolution : ", frame.shape)

    return {'total_frames': video_len, 'fps': video_fps,
            'duration': str(video_mins) + ' mins ' + str(video_secs) + ' secs',
            'resolution': frame.shape}


def get_frames(video_name, frame_nums):
    """
    Sample frames based on frame number 
    :param video_name: 
    :param frame_nums: frame numbers to filter from video
    :return: list of frames
    """
    frames = []
    cap = cv2.VideoCapture(video_name)  # video_name is the video being called

    for frame_num in frame_nums:
        cap.set(1, frame_num);  # Where frame_no is the frame you want
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return frames


def get_predictions(img_frame, detector_model, pose_model, img_size=(512, 512), person_id=0, conf_thresh=0.1,
                    box_size_thresh=(70, 70), visualize=True):
    """
    Predict keypoints in image and show image
    :param img_frame: 
    :param detector_model: person detector model
    :param pose_model: pose estimation model
    :param img_size: detector model input size
    :param person_id: person class for detector
    :param conf_thresh: threshold for valid detection
    :param box_size_thresh: min size for bbox
    :param visualize: print key points
    :return: 
    """
    x, orig_img = preprocess(img_frame, img_size)
    x, orig_img = mx.nd.array(x), mx.nd.array(orig_img)
    class_ids, scores, bboxes = detector_model(x)

    # bbox coordinates of top prediction
    x_min, y_min, x_max, y_max = bboxes[0][0][0], bboxes[0][0][1], bboxes[0][0][2], bboxes[0][0][3]
    w, h = (x_max - x_min).asscalar(), (y_max - y_min).asscalar()

    # do pose estimation only for valid predictions
    if (scores[:, 0:1, :][0][0].asscalar() > conf_thresh) and w > box_size_thresh[0] and h > box_size_thresh[1]:

        orig_img, class_ids, scores, bboxes = to_np(orig_img), to_np(class_ids), to_np(scores), to_np(bboxes)
        
        pose_input, upscale_bbox = detector_to_simple_pose(orig_img, class_ids[:, 0:1, :], scores[:, 0:1, :],
                                                           bboxes[:, 0:1, :], thr=conf_thresh, person_id=person_id)

        pose_input, upscale_bbox = to_mx(pose_input), to_mx(upscale_bbox)

        # Get keypoint heatmap predictions
        predicted_heatmap = pose_model(pose_input)
        predicted_heatmap, upscale_bbox = to_np(predicted_heatmap), to_np(upscale_bbox)

        # Convert the heat map to (x,y) coordinate space in the frame and return confidence scores for 17 keypoints
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        xpart_tracker, ypart_tracker = reset_tracker()
        xpart_tracker, ypart_tracker = update_x(pred_coords[0][:, 0], xpart_tracker), update_y(pred_coords[0][:, 1],
                                                                                               ypart_tracker)

        print('Ankle shoulder distance : ', round(ypart_tracker['anks-shdr'][-1], 3))

        if visualize:
            ax = utils.viz.plot_keypoints(orig_img, pred_coords, confidence,
                                          class_ids[:, 0:1, :], bboxes[:, 0:1, :], scores[:, 0:1, :],
                                          box_thresh=conf_thresh, keypoint_thresh=0)

            plt.show()