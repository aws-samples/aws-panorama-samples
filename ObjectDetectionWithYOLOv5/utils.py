import time
import math
import random as rnd

import cv2
import numpy as np


class Processor():
    def __init__(self, class_names, input_size=640, keep_ratio=False):
        """Constructor
        
        Args:
            class_names (list(str)): class names of the model
            input_size (int): model's input size (actual input is of shape [1, 3, input_size, input_size])
            keep_ratio (bool): if True then non-square input images will be made square prior to being resized to input size
        """
        self.input_size = input_size
        self.class_names = class_names
        self.class_count = len(class_names)
        self.feature_count = self.class_count + 5 # outputs per anchor (bbox, confidence and class probabilities)
        self.keep_ratio = keep_ratio
        
        self.strides = np.array([8., 16., 32.])
        self.output_shapes = []
        for s in self.strides:
            d = int(self.input_size / s)
            self.output_shapes.append((1, 3, d, d, self.feature_count))
        
        # Source: https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml
        anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ])
        
        # Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        self.nl = len(anchors)
        self.na = len(anchors[0])
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)
        
        self.colors = [[rnd.randint(0, 255) for _ in range(3)] for _ in range(self.class_count)]

    def preprocess(self, image):
 
        pad_y, pad_x = 0, 0
        if self.keep_ratio:
            # Make the image square if needed by adding extra pads to
            # preserve the original height to width ratio after resize
            h, w = image.shape[:2]  
            if h != w:
                max_size = max(h, w)
                pad_y, pad_x = max_size - h, max_size - w
                img = np.zeros((max_size, max_size, 3), dtype="float32")
                img[0:max_size-pad_y, 0:max_size-pad_x] = image[:]
                image = img
        
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.
        
        # Convert to an expected shape
        img = img.transpose(2, 0, 1)  # to CHW
        img = np.expand_dims(img, axis=0).copy()  # Must use a copy here, otherwise it may not work as expected on Panorama
        
        # Alternatively, the last conversion can done using 
        # img = np.asarray([[img[:, :, c] for c in range(img.shape[-1])]])
 
        return img 
    
    def post_process(self, outputs, source_image_shape, conf_thres=0.5, image=None):
        """Transforms raw output into boxes, scores, classes. Applies NMS thresholding on 
        bounding boxes and scores
        Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        
        Args:
            outputs ([ny.array]): list of raw model outputs
            conf_thres (float): minimum confidence threshold
            image (np.array): if provided then detected bounding boxes will be drawn on this image
        Returns:
            boxes: normalised (i.e. values in the range [0, 1)) x1, y1, x2, y2 array of shape (-1, 4)
            scores: class * obj prob array (-1, 1) 
            cids: class ID values array (-1, 1)
        """
        scaled = []
        grids = []
        for out in outputs:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor
            out = out.reshape((1, -1, self.feature_count))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        boxes, scores, cids = self.nms(pred)
        
        # Normalise box coordinates to be in the range (0, 1]
        h, w = source_image_shape[:2]
        h1, w1 = h, w
        if self.keep_ratio and h != w:
            # Padding was used during pre-process to make the source image square
            h1 = w1 = max(h, w)
            
        y_scale = h1 / float(self.input_size) / h
        x_scale = w1 / float(self.input_size) / w
        boxes[:, 0] *= x_scale
        boxes[:, 1] *= y_scale
        boxes[:, 2] *= x_scale
        boxes[:, 3] *= y_scale
        boxes = np.clip(boxes, 0, 1)
        
        if image is not None:
            self.draw_cv2(image, boxes, scores, cids)

        return (boxes, scores, cids), image
    
    def make_grid(self, nx, ny):
        """Create scaling array based on box location
        Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        
        Args:
            nx: x-axis num boxes
            ny: y-axis num boxes
        Returns:
            grid: numpy array of shape (1, 1, nx, ny, 2)
        """
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)
    
    def non_max_suppression(self, boxes, scores, cids, iou_thres=0.6):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = scores.flatten().argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where( ovr <= iou_thres)[0]
            order = order[inds + 1]
        boxes = boxes[keep]
        scores = scores[keep]
        cids = cids[keep]
        return boxes, scores, cids

    def nms(self, pred, iou_thres=0.6):
        boxes = self.xywh2xyxy(pred[..., 0:4])
        scores = np.amax(pred[:, 5:], 1, keepdims=False)  # only highest score
        cids = np.argmax(pred[:, 5:], axis=-1)
        return self.non_max_suppression(boxes, scores, cids)

    def xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where x1y1=top-left, x2y2=bottom-right"""
        
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    def draw_cv2(self, image, boxes, scores, class_ids):
        """Draw the detection boxes on the given image"""
        
        h, w = image.shape[:2]
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
            
            # Box coordinates are normalised, convert to absolute and clip to image boundaries
            x1 = np.clip(int(x1 * w), 0, w-1)
            y1 = np.clip(int(y1 * h), 0, h-1)
            x2 = np.clip(int(x2 * w), 0, w-1)
            y2 = np.clip(int(y2 * h), 0, h-1)
            
            cid = int(cls_id)
            c = self.colors[cid]
            label = f'{self.class_names[cid]} {score:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), c, 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1, int(y1 * 0.95)), font, 1, c, 3)
        return image


def send_to_s3(img, s3_key, bucket):
    import boto3
    
    s3 = boto3.resource('s3')
    img_data = cv2.imencode('.png', img)[1].tostring()
    s3.Object(bucket, s3_key).put(Body=img_data, ContentType='image/PNG')
